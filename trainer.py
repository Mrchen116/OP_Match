import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
    save_checkpoint, ova_ent, \
    test, test_ood, exclude_dataset

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()


    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)    # vars得到属性名和属性值的字典
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch     # 2*随机水平翻转+裁剪，1*随机水平翻转
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak  # 翻转中心裁剪 翻转随机裁剪 翻转中心裁剪
    else:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak

    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)   # unlabeled_dataset_all 比 unlabeled_dataset 增强弱一点
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)          # 使用更弱的增强
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)
    unlabeled_all_iter = iter(unlabeled_trainloader_all)

    for epoch in range(args.start_epoch, args.epochs):
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            exclude_dataset(args, unlabeled_dataset, ema_model.ema)     # 筛选出所有被ova认为是inlier的，其他不参与打伪标签

            # 有all和没all两者区别是增强不同，这个是用来做fix match pseudo label train
            unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                               sampler = train_sampler(unlabeled_dataset),
                                               batch_size = args.batch_size * args.mu,
                                               num_workers = args.num_workers,
                                               drop_last = True)
            unlabeled_iter = iter(unlabeled_trainloader)

        # epoch 不是按照过一遍数据算，是按给定的eval_step
        for batch_idx in range(args.eval_step):
            ## Data loading

            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()      # 水平翻转随机裁剪，水平翻转
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            if epoch >= args.start_fix:
                try:
                    (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()          # 水平翻转随机裁剪 fixmatch强增强
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()  # 水平翻转随机裁剪 水平翻转随机裁剪
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)    # unlabeled 水平翻转随机裁剪 * 2
            inputs = torch.cat([inputs_x, inputs_x_s,                  # labeled 水平翻转 水平翻转随机裁剪
                                inputs_all], 0).to(args.device)        # 相当于b_size*2 + b_size*mu*2放到一个batch中
            targets_x = targets_x.to(args.device)       # labeled的类别
            ## Feed data
            logits, logits_open = model(inputs)
            logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)   # 取出unlabeled的两个增强的的ova输出

            ## Loss for labeled samples
            Lx = F.cross_entropy(logits[:2*b_size],
                                      targets_x.repeat(2), reduction='mean')   # 有标签数据做CEloss 注意：labeled数据都来自已知类
            Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))         # 式子(1)

            ## Open-set entropy minimization   式子(2)
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            ## Soft consistenty regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1 - logits_open_u2)**2, 1), 1))

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)  # 水平翻转随机裁剪 fixmatch强增强
                logits, logits_open_fix = model(inputs_ws)
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)    # 用弱增强打伪标签
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()     # 大于阈值的为True
                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())       # 更新阈值以上的概率

            else:
                L_fix = torch.zeros(1).to(args.device).mean()
            loss = Lx + Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + L_fix
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg        # fixmatch有标签数据的CE loss
            output_args["loss_o"] = losses_o.avg        # ova 有标签数据，式子(1)
            output_args["loss_oem"] = losses_oem.avg    # ova 无监督 熵最小化
            output_args["loss_socr"] = losses_socr.avg  # ova 拉近两个增强得到logit的L2距离
            output_args["loss_fix"] = losses_fix.avg    # fixmatch打伪标签，弱增强学习
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]


            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)   # 验证集
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id \
                = test(args, test_loader, test_model, epoch)                # 测试集

            for ood in ood_loaders.keys():
                roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_oem', losses_oem.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_fix', losses_fix.avg, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)       # Fixmatch伪标签阈值上的概率
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)    # 不考虑ova的输出的情况下，计算inlier的准确率
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.test_roc', test_roc, epoch)
            args.writer.add_scalar('test/4.test_error_rate', 100 - test_acc_close, epoch)

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk                # outlier的二分类准确率
                roc_valid = test_roc                # 判断是否是outlier的二分类roc
                roc_softm_valid = test_roc_softm    # 不考虑ova输出，所有的预测inlier类别的概率值取负数，算outlier的roc
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,       # 全程没p用
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Valid roc: {:.3f}'.format(roc_valid))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()
