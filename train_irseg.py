import os
import shutil
import json
import time

from apex import amp
import apex
import copy

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from toolbox.loss import MscCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger
from toolbox import load_ckpt
from toolbox.losses import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
    LovaszSoftmax, LDAMLoss, CCLoss1, KLLoss, ResizeCrossEntropyLoss, OhemCELoss
from toolbox.optimizer.SGD import SGD_GC, SGD_GCC
from torch.optim import lr_scheduler

torch.manual_seed(123)
cudnn.benchmark = True

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    # print(model)
    device = torch.device('cuda:0')
    model.to(device)
    if args.resume is not '':
        checkpoint_dict = torch.load(args.resume, map_location={'cuda:2': 'cuda:0'})
        model_dict = model.state_dict()
        backbone_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        need_remove_keys = []
        for key in backbone_dict.keys():
            if key.startswith('classifier'):
                need_remove_keys.append(key)
        for key in need_remove_keys:
            del backbone_dict[key]
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
        have_rate = len(backbone_dict.items()) / len(model_dict.items())
        print(f'loading pretrained model {have_rate} in {args.resume} successful ...')





    # dataloader
    trainset, *testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    val_loader = DataLoader(testset[-1], batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)

    params_list = model.parameters()
    # params_list = group_weight_decay(model)
    # cfg['lr_start'] = 1e-4
    optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    # optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    # optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], momentum=0.9, weight_decay=cfg['weight_decay'])
    # optimizer = SGD_GCC(params_list, lr=0.1, momentum=0.9, weight_decay=1e-4)

    # lr scheduling
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # PST900
    # lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
    #                                 iters_per_epoch=len(self.train_loader), power=0.9)

    classweight = ClassWeight(cfg['class_weight'])
    # class_weight = classweight.get_weight(train_loader, cfg['n_classes'])
    # print(class_weight)
    # class_weight = torch.from_numpy(class_weight).float().to(device)

    class_weight = torch.from_numpy(trainset.class_weight).float().to(device)
    # class_weight[cfg['id_unlabel']] = 0

    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    # criterion = LovaszSoftmax().to(device)
    # criterion = FocalLoss2d().to(device)
    # criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=False, aux_weight=0.4,
    #                                                 ignore_index=-1).to(device)
    # print(class_weight.size(), class_weight)
    # criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion_0 = nn.CrossEntropyLoss().to(device)
    # criterion_0 = nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion_1 = nn.CrossEntropyLoss().to(device)
    # criterion_1 = nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion_2 = nn.CrossEntropyLoss().to(device)
    # criterion_2 = nn.CrossEntropyLoss(weight=class_weight).to(device)

    criterion_z1 = LovaszSoftmax().to(device)
    criterion_z2 = LovaszSoftmax().to(device)
    criterion_z3 = LovaszSoftmax().to(device)
    criterion_z4 = LovaszSoftmax().to(device)
    # criterion_boundary = nn.BCEWithLogitsLoss().to(device)
    # criterion_mask = nn.BCEWithLogitsLoss().to(device)
    criterion_boundary = LovaszSoftmax().to(device)
    criterion_mask = LovaszSoftmax().to(device)
    criterion_att3 = CCLoss1()
    criterion_att4 = CCLoss1()
    # criterion = MscCrossEntropyLoss().to(device)
    # criterion = CrossEntropyLoss2dLabelSmooth(weight=class_weight).to(device)
    # score_thres = 0.7
    # n_min = 8 * 480 * 640 // 16
    # criterion_z1 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    # criterion_z2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    # criterion_z3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    # criterion_z4 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()

    out_train_loss_meter = averageMeter()
    out_val_loss_meter = averageMeter()
    R_train_loss_meter = averageMeter()
    R_val_loss_meter = averageMeter()
    T_train_loss_meter = averageMeter()
    T_val_loss_meter = averageMeter()
    Z1_train_loss_meter = averageMeter()
    Z1_val_loss_meter = averageMeter()
    Z2_train_loss_meter = averageMeter()
    Z2_val_loss_meter = averageMeter()
    Z3_train_loss_meter = averageMeter()
    Z3_val_loss_meter = averageMeter()
    Z4_train_loss_meter = averageMeter()
    Z4_val_loss_meter = averageMeter()
    mask_train_loss_meter = averageMeter()
    mask_val_loss_meter = averageMeter()
    bound_train_loss_meter = averageMeter()
    bound_val_loss_meter = averageMeter()
    att3_train_loss_meter = averageMeter()
    att3_val_loss_meter = averageMeter()
    att4_train_loss_meter = averageMeter()
    att4_val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'])#, ignore_index=cfg['id_unlabel'])

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    maxmiou = 0
    maxmAcc = 0
    for ep in range(cfg['epochs']):
        #########
        # lr_this_epo = cfg['lr_start'] * cfg['lr_decay'] ** (ep - 1)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_this_epo
        #########
        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                predict = model(image)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                mask = sample['mask'].to(device)#.unsqueeze(1)
                boundary = sample['boundary'].to(device)#.unsqueeze(1)
                attention_map = sample['attention_map'].to(device)#.unsqueeze(1)
                label = sample['label'].to(device)
                out, out_r, out_d, out_Z1, out_Z2, out_Z3, out_Z4, out_LV1, out_LV2, out_Att3, out_Att4 = model(image, depth)
            loss0 = criterion_0(out, label)
            loss1 = criterion_1(out_r, label)
            loss2 = criterion_2(out_d, label)
            loss_Z1 = criterion_z1(out_Z1, label)
            loss_Z2 = criterion_z2(out_Z2, label)
            loss_Z3 = criterion_z3(out_Z3, label)
            loss_Z4 = criterion_z4(out_Z4, label)
            # print(label.shape, mask.shape, boundary.shape)
            loss_mask = criterion_mask(out_LV2, mask)
            loss_bound = criterion_boundary(out_LV1, boundary)
            loss_Att3 = criterion_att3(out_Att3, attention_map)
            loss_Att4 = criterion_att4(out_Att4, attention_map)

            loss = loss0+loss1+loss2 + \
                   2 * loss_Z1 + loss_Z2 + loss_Z3 + loss_Z4 + \
                   loss_mask + loss_bound + loss_Att3 + loss_Att4

            # loss = F.cross_entropy(predict, label)
            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item())

            out_train_loss_meter.update(loss0.item())
            R_train_loss_meter.update(loss1.item())
            T_train_loss_meter.update(loss2.item())
            Z1_train_loss_meter.update(loss_Z1.item())
            Z2_train_loss_meter.update(loss_Z2.item())
            Z3_train_loss_meter.update(loss_Z3.item())
            Z4_train_loss_meter.update(loss_Z4.item())

            mask_train_loss_meter.update(loss_mask.item())
            bound_train_loss_meter.update(loss_bound.item())
            att3_train_loss_meter.update(loss_Att3.item())
            att4_train_loss_meter.update(loss_Att4.item())

        scheduler.step(ep)

        # val
        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()
            val_loss_meter.reset()
            for i, sample in enumerate(val_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    mask = sample['mask'].to(device)#.unsqueeze(1)
                    boundary = sample['boundary'].to(device)#.unsqueeze(1)
                    attention_map = sample['attention_map'].to(device)  # .unsqueeze(1)
                    label = sample['label'].to(device)
                    out, out_r, out_d, out_Z1, out_Z2, out_Z3, out_Z4, out_LV1, out_LV2, out_Att3, out_Att4 = model(
                        image, depth)
                # print(predict.size(),label.size())
                loss0 = criterion_0(out, label)
                loss1 = criterion_1(out_r, label)
                loss2 = criterion_2(out_d, label)
                loss_Z1 = criterion_z1(out_Z1, label)
                loss_Z2 = criterion_z2(out_Z2, label)
                loss_Z3 = criterion_z3(out_Z3, label)
                loss_Z4 = criterion_z4(out_Z4, label)
                # print(label.shape, mask.shape, boundary.shape)
                loss_mask = criterion_mask(out_LV2, mask)
                loss_bound = criterion_boundary(out_LV1, boundary)
                loss_Att3 = criterion_att3(out_Att3, attention_map)
                loss_Att4 = criterion_att4(out_Att4, attention_map)

                loss = loss0 + loss1 + loss2 + \
                       2 * loss_Z1 + loss_Z2 + loss_Z3 + loss_Z4 + \
                       loss_mask + loss_bound + loss_Att3 + loss_Att4
                # loss = F.cross_entropy(predict, label)
                val_loss_meter.update(loss.item())
                out_val_loss_meter.update(loss0.item())
                R_val_loss_meter.update(loss1.item())
                T_val_loss_meter.update(loss2.item())
                Z1_val_loss_meter.update(loss_Z1.item())
                Z2_val_loss_meter.update(loss_Z2.item())
                Z3_val_loss_meter.update(loss_Z3.item())
                Z4_val_loss_meter.update(loss_Z4.item())

                mask_val_loss_meter.update(loss_mask.item())
                bound_val_loss_meter.update(loss_bound.item())
                att3_val_loss_meter.update(loss_Att3.item())
                att4_val_loss_meter.update(loss_Att4.item())

                predict = out_Z1.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] lr={optimizer.param_groups[0]["lr"]:.12f} train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f}, '
                    f'mAcc={running_metrics_val.get_scores()[0]["class_acc: "]:.3f} '
                    f'miou={running_metrics_val.get_scores()[0]["mIou: "]:.3f} '
                    f'loss0={out_train_loss_meter.avg: .5f}/{out_val_loss_meter.avg: .5f} '
                    f'loss1= {R_train_loss_meter.avg: .5f}/{R_val_loss_meter.avg: .5f} '
                    f'loss2={T_train_loss_meter.avg: .5f}/{T_val_loss_meter.avg: .5f} '
                    f'loss_Z1={Z1_train_loss_meter.avg: .5f}/{Z1_val_loss_meter.avg: .5f} '
                    f'loss_Z2={Z2_train_loss_meter.avg: .5f}/{Z2_val_loss_meter.avg: .5f} '
                    f'loss_Z3={Z3_train_loss_meter.avg: .5f}/{Z3_val_loss_meter.avg: .5f} '
                    f'loss_Z4={Z4_train_loss_meter.avg: .5f}/{Z4_val_loss_meter.avg: .5f} '

                    f'loss_mask={mask_train_loss_meter.avg: .5f}/{mask_val_loss_meter.avg: .5f} '
                    f'loss_bound={bound_train_loss_meter.avg: .5f}/{bound_val_loss_meter.avg: .5f} '
                    f'loss_att3={att3_train_loss_meter.avg: .5f}/{att3_val_loss_meter.avg: .5f} '
                    f'loss_att4={att4_train_loss_meter.avg: .5f}/{att4_val_loss_meter.avg: .5f} '

                    )
        # mAcc = running_metrics_val.get_scores()[0]["class_acc: "]
        miou = running_metrics_val.get_scores()[0]["mIou: "]

        # if mAcc > maxmAcc:
            # maxmAcc = mAcc
            # save_ckpt(logdir, 'maxmAcc_', model)
        if miou > maxmiou:
            maxmiou = miou
            save_ckpt(logdir, model)
            # break


    print(round(maxmiou, 4))
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/configs/irseg.json", help="Configuration file to use")

    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbt'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")

    args = parser.parse_args()

    run(args)
