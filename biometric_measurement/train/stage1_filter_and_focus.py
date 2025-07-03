import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from asyncio.log import logger
import os
import json
import time
import math
import random
import shutil
import argparse
from turtle import forward
import warnings
import numpy as np
from enum import Enum
from PIL import Image
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DistributedSampler, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import logger as lg
from utils import config as cfg
from functools import partial
import pdb
import pickle

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--log-dir', default='../logs', type=str,
                    help='log path')
parser.add_argument('--csv_path_1', default='../logs', type=str,
                    help='csv path')
parser.add_argument('--csv_path_2', default='../logs', type=str,
                    help='csv path')
parser.add_argument('--group-name', default='test', type=str,
                    help='group name')
parser.add_argument('--exp-name', default='test', type=str,
                    help='experiment name')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--train_size', default=0.9, type=float,
                    help='train datset proportion')
parser.add_argument('--random_split', default=0, type=int,
                    help='whether randomly split train and test set')
parser.add_argument('--input_size', default=224, type=int,
                    help='inpe  ut image size')
parser.add_argument('--scheduler', default='step', type=str,
                    help='lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_data_for_plot', dest='save_data_for_plot', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrain', default=False, type=str,
                    help='use pre-trained model, trained by ultrasound image classification')
parser.add_argument('--pretrain_reg', default=False, type=str,
                    help='use pre-trained model, trained by ultrasound image classification')
parser.add_argument('--pretrain_cls', default=False, type=str,
                    help='use pre-trained model, trained by ultrasound image classification')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fold', default=0, type=int,
                    help='which fold is this training run (0-4)')
parser.add_argument('--dropout', default=0, type=float,
                    help='dropout')
parser.add_argument('--loss-type', default='ce', type=str,
                    help='ce or focal')
parser.add_argument('--print_freq', default=100, type=int,
                    help='print log frequency')
parser.add_argument('--num_points', default=2, type=int,
                    help='points to predict')
parser.add_argument('--normalizer', default='stat', type=str,
                    help='[stat, imagenet, original_imagenet]')
parser.add_argument('--gaussian', default=0, type=float, help='add gaussian noise')
parser.add_argument('--colorjitter', default=0, type=int, help='add gaussian noise')
parser.add_argument('--randomaffine', default=0, type=int, help='add gaussian noise')
parser.add_argument('--mse_loss_ratio', default=1.0, type=float, help='mean square loss ratio')
parser.add_argument('--ciou_loss_ratio', default=1.0, type=float, help='complete iou loss ratio')

best_val_loss = math.inf
action_scalar = 1


def main():
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])

    if args.seed is not None:
        seed = args.seed + args.rank
        print('{}\nseed: {}\n{}'.format('*' * 24, seed, '*' * 24))
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        if args.local_rank == 0:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
    else:
        cudnn.benchmark = True

    main_worker(args)


class CustomDataset(Dataset):
    def __init__(self, metadata, transform, is_val=True):
        self.metadata = metadata.reset_index(drop=True)
        self.number_samples = len(self.metadata)
        self.transform = transform
        self.is_val = is_val
        if is_val:
            print('[Val] Number of samples = {}'.format(self.number_samples))
        else:
            print('[Train] Number of samples = {}'.format(self.number_samples))

    def __len__(self):
        return self.number_samples

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['imgpath']
        img = Image.open(image_path)
        ob = self.transform(img)
        is_exist = self.metadata.iloc[idx]['is_intima_exists']

        if is_exist:
            json_path = image_path.replace('.jpg', '.json')
            with open(json_path, 'r') as file:
                data = json.load(file)
            
            imageWidth = data["imageWidth"]
            imageHeight = data["imageHeight"]
 
            rcca = None
            for shape in data["shapes"]:                
                if shape["label"] == "RCCA":
                    rcca = torch.tensor(shape["points"])
                    rcca[:, 0] /= imageWidth
                    rcca[:, 1] /= imageHeight
                else:
                    pass

            if rcca is None:
                raise ValueError(f"JSON file {json_path} lack RCCA label, but is_exist=True")
        
        else:
            rcca = torch.zeros((2, 2))

        return ob, is_exist, rcca


def main_worker(args):
    global best_val_loss
    exp_name = args.exp_name

    date = time.strftime('%Y%m%d', time.localtime())

    log_path = os.path.join(args.log_dir, exp_name,
                            '{}__arch{}_e{}_b{}_lr{}_mse{}_ciou{}_wd{}_g{}_dropout{}_norm{}_imsize{}_wk{}'.format(date, args.arch,
                                                                                                args.epochs,
                                                                                                args.batch_size,
                                                                                                args.lr,
                                                                                                args.mse_loss_ratio,
                                                                                                args.ciou_loss_ratio,
                                                                                                args.weight_decay,
                                                                                                args.gaussian,
                                                                                                args.dropout,
                                                                                                args.normalizer,
                                                                                                args.input_size,
                                                                                                args.workers),
                            'fold-{}'.format(args.fold))
    args.log_path = log_path
    if args.rank == 0:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(log_path, f'{timestamp}.log')
        logger = lg.get_root_logger(log_file=log_file)
        cfg.log_args_to_file(args, 'args', logger=logger)
        logger.info(f'{log_path}')
        logger.info(f'Distributed training: True')
    else:
        logger = None

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    if args.normalizer == 'imagenet':
        normalize = transforms.Normalize(
            mean=[0.193, 0.193, 0.193],
            std=[0.224, 0.224, 0.224]
        )
    elif args.normalizer == 'stat':
        normalize = transforms.Normalize(
            mean=[0.18136882, 0.18137674, 0.18136712],
            std=[0.1563932, 0.1563886, 0.15638869]
        )
    elif args.normalizer == 'original_imagenet':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        raise NotImplementedError
    transform_fns = [
        transforms.Resize((args.input_size,args.input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if args.gaussian > 0:
        transform_fns = list(transform_fns[:2] + [transforms.GaussianBlur(kernel_size=(args.gaussian, args.gaussian), sigma=(0.1, 2))] + transform_fns[2:])
    if args.colorjitter > 0:
        transform_fns = list(transform_fns[:2] + [transforms.RandomApply([
            transforms.ColorJitter(brightness=args.colorjitter, contrast=args.colorjitter, saturation=0., hue=0.)], p=0.5)] + transform_fns[2:])
    train_transform = transforms.Compose(list(transform_fns))

    val_transform_fns = [
        transforms.Resize((args.input_size,args.input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize,
    ]  
    val_transform = transforms.Compose(list(val_transform_fns))
    
    data_df_1 = pd.read_csv(args.csv_path_1)
    data_df_2 = pd.read_csv(args.csv_path_2)
    all_data_df = pd.concat([data_df_1, data_df_2], ignore_index=True)
    
    train_patients = []
    val_patients = all_data_df[all_data_df['fold'] == args.fold]['patient'].unique().tolist()
    for i in range(5):
        if i != args.fold:
            train_patients.extend(all_data_df[all_data_df['fold'] == i]['patient'].unique().tolist()) 
    indices = all_data_df['patient'].isin(train_patients)
    train_indices = indices[indices].index
    indices = all_data_df['patient'].isin(val_patients)
    val_indices = indices[indices].index
    train_metadata = all_data_df.loc[train_indices]
    val_metadata = all_data_df.loc[val_indices]
    train_dataset = CustomDataset(metadata=train_metadata, transform=train_transform, is_val=False)
    val_dataset = CustomDataset(metadata=val_metadata, transform=val_transform, is_val=True)
    
    if args.evaluate:
        model_without_ddp_cls, model_cls, model_without_ddp_reg, model_reg = load_pretrained_classification_model_for_evaluate(args, logger)
    else:
        model_without_ddp, model = load_pretrained_classification_model(args, logger)

    if not args.evaluate:
        if args.pretrain:
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError

    train_writer = SummaryWriter(os.path.join(log_path, 'train'))
    val_writer = SummaryWriter(os.path.join(log_path, 'val'))

    # loss
    cls_criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    reg_criterion = nn.MSELoss(reduction='none').cuda(args.local_rank)

    # Data Loader
    available_gpu_numbers = torch.cuda.device_count()
    print('Available GPU Numbers = {}'.format(available_gpu_numbers))

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // available_gpu_numbers, num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // available_gpu_numbers, num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)

    if args.evaluate:
        validate_cls_and_reg(val_loader, model_cls, model_reg, cls_criterion, reg_criterion, 1, args, val_writer, logger)
        return

    best_cls_epoch = 0
    best_cls_acc = 0
    best_pixel_dis_epoch = 0
    best_reg_pixel_distance_error = 10000000
    best_iou_epoch = 0
    best_iou = 0
    best_both_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, cls_criterion, reg_criterion, optimizer, epoch, args, train_writer, logger)

        # evaluate on validation set
        cls_loss, cls_acc, mse_loss, ciou_loss, all_loss, iou, reg_pixel_distance_error = validate(val_loader, model, cls_criterion, reg_criterion, epoch, args, val_writer, logger)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_cls_best = cls_acc >= best_cls_acc
        is_iou_best = iou >= best_iou
        is_pixel_dis_best = reg_pixel_distance_error <= best_reg_pixel_distance_error
        is_both_best = (cls_acc >= best_cls_acc) and (iou >= best_iou)
        if is_cls_best:
            best_cls_epoch = epoch + 1
            best_cls_acc = cls_acc
        
        if is_iou_best:
            best_iou_epoch = epoch + 1
            best_iou = iou

        if is_pixel_dis_best:
            best_pixel_dis_epoch = epoch + 1  
            best_reg_pixel_distance_error = reg_pixel_distance_error

        if is_both_best:
            best_both_epoch = epoch + 1
            best_cls_acc = cls_acc
            best_iou = iou

        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_without_ddp.state_dict(),
                'cls_loss': cls_loss,
                'cls_acc': cls_acc,
                'reg_loss': mse_loss,
                'reg_pixel_distance_error': reg_pixel_distance_error,
                'ciou_loss': ciou_loss,
                'all_loss': all_loss,
                'iou': iou,
                'best_cls_acc': best_cls_acc,
                'best_iou': best_iou,
                'best_reg_pixel_distance_error': best_reg_pixel_distance_error,
                'best_both_epoch': best_both_epoch,
                'best_cls_epoch': best_cls_epoch,
                'best_iou_epoch': best_iou_epoch,
                'best_pixel_dis_epoch': best_pixel_dis_epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, log_path, is_cls_best, is_iou_best, is_both_best)

        if args.local_rank == 0:
            logger.info("=> Epoch[{}] \n" \
                        "=> Val. ALL-Loss ----------    \t\t '{:6.5f}'\n" \
                        "=> Val. CIoU-Loss ---------    \t\t '{:6.5f}'\n" \
                        "=> Val. MSE-Loss ----------    \t\t '{:6.5f}'\n" \
                        "=> Val. CLS-Loss ----------    \t\t '{:6.5f}'\n" \
                        "=> Val. REG-Err ----------    \t\t '{:6.5f}'\n" \
                        "=> Val. REG-IoU/CLS-Acc ---    \t\t '{:6.2f}'/'{:6.2f}'".format(epoch + 1, all_loss, ciou_loss, mse_loss, cls_loss, reg_pixel_distance_error, iou, cls_acc))
            logger.info("=> Best REG-IoU on Epoch{} '{:6.2f}' \n".format(best_iou_epoch, best_iou))
            logger.info("=> Best CLS-Acc on Epoch{} '{:6.2f}' \n".format(best_cls_epoch, best_cls_acc))
            logger.info("=> Best REG-Err on Epoch{} '{:6.5f}' \n".format(best_pixel_dis_epoch, best_reg_pixel_distance_error))
            logger.info("=> Best Both on Epoch{} \n".format(best_both_epoch))
    
    if args.local_rank == 0:
        logger.info("=> Best REG-IoU on Epoch{} '{:6.2f}' \n".format(best_iou_epoch, best_iou))
        logger.info("=> Best CLS-Acc on Epoch{} '{:6.2f}' \n".format(best_cls_epoch, best_cls_acc))
        logger.info("=> Best REG-Err on Epoch{} '{:6.5f}' \n".format(best_pixel_dis_epoch, best_reg_pixel_distance_error))
        logger.info("=> Best Both on Epoch{} \n".format(best_both_epoch))

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def train(train_loader, model, cls_criterion, reg_criterion, optimizer, epoch, args, train_writer, logger):
    batch_time = AverageMeter('Time', ':6.3f')  # new meters every epoch
    data_time = AverageMeter('Data', ':6.3f')
    record_all_loss = AverageMeter(f'ALL-Loss', ':.4e')
    record_cls_loss = AverageMeter(f'CLS-Loss', ':.4e')
    record_mse_loss = AverageMeter(f'MSE-Loss', ':.4e')
    record_ciou_loss = AverageMeter(f'CIoU-Loss', ':.4e')
    record_accuracy = AverageMeter(f'Acc@1', ':6.2f')
    record_iou = AverageMeter(f'IoU', ':6.2f')
    reg_pixel_distance_error = AverageMeter(f'Reg-Pixel-Error', ':6.2f')
    lr = AverageMeter(f'LR', ':.4e', summary_type=Summary.NONE)
    meters = [batch_time, data_time, lr, record_all_loss, record_cls_loss, record_mse_loss, record_ciou_loss, record_accuracy, record_iou, reg_pixel_distance_error]
    progress = ProgressMeter(meters, prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # switch to train mode
    model.train()
    end = time.time()
    
    for iter_id, (image, cls_label, bbox_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        features = model(image.cuda(args.rank, non_blocking=True))
        gt_cls = cls_label.cuda(args.rank, non_blocking=True)
        gt_bbox = bbox_label.cuda(args.rank, non_blocking=True).reshape(-1, 2 * 2)

        # compute output
        pred_cls = model.module.cls_head(features)
        pred_reg = model.module.reg_head(features)  
        pred_bbox = xywh2xyxy(pred_reg)  # assume output is (x_min, y_min, w, h)

        # compute loss
        cls_loss = cls_criterion(pred_cls, gt_cls)
        raw_mse_loss = reg_criterion(pred_bbox, gt_bbox).mean(dim=1) 
        if sum(gt_cls) > 0:
            mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum() / sum(gt_cls))
        else:
            mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum())
        ciou_loss = args.ciou_loss_ratio * Complete_IoU_Loss(pred_bbox, gt_bbox)

        total_loss = cls_loss + mse_loss + ciou_loss

        # record loss
        record_cls_loss.update(cls_loss.item(), features.size(0))
        record_mse_loss.update(mse_loss.item(), features.size(0))
        record_ciou_loss.update(ciou_loss.item(), features.size(0))
        record_all_loss.update(total_loss.item(), features.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # calculate accuracy
        acc1 = calculate_accuracy(pred_cls, gt_cls, topk=(1,))
        record_accuracy.update(acc1[0].item(), pred_cls.size(0))

        # calculate iou
        # cur_iou = iou(pred_bbox, gt_bbox)
        raw_iou = iou(pred_bbox, gt_bbox)
        if sum(gt_cls) > 0:
            cur_iou = (raw_iou * gt_cls).sum() / sum(gt_cls)
            record_iou.update(cur_iou.item(), sum(gt_cls))
        else:
            pass

        # calculate regression pixel distance
        pix_dis = calculate_pixel_distabce(pred_bbox, gt_bbox, gt_cls)
        reg_pixel_distance_error.update(pix_dis.item(), pred_bbox.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        lr.update(optimizer.param_groups[0]['lr'], 1)

        if iter_id % args.print_freq == 0 and args.local_rank == 0:
            extra_info = '[{}/{}]'.format(iter_id + 1, len(train_loader))
            progress.display(logger, extra_info)

    if args.local_rank == 0:
        progress.display(logger)
        if train_writer is not None:
            train_writer.add_scalar('Train/ALL-Loss', record_all_loss.avg, epoch)
            train_writer.add_scalar('Train/CLS-Loss', record_cls_loss.avg, epoch)
            train_writer.add_scalar('Train/MSE-Loss', record_mse_loss.avg, epoch)
            train_writer.add_scalar('Train/CIoU-Loss', record_ciou_loss.avg, epoch)
            train_writer.add_scalar('Train/Acc', record_accuracy.avg, epoch)
            train_writer.add_scalar('Train/IoU', record_iou.avg, epoch)
            train_writer.add_scalar('Train/Pixel-Dis-Error', reg_pixel_distance_error.avg, epoch)
            train_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)


def validate(val_loader, model, cls_criterion, reg_criterion, epoch, args, val_writer, logger):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    record_all_loss = AverageMeter(f'ALL-Loss', ':.4e')
    record_cls_loss = AverageMeter(f'CLS-Loss', ':.4e')
    record_mse_loss = AverageMeter(f'MSE-Loss', ':.4e')
    record_ciou_loss = AverageMeter(f'CIoU-Loss', ':.4e')
    record_iou = AverageMeter(f'IoU', ':6.2f')
    record_accuracy = AverageMeter(f'Acc@1', ':6.2f')
    reg_pixel_distance_error = AverageMeter(f'Reg-Pixel-Error', ':6.2f')
    meters = [batch_time, record_all_loss, record_cls_loss, record_mse_loss, record_ciou_loss, record_accuracy, record_iou, reg_pixel_distance_error]
    progress = ProgressMeter(meters, prefix="Val: [{}]".format(epoch + 1))

    single_pred_exists = []
    single_pred_reg = []
    single_pred_bbox = []
    with torch.no_grad():
        end = time.time()
        for iter_id, (image, cls_label, bbox_label) in enumerate(val_loader):
            # measure data loading time
            features = model(image.cuda(args.rank, non_blocking=True))
            gt_cls = cls_label.cuda(args.rank, non_blocking=True)
            gt_bbox = bbox_label.cuda(args.rank, non_blocking=True).reshape(-1, 2 * 2)

            # compute output
            pred_cls = model.module.cls_head(features)
            pred_reg = model.module.reg_head(features)  
            pred_bbox = xywh2xyxy(pred_reg)  # assume output is (x_min, y_min, w, h)
                
            # compute loss
            cls_loss = cls_criterion(pred_cls, gt_cls)
            raw_mse_loss = reg_criterion(pred_bbox, gt_bbox).mean(dim=1) 
            if sum(gt_cls) > 0:
                mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum() / sum(gt_cls))
            else:
                mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum())
            ciou_loss = args.ciou_loss_ratio * Complete_IoU_Loss(pred_bbox, gt_bbox)

            total_loss = cls_loss + mse_loss + ciou_loss

            # record loss
            record_cls_loss.update(cls_loss.item(), features.size(0))
            record_mse_loss.update(mse_loss.item(), features.size(0))
            record_ciou_loss.update(ciou_loss.item(), features.size(0))
            record_all_loss.update(total_loss.item(), features.size(0))

            # calculate accuracy
            acc1 = calculate_accuracy(pred_cls, gt_cls, topk=(1,))
            record_accuracy.update(acc1[0].item(), pred_cls.size(0))

            # calculate iou
            # cur_iou = iou(pred_bbox, gt_bbox)
            raw_iou = iou(pred_bbox, gt_bbox)
            if sum(gt_cls) > 0:
                cur_iou = (raw_iou * gt_cls).sum() / sum(gt_cls)
                record_iou.update(cur_iou.item(), sum(gt_cls))
            else:
                pass

            # calculate regression pixel distance
            pix_dis = calculate_pixel_distabce(pred_bbox, gt_bbox, gt_cls)
            reg_pixel_distance_error.update(pix_dis.item(), pred_bbox.size(0))

            if iter_id % args.print_freq == 0 and args.local_rank == 0:
                extra_info = '[{}/{}]'.format(iter_id + 1, len(val_loader))
                progress.display(logger, extra_info)
        
        # sync losses
        record_all_loss.all_reduce()
        record_cls_loss.all_reduce()
        record_mse_loss.all_reduce()
        record_ciou_loss.all_reduce()
        record_iou.all_reduce()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter_id == len(val_loader) - 1) and args.local_rank == 0:
            progress.display(logger)

    if args.local_rank == 0:
        progress.display_summary(logger)
        if val_writer is not None:
            val_writer.add_scalar('Val/ALL-Loss', record_all_loss.avg, epoch)
            val_writer.add_scalar('Val/CLS-Loss', record_cls_loss.avg, epoch)
            val_writer.add_scalar('Val/MSE-Loss', record_mse_loss.avg, epoch)
            val_writer.add_scalar('Val/CIoU-Loss', record_ciou_loss.avg, epoch)
            val_writer.add_scalar('Val/Acc', record_accuracy.avg, epoch)
            val_writer.add_scalar('Val/IoU', record_iou.avg, epoch)
            val_writer.add_scalar('Val/Pixel-Dis-Error', reg_pixel_distance_error.avg, epoch)

    return record_cls_loss.avg.cpu().numpy(), record_accuracy.avg.cpu().numpy(), record_mse_loss.avg.cpu().numpy(), record_ciou_loss.avg.cpu().numpy(), record_all_loss.avg.cpu().numpy(), record_iou.avg.cpu().numpy(), reg_pixel_distance_error.avg.cpu().numpy()

def validate_cls_and_reg(val_loader, model_cls, model_reg, cls_criterion, reg_criterion, epoch, args, val_writer, logger):
    model_cls.eval()
    model_reg.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    record_all_loss = AverageMeter(f'ALL-Loss', ':.4e')
    record_cls_loss = AverageMeter(f'CLS-Loss', ':.4e')
    record_mse_loss = AverageMeter(f'MSE-Loss', ':.4e')
    record_ciou_loss = AverageMeter(f'CIoU-Loss', ':.4e')
    record_iou = AverageMeter(f'IoU', ':6.2f')
    record_accuracy = AverageMeter(f'Acc@1', ':6.2f')
    reg_pixel_distance_error = AverageMeter(f'Reg-Pixel-Error', ':6.2f')
    meters = [batch_time, record_all_loss, record_cls_loss, record_mse_loss, record_ciou_loss, record_accuracy, record_iou, reg_pixel_distance_error]
    progress = ProgressMeter(meters, prefix="Val: [{}]".format(epoch + 1))

    single_pred_exists = []
    single_label_exists = []
    single_pred_reg = []
    single_pred_bbox = []
    single_gt_bbox = []
    single_ciou_loss = []
    with torch.no_grad():
        end = time.time()
        for iter_id, (image, cls_label, bbox_label) in enumerate(val_loader):
            # measure data loading time
            features = model_cls(image.cuda(args.rank, non_blocking=True))
            gt_cls = cls_label.cuda(args.rank, non_blocking=True)
            gt_bbox = bbox_label.cuda(args.rank, non_blocking=True).reshape(-1, 2 * 2)

            # compute output
            pred_cls = model_cls.module.cls_head(features)
            if args.save_data_for_plot:
                for ind in range(features.shape[0]):
                    single_pred_exists.append(pred_cls[ind].tolist())
                    single_label_exists.append(gt_cls[ind].tolist())
            
            
            features = model_reg(image.cuda(args.rank, non_blocking=True))
            # compute output
            pred_reg = model_reg.module.reg_head(features)  
            pred_bbox = xywh2xyxy(pred_reg)  # assume output is (x_min, y_min, w, h)
            if args.save_data_for_plot:
                for ind in range(features.shape[0]):
                    single_pred_reg.append(pred_reg[ind].tolist())
                    single_pred_bbox.append(pred_bbox[ind].tolist())
                    single_gt_bbox.append(gt_bbox[ind].tolist())
            
            
            # compute loss
            cls_loss = cls_criterion(pred_cls, gt_cls)
            raw_mse_loss = reg_criterion(pred_bbox, gt_bbox).mean(dim=1) 
            if sum(gt_cls) > 0:
                mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum() / sum(gt_cls))
            else:
                mse_loss = args.mse_loss_ratio * ((raw_mse_loss * gt_cls).sum())
            ciou_loss = args.ciou_loss_ratio * Complete_IoU_Loss(pred_bbox, gt_bbox)
            
            if args.save_data_for_plot:
                for ind in range(features.shape[0]):
                    single_ciou_loss.append(Complete_IoU_Loss(pred_bbox[ind].unsqueeze(0), gt_bbox[ind].unsqueeze(0)).item())
            
            total_loss = cls_loss + mse_loss + ciou_loss

            # record loss
            record_cls_loss.update(cls_loss.item(), features.size(0))
            record_mse_loss.update(mse_loss.item(), features.size(0))
            record_ciou_loss.update(ciou_loss.item(), features.size(0))
            record_all_loss.update(total_loss.item(), features.size(0))

            # calculate accuracy
            acc1 = calculate_accuracy(pred_cls, gt_cls, topk=(1,))
            record_accuracy.update(acc1[0].item(), pred_cls.size(0))

            # calculate iou
            # cur_iou = iou(pred_bbox, gt_bbox)
            raw_iou = iou(pred_bbox, gt_bbox)
            if sum(gt_cls) > 0:
                cur_iou = (raw_iou * gt_cls).sum() / sum(gt_cls)
                record_iou.update(cur_iou.item(), sum(gt_cls))
            else:
                pass

            # calculate regression pixel distance
            pix_dis = calculate_pixel_distabce(pred_bbox, gt_bbox, gt_cls)
            reg_pixel_distance_error.update(pix_dis.item(), pred_bbox.size(0))

            if iter_id % args.print_freq == 0 and args.local_rank == 0:
                extra_info = '[{}/{}]'.format(iter_id + 1, len(val_loader))
                progress.display(logger, extra_info)
        
        # sync losses
        if args.save_data_for_plot:
            output_path = os.path.join(args.log_path, "single_pred_exists.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_pred_exists, file) 
                
            output_path = os.path.join(args.log_path, "single_label_exists.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_label_exists, file)  
                 
            output_path = os.path.join(args.log_path, "single_pred_reg.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_pred_reg, file)  
                 
            output_path = os.path.join(args.log_path, "single_pred_bbox.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_pred_bbox, file) 
                
            output_path = os.path.join(args.log_path, "single_ciou_loss.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_ciou_loss, file)   
                
            output_path = os.path.join(args.log_path, "single_gt_bbox.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(single_gt_bbox, file)   
        
        record_all_loss.all_reduce()
        record_cls_loss.all_reduce()
        record_mse_loss.all_reduce()
        record_ciou_loss.all_reduce()
        record_iou.all_reduce()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter_id == len(val_loader) - 1) and args.local_rank == 0:
            progress.display(logger)

    if args.local_rank == 0:
        progress.display_summary(logger)
        if val_writer is not None:
            val_writer.add_scalar('Val/ALL-Loss', record_all_loss.avg, epoch)
            val_writer.add_scalar('Val/CLS-Loss', record_cls_loss.avg, epoch)
            val_writer.add_scalar('Val/MSE-Loss', record_mse_loss.avg, epoch)
            val_writer.add_scalar('Val/CIoU-Loss', record_ciou_loss.avg, epoch)
            val_writer.add_scalar('Val/Acc', record_accuracy.avg, epoch)
            val_writer.add_scalar('Val/IoU', record_iou.avg, epoch)
            val_writer.add_scalar('Val/Pixel-Dis-Error', reg_pixel_distance_error.avg, epoch)

    return record_cls_loss.avg.cpu().numpy(), record_accuracy.avg.cpu().numpy(), record_mse_loss.avg.cpu().numpy(), record_ciou_loss.avg.cpu().numpy(), record_all_loss.avg.cpu().numpy(), record_iou.avg.cpu().numpy(), reg_pixel_distance_error.avg.cpu().numpy()


class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x
    
def load_pretrained_classification_model_for_evaluate(args, logger):
    
    #############################################################################
    if 'resnet' in args.arch:
        model_cls = models.__dict__[args.arch](weights="IMAGENET1K_V2")
        in_features = model_cls.fc.in_features
        model_cls.fc = nn.Identity()
    elif 'convnext' in args.arch:
        model_cls = models.__dict__[args.arch](weights="IMAGENET1K_V1")
        in_features = model_cls.classifier[2].in_features
        model_cls.classifier = IdentityModule()
    else:
        raise NotImplementedError

    cls_head = CLSPredHead(in_features, args.dropout, args.arch)
    model_cls.add_module("cls_head", cls_head)
    reg_head = REGPredHead(in_features, args.dropout, args.arch)
    model_cls.add_module("reg_head", reg_head)

    # create actor model
    if args.local_rank == 0:
        logger.info("=> creating model '{}'".format(args.arch))
        
    ############################################################################
    if 'resnet' in args.arch:
        model_reg = models.__dict__[args.arch](weights="IMAGENET1K_V2")
        in_features = model_reg.fc.in_features
        model_reg.fc = nn.Identity()
    elif 'convnext' in args.arch:
        model_reg = models.__dict__[args.arch](weights="IMAGENET1K_V1")
        in_features = model_reg.classifier[2].in_features
        model_reg.classifier = IdentityModule()
    else:
        raise NotImplementedError

    cls_head = CLSPredHead(in_features, args.dropout, args.arch)
    model_reg.add_module("cls_head", cls_head)
    reg_head = REGPredHead(in_features, args.dropout, args.arch)
    model_reg.add_module("reg_head", reg_head)

    # create actor model
    if args.local_rank == 0:
        logger.info("=> creating model '{}'".format(args.arch))
    
    ####################################################################################
    if args.pretrain_cls:
        if os.path.isfile(args.pretrain_cls):
            if args.local_rank == 0:
                logger.info("=> loading pretrained classification model '{}'".format(args.resume))
            checkpoint = torch.load(args.pretrain_cls, map_location='cpu')
            model_cls.load_state_dict(checkpoint['state_dict'])
        else:
            if args.local_rank == 0:
                logger.info("=> no pretrained classification model found at '{}'".format(args.resume))
                
    if args.pretrain_reg:
        if os.path.isfile(args.pretrain_reg):
            if args.local_rank == 0:
                logger.info("=> loading pretrained classification model '{}'".format(args.resume))
            checkpoint = torch.load(args.pretrain_reg, map_location='cpu')
            model_reg.load_state_dict(checkpoint['state_dict'])
        else:
            if args.local_rank == 0:
                logger.info("=> no pretrained classification model found at '{}'".format(args.resume))
                

    model_cls.cuda()
    model_without_ddp_cls = model_cls
    model_cls = nn.parallel.DistributedDataParallel(model_cls, device_ids=[args.local_rank])
    
    model_reg.cuda()
    model_without_ddp_reg = model_reg
    model_reg = nn.parallel.DistributedDataParallel(model_reg, device_ids=[args.local_rank])
    return model_without_ddp_cls, model_cls, model_without_ddp_reg, model_reg


def load_pretrained_classification_model(args, logger):
    if 'resnet' in args.arch:
        model = models.__dict__[args.arch](weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif 'convnext' in args.arch:
        model = models.__dict__[args.arch](weights="IMAGENET1K_V1")
        in_features = model.classifier[2].in_features
        model.classifier = IdentityModule()
    else:
        raise NotImplementedError

    cls_head = CLSPredHead(in_features, args.dropout, args.arch)
    model.add_module("cls_head", cls_head)
    reg_head = REGPredHead(in_features, args.dropout, args.arch)
    model.add_module("reg_head", reg_head)

    # create actor model
    if args.local_rank == 0:
        logger.info("=> creating model '{}'".format(args.arch))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            if args.local_rank == 0:
                logger.info("=> loading pretrained classification model '{}'".format(args.resume))
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            if args.local_rank == 0:
                logger.info("=> no pretrained classification model found at '{}'".format(args.resume))

    model.cuda()
    model_without_ddp = model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    return model_without_ddp, model


def save_checkpoint(state, log_path, is_cls_best, is_reg_best, is_both_best, filename='checkpoint.pth.tar'):
    file_path = os.path.join(log_path, filename)
    torch.save(state, file_path)
    if is_cls_best:
        best_file_path = os.path.join(log_path, 'model_cls_best.pth.tar')
        logger.info(f'checkpoint saved at {best_file_path}')
        shutil.copyfile(file_path, best_file_path)
    if is_reg_best:
        best_file_path = os.path.join(log_path, 'model_reg_best.pth.tar')
        logger.info(f'checkpoint saved at {best_file_path}')
        shutil.copyfile(file_path, best_file_path)
    if is_both_best:
        best_file_path = os.path.join(log_path, 'model_both_best.pth.tar')
        logger.info(f'checkpoint saved at {best_file_path}')
        shutil.copyfile(file_path, best_file_path)


class CLSPredHead(nn.Module):
    def __init__(self, in_features, dropout, arch) -> None:
        super().__init__()
        if 'resnet' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 2),
            )
        elif 'convnext' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                partial(nn.LayerNorm, eps=1e-6)(in_features),
                nn.Linear(in_features, 2),
            )

    def forward(self, x):
        return self.trunk(x)
    

class REGPredHead(nn.Module):
    def __init__(self, in_features, dropout, arch) -> None:
        super().__init__()
        if 'resnet' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 4),
                nn.Sigmoid()
            )
        elif 'convnext' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                partial(nn.LayerNorm, eps=1e-6)(in_features),
                nn.Linear(in_features, 4),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.trunk(x)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def merge_data(states, actions, extra_data_path):
    extra_data = np.load(extra_data_path, allow_pickle=True).item()
    assert isinstance(extra_data['states'], list) and isinstance(extra_data['actions'], list)
    states.extend(extra_data['states'])
    actions.extend(extra_data['actions'])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = torch.tensor(0.).cuda()  # ensure sum is a CUDA tensor
        self.count = torch.tensor(0.).cuda()  # ensure count is a CUDA tensor

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.stack([self.sum, self.count])  # stack the CUDA tensors
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total  # update sum and count with the reduced tensors
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = '{name} {val:.3f}'
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, logger, extra_info=None):
        entries = [self.prefix]
        if extra_info:
            entries += [str(extra_info)]
        entries += [str(meter) for meter in self.meters]
        string = '\t'.join(entries)
        lg.print_log(string, logger=logger)

    def display_summary(self, logger):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        string = ' '.join(entries)
        lg.print_log(string, logger=logger)


def xywh2xyxy(output):
    x_min = output[:, 0]
    y_min = output[:, 1]
    x_max = torch.clamp((output[:, 0] + output[:, 2]), max=1.0)
    y_max = torch.clamp((output[:, 1] + output[:, 3]), max=1.0)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def calculate_pixel_distabce(output, target, gt_cls):
    img_width = 566
    img_height = 624
    with torch.no_grad():
        diff = (output - target).reshape(-1, 2, 2)
        diff[:, :, 0] *= img_width
        diff[:, :, 1] *= img_height

        if sum(gt_cls) > 0:
            diff = (torch.mean(torch.sum(torch.square(diff), dim=2), dim=1) * gt_cls).sum() / sum(gt_cls)
        else:
            diff = (torch.mean(torch.sum(torch.square(diff), dim=2), dim=1) * gt_cls).sum()

        return diff


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def iou(box1, box2, wh=False):
    inter_max_xy = torch.min(box1[:, 2:], box2[:, 2:])
    inter_min_xy = torch.max(box1[:, :2], box2[:, :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    
    w1 = torch.clamp((box1[:, 2] - box1[:, 0]), min=0) 
    h1 = torch.clamp((box1[:, 3] - box1[:, 1]), min=0)
    w2 = torch.clamp((box2[:, 2] - box2[:, 0]), min=0)
    h2 = torch.clamp((box2[:, 3] - box2[:, 1]), min=0)
    area1 = w1 * h1 
    area2 = w2 * h2

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou


def Complete_IoU_Loss(box1, box2):
    """
    Reference: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
    https://arxiv.org/pdf/1911.08287.pdf
    Input:
        box1: pred bbox, shape (bs, 4)
        box2: ground truth box, shape (bs, 4)
    """
    
    inter_max_xy = torch.min(box1[:, 2:], box2[:, 2:])
    inter_min_xy = torch.max(box1[:, :2], box2[:, :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    
    w1 = torch.clamp((box1[:, 2] - box1[:, 0]), min=0) 
    h1 = torch.clamp((box1[:, 3] - box1[:, 1]), min=0)
    w2 = torch.clamp((box2[:, 2] - box2[:, 0]), min=0)
    h2 = torch.clamp((box2[:, 3] - box2[:, 1]), min=0)
    area1 = w1 * h1 
    area2 = w2 * h2

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    out_max_xy = torch.max(box1[:, 2:], box2[:, 2:])
    out_min_xy = torch.min(box1[:, :2], box2[:, :2])
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2

    center_x1 = (box1[:, 2] + box1[:, 0]) / 2  
    center_y1 = (box1[:, 3] + box1[:, 1]) / 2
    center_x2 = (box2[:, 2] + box2[:, 0]) / 2
    center_y2 = (box2[:, 3] + box2[:, 1]) / 2
    
    center_diag_distance = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    diag_ratio = torch.clamp((center_diag_distance / (outer_diag + 1e-6)), max=1)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)), 2)  # shape (bs,)
    a = v / ((1 - iou) + v + 1e-6)

    ciou = 1 - iou + diag_ratio + a * v 

    return ciou.mean()


def Complete_IoU_Loss_single_box(box1, box2):
    """
    Reference: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
    https://arxiv.org/pdf/1911.08287.pdf
    Input:
        box1: pred bbox
        box2: ground truth box
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    inter_x1 = torch.max([xmin1, xmin2])
    inter_y1 = torch.max([ymin1, ymin2])
    inter_x2 = torch.min([xmax1, xmax2])
    inter_y2 = torch.min([ymax1, ymax2])	
    
    w1 = xmax1-xmin1
    h1 = ymax1-ymin1
    w2 = xmax2-xmin2
    h2 = ymax2-ymin2
    area1 = w1 * h1
    area2 = w2 * h2
    inter_area = (torch.max([0, inter_x2 - inter_x1])) * (torch.max([0, inter_y2 - inter_y1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    outer_x1 = torch.min([xmin1, xmin2])
    outer_y1 = torch.min([ymin1, ymin2])
    outer_x2 = torch.max([xmax1, xmax2])
    outer_y2 = torch.max([ymax1, ymax2])

    outer_diag_distance = (outer_x2 - outer_x1) ** 2 + (outer_y2 - outer_y1) ** 2

    center_x1 = (xmin1 + xmax1) / 2
    center_y1 = (ymin1 + ymax1) / 2
    center_x2 = (xmin2 + xmax2) / 2
    center_y2 = (ymin2 + ymax2) / 2
    
    center_diag_distance = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    diag_ratio = center_diag_distance / (outer_diag_distance + 1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    a = v / ((1 - iou) + v + 1e-6)

    ciou = 1 - iou + diag_ratio + a * v

    return ciou


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    main()
