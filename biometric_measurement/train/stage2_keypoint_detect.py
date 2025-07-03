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
parser.add_argument('--pretrain', default=False, type=str,
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
parser.add_argument('--num_points', default=20, type=int,
                    help='points to predict')
parser.add_argument('--normalizer', default='stat', type=str,
                    help='[stat, imagenet, original_imagenet]')
parser.add_argument('--angle_loss_ratio', default=1.0, type=float,
                    help='angle loss ratio')
parser.add_argument('--gaussian', default=0, type=int, help='add gaussian noise')
parser.add_argument('--colorjitter', default=0, type=float, help='add gaussian noise')
parser.add_argument('--randomaffine', default=0, type=int, help='add gaussian noise')
parser.add_argument('--linear_init_std', default=0.001, type=float, help='add gaussian noise')
parser.add_argument('--boxplot', dest='boxplot', action='store_true',
                    help='boxplot model on validation set')

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
        print('Number of samples = {}'.format(self.number_samples))

    def __len__(self):
        return self.number_samples

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['imgpath']
        image_path = image_path.replace('.jpg', '.pth').replace('corrected_annotated_data', 'corrected_annotated_data_after_process')
        img = torch.load(image_path, map_location='cpu')
        ob = self.transform(img)

        # 读取json文件
        json_path = self.metadata.iloc[idx]['jsonpath']
        json_path = json_path.replace('corrected_annotated_data', 'corrected_annotated_data_after_process')
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        imageHeight = data["imageHeight"]
        imageWidth = data["imageWidth"]

        intimaup, intimadown, media, extima = None, None, None, None
        for shape in data["shapes"]:
            if shape["label"] == "intima-up":
                intimaup = torch.tensor(shape["points"])
                intimaup[:, 0] /= imageWidth
                intimaup[:, 1] /= imageHeight
            elif shape["label"] == "intima-down":
                intimadown = torch.tensor(shape["points"])
                intimadown[:, 0] /= imageWidth
                intimadown[:, 1] /= imageHeight
            elif shape["label"] == "media":
                media = torch.tensor(shape["points"])
                media[:, 0] /= imageWidth
                media[:, 1] /= imageHeight
            elif shape["label"] == "extima":
                extima = torch.tensor(shape["points"])
                extima[:, 0] /= imageWidth
                extima[:, 1] /= imageHeight
            else:
                pass
        if self.is_val:
            pix_distance = torch.tensor([data["lumen_pixdistance"], data["media2intiam_pixdistance"], data["extima2intima_pixdistance"]])
            real_distance = torch.tensor([data["lumen_realdistance"], data["media2intiam_realdistance"], data["extima2intima_realdistance"]])
            return ob, intimaup, intimadown, media, extima, pix_distance, real_distance, imageHeight, imageWidth, json_path
        else:
            return ob, intimaup, intimadown, media, extima


def main_worker(args):
    global best_val_loss
    exp_name = args.exp_name

    date = time.strftime('%Y%m%d', time.localtime())

    log_path = os.path.join(args.log_dir, exp_name,
                            '{}__arch{}_e{}_b{}_lr{}_std{}_wd{}_g{}_cj{}_dropout{}_norm{}_imsize{}_wk{}'.format(date, args.arch,
                                                                                                                args.epochs,
                                                                                                                args.batch_size,
                                                                                                                args.lr,
                                                                                                                args.linear_init_std,
                                                                                                                args.weight_decay,
                                                                                                                args.gaussian,
                                                                                                                args.colorjitter,
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
        transforms.ToPILImage(),
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize,
    ]
    val_transform = transforms.Compose(list(transform_fns))

    if args.gaussian > 0:
        transform_fns = list(transform_fns[:2] + [transforms.GaussianBlur(kernel_size=(args.gaussian, args.gaussian), sigma=(0.1, 2))] + transform_fns[2:])
        print(transform_fns)
    if args.colorjitter > 0:
        transform_fns = list(transform_fns[:2] + [transforms.RandomApply([
            transforms.ColorJitter(brightness=args.colorjitter, contrast=args.colorjitter, saturation=0., hue=0.)], p=0.5)] + transform_fns[2:])
    train_transform = transforms.Compose(list(transform_fns))

    raw_data_df = pd.read_csv('...')
    with_intima_data = raw_data_df[raw_data_df['is_intima_exists'] == 1]['imgpath'].unique().tolist()
    indices = raw_data_df['imgpath'].isin(with_intima_data)
    with_intima_data_indices = indices[indices].index
    all_data_df = raw_data_df.loc[with_intima_data_indices]

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

    model_without_ddp, model = load_pretrained_classification_model(args, logger)

    if args.pretrain:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        if os.path.isfile(args.pretrain):
            if args.local_rank == 0:
                logger.info("=> loading pretrained classification model '{}'".format(args.resume))
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            if args.local_rank == 0:
                logger.info("=> no pretrained classification model found at '{}'".format(args.resume))
                raise FileNotFoundError("=> no pretrained classification model found at '{}'".format(args.resume))
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        raise NotImplementedError

    train_writer = SummaryWriter(os.path.join(log_path, 'train'))
    val_writer = SummaryWriter(os.path.join(log_path, 'val'))

    criterion = nn.MSELoss().cuda(args.local_rank)

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

    average_position = calculate_average_position(train_loader, args, -1, logger) 
    if args.evaluate:
        val_loss, val_loss_distance, val_loss_angle, lumen_pix_error_mean, lumen_pix_error_std, \
        lumen_real_error_mean, lumen_real_error_std, media2intima_pix_error_mean, media2intima_pix_error_std, media2intima_real_error_mean, \
        media2intima_real_error_std, extima2intima_pix_error_mean, extima2intima_pix_error_std, extima2intima_real_error_mean, \
        extima2intima_real_error_std, lumen_pix_error, media2intima_pix_error, extima2intima_pix_error =validate(val_loader, model, criterion, 0, args, val_writer, logger, average_position)
        print("lumen_pix_error:",lumen_pix_error.shape)
        print("media2intima_pix_error:",media2intima_pix_error.shape)
        print("extima2intima_pix_error:",extima2intima_pix_error.shape)
        if args.boxplot:
            output_path = os.path.join(args.log_path, "lumen_pix_error.pkl")
            print("store in :",output_path)
            with open(output_path, 'wb') as file:
                pickle.dump(lumen_pix_error.tolist(), file)
        if args.boxplot:
            output_path = os.path.join(args.log_path, "media2intima_pix_error.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(media2intima_pix_error.tolist(), file)   
        if args.boxplot:
            output_path = os.path.join(args.log_path, "extima2intima_pix_error.pkl")
            with open(output_path, 'wb') as file:
                pickle.dump(extima2intima_pix_error.tolist(), file)  
        return

    lowest_pix_error_mean = 10000000
    lowest_pix_error_std = 10000000
    best_results = [0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0]
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args, train_writer, logger, average_position)

        val_loss, val_loss_distance, val_loss_angle, lumen_pix_error_mean, lumen_pix_error_std, \
            lumen_real_error_mean, lumen_real_error_std, media2intima_pix_error_mean, media2intima_pix_error_std, media2intima_real_error_mean, \
            media2intima_real_error_std, extima2intima_pix_error_mean, extima2intima_pix_error_std, extima2intima_real_error_mean, \
                extima2intima_real_error_std, lumen_pix_error, media2intima_pix_error, extima2intima_pix_error = validate(val_loader, model, criterion, epoch, args, val_writer, logger, average_position)

        scheduler.step()

        is_best = (np.mean([lumen_pix_error_mean, media2intima_pix_error_mean, extima2intima_pix_error_mean]) < lowest_pix_error_mean) and \
            (np.mean([lumen_pix_error_std, media2intima_pix_error_std, extima2intima_pix_error_std]) < lowest_pix_error_std)
        if is_best:
            lowest_pix_error_mean = min(np.mean([lumen_pix_error_mean, media2intima_pix_error_mean, extima2intima_pix_error_mean]), lowest_pix_error_mean)
            lowest_pix_error_std = min(np.mean([lumen_pix_error_std, media2intima_pix_error_std, extima2intima_pix_error_std]), lowest_pix_error_std)
            best_epoch = epoch + 1
            best_results = [lumen_pix_error_mean,         lumen_pix_error_std,         lumen_real_error_mean,         lumen_real_error_std,
                            media2intima_pix_error_mean,  media2intima_pix_error_std,  media2intima_real_error_mean,  media2intima_real_error_std,
                            extima2intima_pix_error_mean, extima2intima_pix_error_std, extima2intima_real_error_mean, extima2intima_real_error_std]
            print("lumen_pix_error:",lumen_pix_error.shape)
            print("media2intima_pix_error:",media2intima_pix_error.shape)
            print("extima2intima_pix_error:",extima2intima_pix_error.shape)
            if args.boxplot:
                output_path = os.path.join(args.log_path, "lumen_pix_error.pkl")
                print("store in :",output_path)
                with open(output_path, 'wb') as file:
                    pickle.dump(lumen_pix_error.tolist(), file)
            if args.boxplot:
                output_path = os.path.join(args.log_path, "media2intima_pix_error.pkl")
                with open(output_path, 'wb') as file:
                    pickle.dump(media2intima_pix_error.tolist(), file)   
            if args.boxplot:
                output_path = os.path.join(args.log_path, "extima2intima_pix_error.pkl")
                with open(output_path, 'wb') as file:
                    pickle.dump(extima2intima_pix_error.tolist(), file)  
                            

        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_without_ddp.state_dict(),
                'lowest_pix_error_mean': lowest_pix_error_mean,
                'lowest_pix_error_std': lowest_pix_error_std,
                'loss': val_loss,
                'loss-distance': val_loss_distance,
                'loss-angle': val_loss_angle,
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, log_path, is_best)

        if args.local_rank == 0:
            logger.info("=> Epoch[{}] \n" \
                        "=> Val. Loss Total    \t\t '{:.5f}' \n" \
                        "=> Val. Loss Distance \t\t '{:.5f}' \n" \
                        "=> Val. Loss Angle--- \t\t '{:.5f}' \n" \
                        "=> Val. CALD in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n" \
                        "=> Val. CIMT in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n" \
                        "=> Val. CIET in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n".format(epoch + 1, val_loss, val_loss_distance, val_loss_angle,
                                                                                                                       lumen_pix_error_mean, lumen_pix_error_std, lumen_real_error_mean * 10, lumen_real_error_std *10,
                                                                                                                       media2intima_pix_error_mean, media2intima_pix_error_std, media2intima_real_error_mean * 10, media2intima_real_error_std * 10,
                                                                                                                       extima2intima_pix_error_mean, extima2intima_pix_error_std, extima2intima_real_error_mean * 10, extima2intima_real_error_std * 10))
            logger.info("=> Best Pixel Error Mean/Std on Epoch{} '{}/{}'".format(best_epoch, lowest_pix_error_mean, lowest_pix_error_std))
    
    if args.local_rank == 0:
        logger.info("=> Best Results on Epoch{} \n" \
                    "=> Val. CALD in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n" \
                    "=> Val. CIMT in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n" \
                    "=> Val. CIET in pixel MAE/Std '{:.2f}'/'{:.2f}' \t in mm MAE/Std '{:.2f}'/'{:.2f}' \n".format(best_epoch, best_results[0], best_results[1], best_results[2] * 10, best_results[3] * 10,
                                                                                                                best_results[4], best_results[5], best_results[6] * 10, best_results[7] * 10,
                                                                                                                best_results[8], best_results[9], best_results[10] * 10, best_results[11] * 10))

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def calculate_average_position(train_loader, args, epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f') 
    data_time = AverageMeter('Data', ':6.3f')
    meters = [batch_time, data_time]
    progress = ProgressMeter(meters, prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # average position
    average_position = torch.zeros((20, 2)).cuda(args.rank, non_blocking=True).float()

    end = time.time()
    for iter_id, (image, intimaup, intimadown, media, extima) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        gt_intimaup = intimaup.cuda(args.rank, non_blocking=True).float()
        gt_intimadown = intimadown.cuda(args.rank, non_blocking=True).float()
        gt_media = media.cuda(args.rank, non_blocking=True).float()
        gt_extima = extima.cuda(args.rank, non_blocking=True).float()
        gt = torch.concat([gt_intimaup, gt_intimadown, gt_media, gt_extima], dim=1)  # shape: bs, num_points, 2 
        average_position += torch.sum(gt, dim=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_id % args.print_freq == 0 and args.local_rank == 0:
            extra_info = '[{}/{}]'.format(iter_id + 1, len(train_loader))
            progress.display(logger, extra_info)

    dist.all_reduce(average_position, dist.ReduceOp.SUM, async_op=False)
    average_position /= len(train_loader.dataset)

    if args.local_rank == 0:
        print(len(train_loader.dataset))
        progress.display(logger)
    
    return average_position


def train(train_loader, model, criterion, optimizer, epoch, args, train_writer, logger, average_position):
    batch_time = AverageMeter('Time', ':6.3f')  
    data_time = AverageMeter('Data', ':6.3f')
    losses = {"distance": AverageMeter(f'Loss-distance', ':.4e'),
              "angle": AverageMeter(f'Loss-angle', ':.4e'),
              "total": AverageMeter(f'Loss', ':.4e')}
    lr = AverageMeter(f'LR', ':.4e', summary_type=Summary.NONE)
    meters = [batch_time, data_time, losses["total"], losses["distance"], losses["angle"], lr]
    progress = ProgressMeter(meters, prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # average position
    average_position = average_position.unsqueeze(0).reshape(-1, args.num_points * 2).cuda(args.rank, non_blocking=True).float()  # shape: 1, num_points * 2
        
    # switch to train mode
    model.train()
    end = time.time()
    for iter_id, (image, intimaup, intimadown, media, extima) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        features, pred_offsets = model(image.cuda(args.rank, non_blocking=True))
        x_coords = torch.zeros(args.num_points).unsqueeze(0).cuda(args.rank, non_blocking=True)
        x_coords = x_coords.expand(features.shape[0], -1).detach()
        pred_offsets = torch.stack([x_coords, pred_offsets], dim=2).reshape(features.shape[0], -1) 
        
        gt_intimaup = intimaup.cuda(args.rank, non_blocking=True).float()
        gt_intimadown = intimadown.cuda(args.rank, non_blocking=True).float()
        gt_media = media.cuda(args.rank, non_blocking=True).float()
        gt_extima = extima.cuda(args.rank, non_blocking=True).float()
        gt = torch.concat([gt_intimaup, gt_intimadown, gt_media, gt_extima], dim=1).reshape(-1, args.num_points * 2)

        pred_points = pred_offsets + average_position

        dis_loss = criterion(pred_points, gt)
        angle_loss = calculate_angle_loss(pred_points[:, 10:20], pred_points[:, 20:30], pred_points[:, 30:])
        total_loss = dis_loss + args.angle_loss_ratio * angle_loss

        losses["distance"].update(dis_loss.item(), features.size(0))
        losses["angle"].update(angle_loss.item(), features.size(0))
        losses["total"].update(total_loss.item(), features.size(0))
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        # record learning rate
        lr.update(optimizer.param_groups[0]['lr'], 1)

        if iter_id % args.print_freq == 0 and args.local_rank == 0:
            extra_info = '[{}/{}]'.format(iter_id + 1, len(train_loader))
            progress.display(logger, extra_info)

    if args.local_rank == 0:
        progress.display(logger)
        if train_writer is not None:
            train_writer.add_scalar('Train/Loss', losses["total"].avg, epoch)
            train_writer.add_scalar('Train/Loss-distance', losses["distance"].avg, epoch)
            train_writer.add_scalar('Train/Loss-angle', losses["angle"].avg, epoch)
            train_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)


def fit_line(points):
    points_x = points[:,:,0]
    points_y = points[:,:,1]
    
    n_points = points_x.size(1)
    mean_x = points_x.mean(dim=1).unsqueeze(1)
    mean_y = points_y.mean(dim=1).unsqueeze(1)
    xy_mean = (points_x * points_y).mean(dim=1).unsqueeze(1)
    
    x_squared_mean = (points_x ** 2).mean(dim=1).unsqueeze(1)
    
    slope_m = (xy_mean - mean_x * mean_y) / (x_squared_mean - mean_x ** 2)+1e-10
    
    return slope_m


def validate(val_loader, model, criterion, epoch, args, val_writer, logger, average_position):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = {"distance": AverageMeter(f'Loss-distance', ':.4e'),
              "angle": AverageMeter(f'Loss-angle---', ':.4e'),
              "total": AverageMeter(f'Loss', ':.4e')}
    meters = [batch_time, losses["total"], losses["distance"], losses["angle"]]
    progress = ProgressMeter(meters, prefix="Val: [{}]".format(epoch + 1))

    # average position
    average_position = average_position.unsqueeze(0).reshape(-1, args.num_points * 2).cuda(args.rank, non_blocking=True).float()  

    depth = 4 # cm
    ori_imageHeight = 566 # pixel
    pix2real_ratio = depth / ori_imageHeight

    local_pix_preds = []
    local_real_preds = []
    local_pix_gt = []
    local_real_gt = []
    with torch.no_grad():
        end = time.time()
        for iter_id, (image, intimaup, intimadown, media, extima, pix_distance, real_distance, imageHeight, imageWidth, json_path) in enumerate(val_loader):
            features, pred_offsets = model(image.cuda(args.rank, non_blocking=True))
            x_coords = torch.zeros(args.num_points).unsqueeze(0).cuda(args.rank, non_blocking=True)
            x_coords = x_coords.expand(features.shape[0], -1).detach()
            pred_offsets = torch.stack([x_coords, pred_offsets], dim=2).reshape(features.shape[0], -1)
        
            gt_intimaup = intimaup.cuda(args.rank, non_blocking=True).float() 
            gt_intimadown = intimadown.cuda(args.rank, non_blocking=True).float()
            gt_media = media.cuda(args.rank, non_blocking=True).float()
            gt_extima = extima.cuda(args.rank, non_blocking=True).float()
            gt = torch.concat([gt_intimaup, gt_intimadown, gt_media, gt_extima], dim=1).reshape(-1, args.num_points * 2)
            pix_distance = pix_distance.cuda(args.rank, non_blocking=True).float()
            real_distance = real_distance.cuda(args.rank, non_blocking=True).float()
            imageHeight = imageHeight.cuda(args.rank, non_blocking=True).float()
            imageWidth = imageWidth.cuda(args.rank, non_blocking=True).float()
            
            pred_points = pred_offsets + average_position

            dis_loss = criterion(pred_points, gt)
            angle_loss = calculate_angle_loss(pred_points[:, 10:20], pred_points[:, 20:30], pred_points[:, 30:])
            total_loss = dis_loss + args.angle_loss_ratio * angle_loss

            losses["distance"].update(dis_loss.item(), features.size(0))
            losses["angle"].update(angle_loss.item(), features.size(0))
            losses["total"].update(total_loss.item(), features.size(0))

            scale2pix_pred_points = pred_points.reshape(-1, args.num_points, 2)
            scale2pix_pred_points[:, :, 0] *= imageWidth.unsqueeze(1)
            scale2pix_pred_points[:, :, 1] *= imageHeight.unsqueeze(1)
            pred_intimaup = scale2pix_pred_points[:, :5, :]
            pred_intimadown = scale2pix_pred_points[:, 5:10, :]
            pred_media = scale2pix_pred_points[:, 10:15, :]
            pred_extima = scale2pix_pred_points[:, 15:, :]
            
            slope_intimaup = fit_line(pred_intimaup)
            slope_intimadown = fit_line(pred_intimadown)
            slope_media = fit_line(pred_media)
            slope_extima = fit_line(pred_extima)
            
            angle_for_lumen_pixdistances = torch.atan((slope_intimaup + slope_intimadown) / 2).squeeze(1)
            angle_for_media2intima_pixdistances = torch.atan((slope_media + slope_intimadown) / 2).squeeze(1)
            angle_for_extima2intima_pixdistances = torch.atan((slope_extima + slope_intimadown) / 2).squeeze(1)
            
            pred_lumen_pixdistances = torch.mean(torch.norm(pred_intimadown - pred_intimaup, dim=2), dim=1)*torch.cos(angle_for_lumen_pixdistances)
            pred_media2intima_pixdistances = torch.mean(torch.norm(pred_media - pred_intimadown, dim=2), dim=1)*torch.cos(angle_for_media2intima_pixdistances)
            pred_extima2intima_pixdistances = torch.mean(torch.norm(pred_extima - pred_intimadown, dim=2), dim=1)*torch.cos(angle_for_extima2intima_pixdistances)
            pred_pixdistance = torch.stack([pred_lumen_pixdistances, pred_media2intima_pixdistances, pred_extima2intima_pixdistances], dim=1)
    
            pred_realdistance = pred_pixdistance * pix2real_ratio
            local_pix_preds.append(pred_pixdistance)
            local_real_preds.append(pred_realdistance)
            local_pix_gt.append(pix_distance)
            local_real_gt.append(real_distance)
            
            if iter_id % args.print_freq == 0 and args.local_rank == 0:
                extra_info = '[{}/{}]'.format(iter_id + 1, len(val_loader))
                progress.display(logger, extra_info)
        
        local_pix_preds = torch.concat(local_pix_preds, dim=0)
        local_real_preds = torch.concat(local_real_preds, dim=0)
        local_pix_gt = torch.concat(local_pix_gt, dim=0)
        local_real_gt = torch.concat(local_real_gt, dim=0)

        all_pix_preds = [torch.zeros_like(local_pix_preds) for _ in range(args.world_size)]
        all_real_preds = [torch.zeros_like(local_real_preds) for _ in range(args.world_size)]
        all_pix_gt = [torch.zeros_like(local_pix_gt) for _ in range(args.world_size)]
        all_real_gt = [torch.zeros_like(local_real_gt) for _ in range(args.world_size)]
        
        
        dist.all_gather(tensor_list=all_pix_preds, tensor=local_pix_preds)
        dist.all_gather(tensor_list=all_real_preds, tensor=local_real_preds)
        dist.all_gather(tensor_list=all_pix_gt, tensor=local_pix_gt)
        dist.all_gather(tensor_list=all_real_gt, tensor=local_real_gt)

        
        all_pix_preds = torch.concat(all_pix_preds, dim=0)
        all_real_preds = torch.concat(all_real_preds, dim=0)
        all_pix_gt = torch.concat(all_pix_gt, dim=0)
        all_real_gt = torch.concat(all_real_gt, dim=0)
        
        lumen_pix_error = torch.abs(all_pix_preds[:, 0] - all_pix_gt[:, 0])
        lumen_pix_error_mean = torch.mean(lumen_pix_error)
        lumen_pix_error_std = torch.std(lumen_pix_error)
        lumen_real_error = torch.abs(all_real_preds[:, 0] - all_real_gt[:, 0])
        lumen_real_error_mean = torch.mean(lumen_real_error)
        lumen_real_error_std = torch.std(lumen_real_error)

        media2intima_pix_error = torch.abs(all_pix_preds[:, 1] - all_pix_gt[:, 1])
        media2intima_pix_error_mean = torch.mean(media2intima_pix_error)
        media2intima_pix_error_std = torch.std(media2intima_pix_error)
        media2intima_real_error = torch.abs(all_real_preds[:, 1] - all_real_gt[:, 1])
        media2intima_real_error_mean = torch.mean(media2intima_real_error)
        media2intima_real_error_std = torch.std(media2intima_real_error)

        extima2intima_pix_error = torch.abs(all_pix_preds[:, 2] - all_pix_gt[:, 2])
        extima2intima_pix_error_mean = torch.mean(extima2intima_pix_error)
        extima2intima_pix_error_std = torch.std(extima2intima_pix_error)
        extima2intima_real_error = torch.abs(all_real_preds[:, 2] - all_real_gt[:, 2])
        extima2intima_real_error_mean = torch.mean(extima2intima_real_error)
        extima2intima_real_error_std = torch.std(extima2intima_real_error)

        # sync losses
        losses["distance"].all_reduce()
        losses["angle"].all_reduce()
        losses["total"].all_reduce()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter_id == len(val_loader) - 1) and args.local_rank == 0:
            progress.display(logger)

    if args.local_rank == 0:
        progress.display_summary(logger)
        if val_writer is not None:
            val_writer.add_scalar('Val/Loss', losses["total"].avg, epoch)
            val_writer.add_scalar('Val/Loss-distance', losses["distance"].avg, epoch)
            val_writer.add_scalar('Val/Loss-angle', losses["angle"].avg, epoch)
            val_writer.add_scalar('Val/Pix-Lumen-Error-Mean', lumen_pix_error_mean.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Lumen-Error-Std', lumen_pix_error_std.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Media2Intima-Error-Mean', media2intima_pix_error_mean.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Media2Intima-Error-Std', media2intima_pix_error_std.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Extima2Intima-Error-Mean', extima2intima_pix_error_mean.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Extima2Intima-Error-Std', extima2intima_pix_error_std.cpu().numpy(), epoch)
            val_writer.add_scalar('Val/Pix-Error-Mean-Mean', np.mean([lumen_pix_error_mean.cpu().numpy(), media2intima_pix_error_mean.cpu().numpy(), extima2intima_pix_error_mean.cpu().numpy()]), epoch)
            val_writer.add_scalar('Val/Pix-Error-Std-Mean', np.mean([lumen_pix_error_std.cpu().numpy(), media2intima_pix_error_std.cpu().numpy(), extima2intima_pix_error_std.cpu().numpy()]), epoch)

    return losses["total"].avg.cpu().numpy(), losses["distance"].avg.cpu().numpy(), losses["angle"].avg.cpu().numpy(), \
        lumen_pix_error_mean.cpu().numpy(), lumen_pix_error_std.cpu().numpy(), lumen_real_error_mean.cpu().numpy(), lumen_real_error_std.cpu().numpy(), \
        media2intima_pix_error_mean.cpu().numpy(), media2intima_pix_error_std.cpu().numpy(), media2intima_real_error_mean.cpu().numpy(), media2intima_real_error_std.cpu().numpy(), \
        extima2intima_pix_error_mean.cpu().numpy(), extima2intima_pix_error_std.cpu().numpy(), extima2intima_real_error_mean.cpu().numpy(), extima2intima_real_error_std.cpu().numpy(), lumen_pix_error, media2intima_pix_error, extima2intima_pix_error


class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class Four_model(nn.Module):
    def __init__(self, args, logger) -> None:
        super().__init__()
        self.model_intimaup = load_pretrained_classification_single_model(args, logger)
        self.model_intimadown = load_pretrained_classification_single_model(args, logger)
        self.model_media = load_pretrained_classification_single_model(args, logger)
        self.model_extima = load_pretrained_classification_single_model(args, logger)

    def forward(self, x):
        intimaup_feature = self.model_intimaup(x)
        intimadown_feature = self.model_intimadown(x)
        media_feature = self.model_media(x)
        extima_feature = self.model_extima(x)
        feature = torch.cat((intimaup_feature, intimadown_feature, media_feature, extima_feature), 1)
        
        intimaup = self.model_intimaup.actors(intimaup_feature)
        intimadown = self.model_intimadown.actors(intimadown_feature)
        media = self.model_media.actors(media_feature)
        extima = self.model_extima.actors(extima_feature)
        pred_offset = torch.cat((intimaup, intimadown, media, extima), 1)
        
        return feature, pred_offset


def load_pretrained_classification_single_model(args, logger):
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

    actors = PointPredHead(in_features, args.num_points, args.dropout, args.arch, args.linear_init_std)
    model.add_module("actors", actors)

    if args.local_rank == 0:
        logger.info("=> creating model '{}'".format(args.arch))

    return model


def load_pretrained_classification_model(args, logger):
    model = Four_model(args, logger)
    model.cuda()
    model_without_ddp = model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    return model_without_ddp, model


def save_checkpoint(state, log_path, is_best, filename='checkpoint.pth.tar'):
    file_path = os.path.join(log_path, filename)
    torch.save(state, file_path)
    if is_best:
        best_file_path = os.path.join(log_path, 'model_best.pth.tar')
        logger.info(f'checkpoint saved at {best_file_path}')
        shutil.copyfile(file_path, best_file_path)


class PointPredHead(nn.Module):
    def __init__(self, in_features, num_points, dropout, arch, std) -> None:
        super().__init__()
        self.arch = arch
        self.std = std
        if 'resnet' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_points // 4),
                nn.Tanh()
            )
        elif 'convnext' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                partial(nn.LayerNorm, eps=1e-6)(in_features),
                nn.Linear(in_features, num_points // 4),
                nn.Tanh()
            )
        else:
            NotImplementedError

        self.init_param()

    def init_param(self):
        if 'resnet' in self.arch:
            nn.init.normal_(self.trunk[1].weight, std=self.std)
            nn.init.zeros_(self.trunk[1].bias)
        elif 'convnext' in self.arch:
            nn.init.normal_(self.trunk[3].weight, std=self.std)
            nn.init.zeros_(self.trunk[3].bias)
        else:
            NotImplementedError

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
        self.sum = torch.tensor(0.).cuda()  
        self.count = torch.tensor(0.).cuda()  

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.stack([self.sum, self.count]) 
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total  
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


def accuracy(output, target, topk=(1,)):
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


def accuracy_ordinal(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        output = output > 0.5
        output = output.to(torch.float)

        pred = []
        for item in output:
            if item.equal(torch.tensor([1, 1]).cuda()):
                pred.append(1)
            elif item.equal(torch.tensor([1, 0]).cuda()):
                pred.append(2)
            elif item.equal(torch.tensor([0, 0]).cuda()):
                pred.append(0)
            else:
                pred.append(-1)
        pred = torch.tensor(pred).cuda()

        batch_size = target.size(0)
        correct = (pred.eq(target.cuda()) == True).sum()

        res = []
        res.append(torch.tensor(100.0 * float(correct) / float(batch_size)).cuda())
        res.append(torch.tensor(100).float().cuda())
        return res
    

def calculate_angle_loss(A, B, C):
    BA = A - B
    BC = C - B
    horizontal = torch.zeros_like(A, device=A.device)
    horizontal[:, 0] = 1.0
    
    cos_angle_BA = F.cosine_similarity(BA, horizontal, dim=1)
    
    cos_angle_CB = F.cosine_similarity(BC, horizontal, dim=1)
    
    loss = torch.abs(cos_angle_BA + cos_angle_CB)
    
    return loss.mean()


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
