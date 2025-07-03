import datetime
import os
import random
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from medpy.metric.binary import hd95

from dataset_with_dete_prior import ImageFolder_DFMS
from misc import AvgMeter, check_mkdir
from model import DFMS

def parse_args():
    parser = argparse.ArgumentParser(description='DFMS Training Script')
    parser.add_argument('--epoch_num', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate')
    parser.add_argument('--lr_step', type=int, default=1500, help='Learning rate decay step')
    parser.add_argument('--lr_decay', type=int, default=50, help='Learning rate decay factor')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--snapshot', type=str, default='100', help='Snapshot name')
    parser.add_argument('--save_fre', type=int, default=5, help='Model save frequency (epochs)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--ckpt_path', type=str, default='...', help='Checkpoint path')
    parser.add_argument('--exp_name', type=str, default='DFMS_train1', help='Experiment name')
    parser.add_argument('--data_root', type=str, default='...', help='Data path')
    return parser.parse_args()


args = parse_args()
torch.cuda.set_device(args.device_id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

ckpt_path = args.ckpt_path
exp_name = args.exp_name

transform = transforms.ToTensor()
target_transform = transforms.ToTensor()

training_root = os.path.join(args.data_root, 'train')
testing_root = os.path.join(args.data_root, 'test')
train_set = ImageFolder_DFMS(training_root, None, transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=12, shuffle=True)
val_set = ImageFolder_DFMS(testing_root, None, transform, target_transform)
val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)

bce_logit = nn.BCEWithLogitsLoss().cuda()

log_path = os.path.join(ckpt_path, exp_name, f'{datetime.datetime.now()}.txt')
avg_path = os.path.join(ckpt_path, exp_name, f'{exp_name}_avg.txt')

def main():
    net = DFMS().cuda()

    optimizer = optim.SGD([
        {'params': [p for n, p in net.named_parameters() if n[-4:] == 'bias'], 'lr': 2 * args.lr},
        {'params': [p for n, p in net.named_parameters() if n[-4:] != 'bias'], 'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum)

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))

    with open(log_path, 'w') as f:
        f.write(str(args) + '\n\n')

    with open(avg_path, 'w') as loss_file_avg:
        train(net, optimizer, loss_file_avg)

def train(net, optimizer, loss_file_avg):
    curr_iter = 0

    for epoch in range(args.epoch_num + 1):
        net.train()
        train_loss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            if curr_iter > args.lr_step:
                lr_factor = (1 - float(epoch) / args.epoch_num) ** 0.9
                optimizer.param_groups[0]['lr'] = 2 * args.lr * lr_factor
                optimizer.param_groups[1]['lr'] = args.lr * lr_factor

            inputs, labels, dete = data
            labels = labels[:, 0:1, :, :]
            inputs, labels, dete = map(lambda x: Variable(x).cuda(), [inputs, labels, dete])
            input_pair = [inputs, dete]

            optimizer.zero_grad()
            outputs = net(input_pair, is_train=True)
            loss = sum([2 * bce_logit(out, labels) for out in outputs])
            loss.backward()
            optimizer.step()

            train_loss_record.update(loss.item(), inputs.size(0))

            log = f'[epoch {epoch}], [iter {curr_iter}], [train loss {train_loss_record.avg:.5f}], [lr {optimizer.param_groups[1]["lr"]:.13f}]'
            print(log)
            with open(log_path, 'a') as f:
                f.write(log + '\n')

            curr_iter += 1

        if epoch % args.save_fre == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
            eval(net, val_loader, loss_file_avg, epoch)

def eval(net, val_loader, loss_file_avg, epoch):
    net.eval()

    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'hd95': [], 'F1': []}

    with torch.no_grad():
        for data in val_loader:
            inputs, labels, dete = data
            inputs_var, dete_var = map(lambda x: Variable(x).cuda(), [inputs, dete])
            labels = labels[:, 0, :, :].numpy()

            prediction = net([inputs_var, dete_var], is_train=False).data.squeeze(0).cpu().numpy()

            metrics['dice'].append(dice_coeff(prediction, labels))
            metrics['iou'].append(iou_score(prediction, labels))
            metrics['precision'].append(ppv(prediction, labels))
            metrics['recall'].append(sensitivity(prediction, labels))
            metrics['hd95'].append(myhd95(prediction, labels))
            metrics['F1'].append(2 * metrics['precision'][-1] * metrics['recall'][-1] / (metrics['precision'][-1] + metrics['recall'][-1] + 1e-5))

    avg_log = f'[epoch {epoch}], ' + ', '.join([f'avg_{k}={np.mean(v):.4f}' for k, v in metrics.items()])
    print(avg_log)
    loss_file_avg.write(avg_log + '\n')

def dice_coeff(pred, mask):
    smooth = 1e-5
    return (2. * (pred * mask).sum() + smooth) / (pred.sum() + mask.sum() + smooth)

def iou_score(pred, mask):
    smooth = 1e-5
    pred_, mask_ = pred > 0.5, mask > 0.5
    return ((pred_ & mask_).sum() + smooth) / ((pred_ | mask_).sum() + smooth)

def sensitivity(pred, mask):
    smooth = 1e-5
    pred_, mask_ = pred > 0.5, mask > 0.5
    return ((pred_ & mask_).sum() + smooth) / (mask_.sum() + smooth)

def ppv(pred, mask):
    smooth = 1e-5
    pred_, mask_ = pred > 0.5, mask > 0.5
    return ((pred_ & mask_).sum() + smooth) / (pred_.sum() + smooth)

def myhd95(pred, mask):
    pred_binary = pred > 0.5
    mask_binary = mask > 0.5
    try:
        return hd95(pred_binary, mask_binary)
    except:
        return 0.0 if np.all(pred_binary == 0) and np.all(mask_binary == 0) else np.sqrt(pred_binary.shape[0]**2 + pred_binary.shape[1]**2)

if __name__ == '__main__':
    main()