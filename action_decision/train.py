import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import time
import math
import random
import argparse
import numpy as np
from enum import Enum
from PIL import Image
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
import torchvision.models as models
from torch.utils.data import DistributedSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import logger as lg
from utils import config as cfg
from functools import partial

l, r, b, t = 520, 1144, 729, 163
action_keys = ['u', 'i', 'o', 'j', 'k', 'l', '7', '8', '9', '4', '5', '6', 'x']
action_key2action_idx = dict(zip(action_keys, range(len(action_keys))))
action_idx2action_key = dict(zip(range(len(action_keys)), action_keys))

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def confusion_matrix(preds, labels, num_classes):
    assert preds.shape == labels.shape
    conf_matrix = torch.zeros(num_classes, num_classes).cuda()
    print(labels.shape, preds.shape)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--log-dir', default='logs', type=str,
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
                    help='input image size')
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
parser.add_argument('--max_action', default=0.06, type=float,
                    help='max action limit')
parser.add_argument('--action-type', default='discrete', type=str,
                    help='discrete or continuous')
parser.add_argument('--reweight', default='none', type=str, help='none/weighted/sqrt-weighted/class-balanced')
parser.add_argument('--gaussian', default=0, type=float, help='add gaussian noise')
parser.add_argument('--colorjitter', default=0, type=int, help='add color jitter')
parser.add_argument('--randomaffine', default=0, type=int, help='add random affine transformation')
parser.add_argument('--aux_coef', default=0.1, type=float, help='distance aux loss')
parser.add_argument('--fold', default=-1, type=int,
                    help='which fold is this training run (0-4)')
parser.add_argument('--stage', default=0, type=int,
                    help='which specific stage to run')
parser.add_argument('--dropout', default=0, type=float,
                    help='dropout')
parser.add_argument('--loss-type', default='ce', type=str,
                    help='ce or focal')
parser.add_argument('--print_freq', default=100, type=int,
                    help='print log frequency')
parser.add_argument('--normalizer', default='stat', type=str,
                    help='[stat, imagenet, original_imagenet]')
parser.add_argument('--data_source', default='data/meta.csv', type=str,
                    help='data source')

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
    def __init__(self, metadata, transform):
        self.metadata = metadata.reset_index(drop=True)
        self.stage_length = len(self.metadata)
        self.transform = transform
        print('Number of samples = {}'.format(self.stage_length))

    def __len__(self):
        return self.stage_length

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx].path
        img = Image.open(image_path)
        img = np.asarray(img.convert('RGB')).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        ob = self.transform(img)
        action_key = torch.tensor(
            action_key2action_idx[self.metadata['action_key'][idx]], dtype=torch.long)
        supervise_distance = torch.tensor(self.metadata['supervise_distance'][idx], dtype=torch.bool)
        d = torch.tensor([self.metadata['d' + str(i)][idx] for i in range(1, 7)])
        return ob, action_key, supervise_distance, d


class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return torch.mean(F_loss)


def main_worker(args):
    global best_val_loss
    exp_name = args.exp_name

    date = time.strftime('%Y%m%d', time.localtime())

    log_path = os.path.join(args.log_dir, 'stage{}'.format(args.stage), 'all-fold', exp_name,
                            '{}__arch{}_e{}_b{}_lr{}_r{}_g{}_aux{}_wd{}_dropout{}_norm{}_imsize{}_wk{}_data{}'.format(date, args.arch,
                                                                                                args.epochs,
                                                                                                args.batch_size,
                                                                                                args.lr,
                                                                                                args.reweight,
                                                                                                args.gaussian,
                                                                                                args.aux_coef,
                                                                                                args.weight_decay,
                                                                                                args.dropout,
                                                                                                args.normalizer,
                                                                                                args.input_size,
                                                                                                args.workers,
                                                                                                args.data_source.split('/')[-1].replace('.csv', '')))
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

    args.num_actions = 13

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
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        normalize,
    ]
    val_transform = transforms.Compose(list(transform_fns))

    if args.gaussian > 0:
        transform_fns.append(AddGaussianNoise(0., args.gaussian))
    if args.colorjitter:
        transform_fns = list(transform_fns[:2] + [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)] + transform_fns[2:])
    if args.randomaffine:
        transform_fns = list(
            transform_fns[:2] + [transforms.RandomAffine(translate=(0.1, 0.1), degrees=0)] + transform_fns[2:])
    train_transform = transforms.Compose(list(transform_fns))
    all_data_df = pd.read_csv(args.data_source)
    assert args.stage is not None
    all_data_df = all_data_df[all_data_df['stage'] == args.stage]
    train_metadata = all_data_df
    train_dataset = CustomDataset(metadata=train_metadata, transform=train_transform)

    # calculate train long-tail
    train_class_cnts = {}
    stage_df = train_metadata[all_data_df.stage == args.stage]
    train_class_cnts[args.stage] = np.array(
        [len(stage_df[train_metadata.action_key == action_idx2action_key[i]]) for i in range(args.num_actions)])
    train_freqs = {s: train_class_cnts[s] / np.sum(train_class_cnts[s]) for s in train_class_cnts}
    if args.rank == 0:
        logger.info(f'class_cnts:{train_class_cnts}')
        logger.info(f'freqs:{train_freqs}')

    # calculate long-tail weight
    class_cnts = {}
    for s in range(1, 5):
        stage_df = all_data_df[all_data_df.stage == s]
        class_cnts[s] = np.array(
            [len(stage_df[all_data_df.action_key == action_idx2action_key[i]]) for i in range(args.num_actions)])
    freqs = {s: class_cnts[s] / np.sum(class_cnts[s]) for s in class_cnts}
    # 'weighted/sqrt-weighted/class-balanced/none'
    if args.reweight == 'weighted':
        weights = {s: (1 / freqs[s]) / args.num_actions for s in class_cnts}
    elif args.reweight == 'sqrt-weighted':
        weights = {s: np.sqrt((1 / freqs[s]) / args.num_actions) for s in class_cnts}
    elif args.reweight == 'class-balanced':
        gamma = 0.99
        temp_weights = {s: (1 - gamma) / (1 - gamma ** class_cnts[s]) for s in class_cnts}
        weights = {s: temp_weights[s] / np.ma.masked_invalid(temp_weights[s]).mean() for s in temp_weights}
    elif args.reweight == 'none':
        weights = {s: np.ones_like(class_cnts[s], dtype=np.float32) for s in class_cnts}

    model_without_ddp, model = load_pretrained_classification_model(args, logger)

    if args.pretrain:
        actor_params = list(model.module.actors.parameters())
        distancer_params = list(model.module.distancers.parameters())
        backbone_params = [v for k, v in model.module.named_parameters() if 'actors' not in k and 'distancers' not in k]
        optimizer = torch.optim.SGD(
            [
                {"params": backbone_params, "lr": args.lr * 0.5},
                {"params": actor_params, "lr": args.lr},
                {"params": distancer_params, "lr": args.lr},
            ],
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    else:
        raise NotImplementedError

    train_writer = SummaryWriter(os.path.join(log_path, 'train'))

    assert args.action_type == 'discrete'
    if args.action_type == 'discrete':
        criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    else:
        raise NotImplementedError

    # Data Loader
    available_gpu_numbers = torch.cuda.device_count()
    print('Available GPU Numbers = {}'.format(available_gpu_numbers))

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // available_gpu_numbers, num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, train_writer, logger, weights)

        scheduler.step()

        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, log_path, epoch + 1)

    if train_writer is not None:
        train_writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, train_writer, logger, weights):
    batch_time = AverageMeter('Time', ':6.3f')  # new meters every epoch
    data_time = AverageMeter('Data', ':6.3f')
    losses = {}
    meters = [batch_time, data_time]
    for i in range(args.stage, args.stage + 1):
        losses[f'losses_{i}'] = AverageMeter(f'Loss_{i}', ':.4e')
        meters.append(losses[f'losses_{i}'])
    for i in range(args.stage, args.stage + 1):
        losses[f'aux_losses_{i}'] = AverageMeter(f'Aux_Loss_{i}', ':.4e')
        meters.append(losses[f'aux_losses_{i}'])
    if args.action_type == 'discrete':
        top1 = {}
        for i in range(args.stage, args.stage + 1):
            top1[f'top1_{i}'] = AverageMeter(f'Acc_{i}@1', ':6.2f')
            meters.append(top1[f'top1_{i}'])
    progress = ProgressMeter(meters, prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # switch to train mode
    model.train()
    end = time.time()
    for iter_id, (states, actions, supervise_distances, distances) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        stage_features = model(states.cuda(args.rank, non_blocking=True))
        loss_cache = []
        stage_actions = actions.cuda(args.rank, non_blocking=True)
        stage_supervise_distances = supervise_distances.cuda(args.rank, non_blocking=True).float()
        stage_distances = distances.cuda(args.rank, non_blocking=True).float()
        stage_weights = torch.tensor(weights[args.stage], dtype=torch.float32).cuda(args.rank,
                                                                                    non_blocking=True).float()
        # compute output
        policy = model.module.actors(stage_features)
        pred_distances = model.module.distancers(stage_features)
        if args.loss_type == 'ce':
            action_loss = nn.CrossEntropyLoss(stage_weights)(policy * action_scalar, stage_actions * action_scalar)
        elif args.loss_type == 'focal':
            action_loss = FocalLoss(weight=stage_weights)(policy * action_scalar, stage_actions * action_scalar)

        distance_loss = nn.MSELoss()(
            pred_distances * stage_supervise_distances.unsqueeze(1),
            stage_distances * stage_supervise_distances.unsqueeze(1),
        )
        loss = action_loss.mean()
        aux_loss = args.aux_coef * distance_loss.mean()
        loss_cache.append(loss + aux_loss)
        if args.action_type == 'discrete':
            acc1 = accuracy(policy, stage_actions, topk=(1,))
            top1[f'top1_{args.stage}'].update(acc1[0].item(), stage_features.size(0))
        losses[f'losses_{args.stage}'].update(loss.item(), stage_features.size(0))
        if args.aux_coef > 0:
            losses[f'aux_losses_{args.stage}'].update(aux_loss.item(), stage_features.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss = loss_cache[0]
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_id % args.print_freq == 0 and args.local_rank == 0:
            extra_info = '[{}/{}]'.format(iter_id + 1, len(train_loader))
            progress.display(logger, extra_info)

    if args.local_rank == 0:
        progress.display(logger)
        if train_writer is not None:
            train_writer.add_scalar(f'Train_{args.stage}/Loss', losses[f'losses_{args.stage}'].avg, epoch)
            train_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            if args.action_type == 'discrete':
                train_writer.add_scalar(f'Train_{args.stage}/Acc1', top1[f'top1_{args.stage}'].avg, epoch)
            if args.aux_coef > 0:
                train_writer.add_scalar(f'Train_{args.stage}/Aux_Loss', losses[f'aux_losses_{args.stage}'].avg,
                                        epoch)


class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


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

    if args.action_type == 'discrete':
        actors = DiscreteActor(in_features, args.num_actions, args.dropout, args.arch)
    else:
        raise NotImplementedError('Only discrete actor is supported')
    distancers = DiscreteActor(in_features, 6, args.dropout, args.arch)
    model.add_module("distancers", distancers)
    model.add_module("actors", actors)

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


def save_checkpoint(state, log_path, epoch):
    file_path = os.path.join(log_path, 'checkpoint-{}.pth.tar'.format(epoch))
    torch.save(state, file_path)


class DiscreteActor(nn.Module):
    def __init__(self, in_features, num_actions, dropout, arch) -> None:
        super().__init__()
        if 'resnet' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_actions),
            )
        elif 'convnext' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                partial(nn.LayerNorm, eps=1e-6)(in_features),
                nn.Linear(in_features, num_actions),
            )

    def forward(self, x):
        return self.trunk(x)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


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
            fmtstr = ''
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
