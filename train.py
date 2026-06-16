# train.py
import os
import datetime
import time
import math
import argparse
import logging
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2, distribute_bn
from timm.scheduler import create_scheduler

# Model imports (Linking to the new PML network)
from models.ResNet19_PML import ResNet19, ResNet_PML19

# ---------------------------------------------------------------------
# Utilities Embedded to ensure Standalone File Constraint (>600 lines)
# ---------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
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


def reduce_mean(tensor, nprocs):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def mkdir(path):
    if is_main_process() and not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    if logger.hasHandlers():
        logger.handlers.clear()

    if is_main_process():
        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def seed_all(seed, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = True


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


# ---------------------------------------------------------------------
# Progressive Multi-Level Self-Distillation Loss
# ---------------------------------------------------------------------
class PMLDistillationLoss(nn.Module):
    def __init__(self, criterion, temperature=3.0, alpha=0.5, lamb=0.9):
        super().__init__()
        self.criterion = criterion
        self.T = temperature
        self.alpha = alpha
        self.lamb = lamb

    def forward(self, outputs, target):
        final_out = outputs[0]
        loss = (1 - self.lamb) * self.criterion(final_out, target)

        if len(outputs) > 1:
            ce_loss = 0.0
            pml_loss = 0.0
            num_surrogates = len(outputs) - 1

            for i in range(1, len(outputs)):
                ce_loss += self.criterion(outputs[i], target)

                if i == len(outputs) - 1:
                    teacher_out = final_out.detach()
                else:
                    teacher_out = outputs[i + 1].detach()

                student_out = outputs[i]
                student_log_probs = F.log_softmax(student_out / self.T, dim=1)
                teacher_probs = F.softmax(teacher_out / self.T, dim=1)
                kl_div = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs) * (self.T * self.T)
                pml_loss += kl_div

            loss += self.lamb * (
                    self.alpha * (ce_loss / num_surrogates) + (1 - self.alpha) * (pml_loss / num_surrogates))

        return loss


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------
def load_data_cifar(use_cifar10=True, download=True, distributed=False, cutout=False, autoaug=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if use_cifar10:
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=download,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=download,
                                                    transform=transform_test)
    else:
        train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=download,
                                                      transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=download,
                                                     transform=transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return train_dataset, test_dataset, train_sampler, test_sampler


# ---------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------
def train(train_loader, model, criterion, optimizer, device, epoch, args, scaler=None, model_ema=None, mixup_fn=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image, target = image.to(device), target.to(device)

        if mixup_fn is not None and len(image.shape) == 4:
            image, target = mixup_fn(image, target)

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        final_output = output[0] if isinstance(output, list) else output
        if len(final_output.shape) == 3:
            final_output = final_output.mean(0)

        if mixup_fn is not None and len(target.shape) == 2:
            acc_target = target.argmax(dim=1)
            acc1, acc5 = accuracy(final_output, acc_target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(final_output, target, topk=(1, 5))

        batch_size = image.shape[0] if len(image.shape) == 4 else image.shape[1]

        if args.distributed:
            dist.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)
        losses.update(reduced_loss.item(), batch_size)
        top1.update(reduced_acc1.item(), batch_size)
        top5.update(reduced_acc5.item(), batch_size)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg


def evaluate(val_loader, model, criterion, device, args, log_suffix=''):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    auxi_top1 = None
    if getattr(args, 'num_pml', 0) > 0:
        auxi_top1 = [AverageMeter('Acc@1', ':6.2f') for _ in range(args.num_pml)]

    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix=f'Test{log_suffix}: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (image, target) in enumerate(val_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)
            loss_calc_input = output
            if isinstance(output, list) and not isinstance(criterion, PMLDistillationLoss):
                loss_calc_input = output[0]

            loss = criterion(loss_calc_input, target)

            if isinstance(output, list):
                auxi_out = output[1:]
                output_final = output[0]
            else:
                output_final = output

            if len(output_final.shape) == 3:
                output_final = output_final.mean(0)

            acc1, acc5 = accuracy(output_final, target, topk=(1, 5))
            batch_size = image.shape[0] if len(image.shape) == 4 else image.shape[1]

            if args.distributed:
                dist.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), batch_size)
            top1.update(reduced_acc1.item(), batch_size)
            top5.update(reduced_acc5.item(), batch_size)

            if isinstance(output, list) and auxi_top1 is not None:
                for j in range(len(auxi_out)):
                    auxi_acc1, _ = accuracy(auxi_out[j], target, topk=(1, 5))
                    reduced_auxi_acc1 = reduce_mean(auxi_acc1, args.nprocs)
                    auxi_top1[j].update(reduced_auxi_acc1.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        if is_main_process():
            print(f' * {log_suffix} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
            if isinstance(output, list) and auxi_top1 is not None:
                for j in range(len(auxi_top1)):
                    print(f' * {log_suffix} Auxi_Acc@1 {auxi_top1[j].avg:.3f}')

    return losses.avg, top1.avg, top5.avg


def main(args, model, criterion):
    args.nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.
    device = torch.device(args.device)

    dataset_train, dataset_test, train_sampler, test_sampler = load_data_cifar(
        use_cifar10=args.dataset == 'cifar10', download=True, distributed=args.distributed,
        cutout=args.cutout, autoaug=args.autoaug)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    if args.test_only:
        evaluate(data_loader_test, model, criterion, device, args)
        return 0.0

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=1.0, switch_prob=0.5,
            mode='batch', label_smoothing=args.smoothing, num_classes=args.num_classes
        )

    model = model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = amp.GradScaler() if args.amp else None

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint.get('max_test_acc1', 0.)

    swa_model, swa_scheduler = None, None
    if args.swa:
        import torch.optim.swa_utils as swa_utils
        swa_model = swa_utils.AveragedModel(model_without_ddp)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)

    mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir,
                              f'{args.model}_T{args.T}_b{args.batch_size}_opt{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}')
    mkdir(output_dir)
    output_dir = os.path.join(output_dir, f'operation_{args.operation}')
    mkdir(output_dir)
    output_dir = os.path.join(output_dir, f'seed{args.seed}')
    mkdir(output_dir)

    if args.distributed:
        dist.barrier()

    logger = get_logger(output_dir + '/training_log.log')
    logger.parent = None
    if args.tb and is_main_process():
        train_tb_writer = SummaryWriter(output_dir + '_logs/train')

    if is_main_process():
        logger.info(output_dir)
        logger.info(args)
        logger.info(f"Start training with seed {args.seed}")

    start_time = time.time()
    validate_loss_fn = nn.CrossEntropyLoss().to(device)
    last_best_checkpoint_path = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train(
            data_loader, model, criterion, optimizer, device, epoch, args,
            scaler=scaler, model_ema=model_ema, mixup_fn=mixup_fn
        )

        if is_main_process():
            logger.info(
                'Train Epoch:[{}/{}]\t loss={:.5f}\t top1 acc={:.3f}\t top5 acc={:.3f}'.format(epoch, args.epochs,
                                                                                               train_loss, train_acc1,
                                                                                               train_acc5))
            if args.tb:
                train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        if args.swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(data_loader_test, model, validate_loss_fn, device, args)

        if model_ema is not None:
            ema_test_loss, ema_test_acc1, ema_test_acc5 = evaluate(
                data_loader_test, model_ema.module, validate_loss_fn, device, args, log_suffix='(EMA)'
            )
            if is_main_process():
                logger.info(f'EMA Test Epoch:[{epoch}/{args.epochs}]\t Acc@1={ema_test_acc1:.3f}')

            current_acc1 = ema_test_acc1
            current_acc5 = ema_test_acc5
        else:
            current_acc1 = test_acc1
            current_acc5 = test_acc5

        if is_main_process():
            logger.info(
                'Test Epoch:[{}/{}]\t loss={:.5f}\t top1 acc={:.3f}\t top5 acc={:.3f}'.format(epoch, args.epochs,
                                                                                              test_loss, test_acc1,
                                                                                              test_acc5))

        if max_test_acc1 < current_acc1:
            max_test_acc1 = current_acc1
            test_acc5_at_max_test_acc1 = current_acc5
            save_max = True
        else:
            save_max = False

        if output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1
            }
            if model_ema:
                checkpoint['state_dict_ema'] = model_ema.module.state_dict()

            save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint_latest.pth'))
            if save_max:
                acc_str = "{:.2f}".format(max_test_acc1).replace('.', '_')
                best_model_path = os.path.join(output_dir, f"model_best_epoch_{epoch}_acc_{acc_str}.pth")
                save_on_master(checkpoint, best_model_path)
                if last_best_checkpoint_path is not None and last_best_checkpoint_path != best_model_path:
                    if is_main_process() and os.path.exists(last_best_checkpoint_path):
                        try:
                            os.remove(last_best_checkpoint_path)
                        except OSError:
                            pass
                last_best_checkpoint_path = best_model_path

    if args.swa:
        if args.distributed:
            distribute_bn(swa_model, args.world_size, True)
        swa_test_loss, swa_acc1, swa_acc5 = evaluate(data_loader_test, swa_model, validate_loss_fn, device, args,
                                                     log_suffix='(SWA)')
        if is_main_process():
            logger.info(f'SWA Final Acc@1: {swa_acc1:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if is_main_process():
        metric_name = "max_ema_test_acc1" if args.model_ema else "max_test_acc1"
        logger.info('Training time {}\t {}: {:.2f}%'.format(total_time_str, metric_name, max_test_acc1))

    return max_test_acc1


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-dir', default='./data/', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', default='ResNet_PML19', type=str)
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--T', default=2, type=int, help='simulation time')
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=0.02, type=float)
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('--output-dir', default='./logs_cifar100/', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--num_pml', default=0, type=int)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--tb', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--autoaug', action='store_true')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--seed', default=[36], nargs='+', type=int, help='list of seeds for initializing training')
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='lower lr bound')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='epochs to warmup LR')
    parser.add_argument('--mixup', type=float, default=0.5, help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.1, help='cutmix alpha')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--model-ema', action='store_true', default=True)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9998)
    parser.add_argument('--swa', action='store_true', default=True)
    parser.add_argument('--swa-start', type=int, default=250)
    parser.add_argument('--swa-lr', type=float, default=0.005)
    parser.add_argument('--zero_init_residual', action='store_true')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after training stops')
    parser.add_argument('--temperature', type=float, default=3.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lamb', type=float, default=0.9)

    args = parser.parse_args()
    return args


def build_model_and_criterion(args):
    if args.mixup > 0 or args.cutmix > 0:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = nn.CrossEntropyLoss()

    criterion = base_criterion

    if 'PML' in args.model:
        # ResNet19 Structure Configuration (Layers: [3, 3, 2] = 8 blocks total)
        pml_poi = [1, 1, 1, 1, 1, 1, 1, 1]
        kernels = [[7, 5, 3], [7, 5, 3], [7, 5, 3], [7, 5, 3],
                   [7, 5, 3], [7, 5, 3], [7, 5, 3], [7, 5, 3]]

        args.num_pml = len(pml_poi)
        pml_pads = 256
        temperature = args.temperature
        alpha = args.alpha
        lamb = args.lamb
        pois = ''.join(str(x) + 'a' for x in pml_poi)
        name_kernels = ''.join([''.join(str(x) for x in ker) + 'a' for ker in kernels])

        args.operation = 'PML_poiL' + pois[:-1] + '_pads' + str(pml_pads) + '_k' + name_kernels + \
                         '_tp' + str(temperature) + '_alpha' + str(alpha) + '_lamb' + str(lamb) + 'snn_autoaug'

        model = ResNet_PML19(num_classes=args.num_classes, pml_kernels=kernels,
                             pml_places=pml_poi, pml_pads=pml_pads, T=args.T)
        criterion = PMLDistillationLoss(criterion=criterion, temperature=temperature, alpha=alpha, lamb=lamb)
    else:
        args.operation = 'sdt_snn_autoaug'
        model = ResNet19(T=args.T, num_classes=args.num_classes, zero_init_residual=args.zero_init_residual)

    return model, criterion


if __name__ == '__main__':
    args = parse_args()
    init_distributed_mode(args)
    seed_results = []
    seeds_list = args.seed

    if is_main_process():
        print(f"============================================")
        print(f"Running Multi-Seed Training with PML-SD: {seeds_list}")
        print(f"============================================")

    for seed in seeds_list:
        args.seed = seed
        seed_all(args.seed, benchmark=False)
        model, criterion = build_model_and_criterion(args)
        best_acc = main(args, model, criterion)

        if isinstance(best_acc, torch.Tensor):
            best_acc = best_acc.item()
        seed_results.append(best_acc)

        if is_main_process():
            print(f"Seed {seed} finished. Best Acc: {best_acc:.2f}%")
            print("-" * 40)

    if is_main_process():
        if len(seed_results) > 1:
            mean_acc = statistics.mean(seed_results)
            std_acc = statistics.stdev(seed_results)
        else:
            mean_acc = seed_results[0]
            std_acc = 0.0

        print("\nResults Across All Seeds:")
        acc_str_list = [f"{acc:.2f}%" for acc in seed_results]
        print(f"Best accuracies: {acc_str_list}")
        print(f"Mean Accuracy: {mean_acc:.2f}%")
        print(f"Standard deviation: {std_acc:.2f}%")