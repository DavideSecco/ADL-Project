# multimodal self-supervised learning with DeCUR
# Adapted from https://github.com/facebookresearch/barlowtwins

from pathlib import Path
import argparse
import json
import math
import os
#import random
#import signal
#import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.distributed 
#import torch.nn.functional as F

# import diffdist
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

from src.models.decur         import DeCUR
from src.models.densedecur    import DenseDeCUR
from src.models.densecl       import DenseCL
from src.dataio.kaist_dataset import KAISTDataset
from src.dataio.loader        import build_kaist_transforms

# no warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# from cvtorchvision import cvtransforms
# from utils.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray,GaussianBlur,Solarize
# import pdb

parser = argparse.ArgumentParser(description='Multimodal self-Supervised Pretraining')
parser.add_argument('--dataset', type=str,
                    help='pretraining dataset', choices=['KAIST'])  
parser.add_argument('--method', type=str,
                    help='pretraining method', choices=['DeCUR','DenseCL','DenseDeCUR'])  
parser.add_argument('--densecl_stream', type=str, default='rgb',
                    choices=['rgb','thermal'],
                    help='per DenseCL: scegli la stream da usare (rgb o thermal)')                  
parser.add_argument('--data1', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data2', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', # 128 <= valore per karolina < 1024
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.002, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.00048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--lr', default=0.2, type=float) # no effect
parser.add_argument('--cos', action='store_true', default=False)
parser.add_argument('--schedule', default=[120,160], nargs='*', type=int)

parser.add_argument('--mode', nargs='*', default=['rgb','thermal'], help='bands to process')
parser.add_argument('--train_frac', type=float, default=1.0)
#parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--resume', type=str, default='',help='resume path.')
parser.add_argument('--dim_common', type=int, default=100) # common dimensions ATTENZIONEEEEEEEEEEEEEEEEE

parser.add_argument('--pretrained', type=str, default='',help='pretrained path.')

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument('--seed', type=int, default=42)


def init_distributed_mode(args):

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # scegli backend in base all'hardware
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        args.gpu_to_work_on = args.rank % gpu_count
        torch.cuda.set_device(args.gpu_to_work_on)
    else:
        args.gpu_to_work_on = None  # fallback CPU-only

    return   


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    global args

    args = parser.parse_args()

    init_distributed_mode(args)
    
    fix_random_seeds(args.seed)
    
    main_worker(gpu=None,args=args)


def main_worker(gpu, args):
    # create tb_writer
    if args.rank==0 and not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoint_dir,'log'))

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    
    # select model
    if args.method == 'DeCUR':
        # per runnare su CPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeCUR(args).to(device)
    elif args.method == 'DenseCL':        
        # per runnare su CPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseCL(pretrained=True).to(device)
    elif args.method == 'DenseDeCUR':
        # per runnare su CPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseDeCUR(args).to(device)

    
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    

    if torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_to_work_on])
    else:
        model = model.cpu()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # select optmizer
    if args.method == 'DeCUR':
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)
    elif args.method == 'DenseCL':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=0.9,
                            weight_decay=1e-4)
    elif args.method == 'DenseDeCUR':
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                weight_decay_filter=True,
                lars_adaptation_filter=True)
        # optimizer = torch.optim.SGD(model.parameters(), args.lr,
        #                    momentum=0.9,
        #                    weight_decay=1e-4)

    # automatically resume from checkpoint if it exists
    if args.resume:
        ckpt = torch.load(args.resume,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0    
    
    # choose which dataset for pretraining
    if args.dataset == 'KAIST':          
        
        rgb_t, th_t = build_kaist_transforms(img_size=224)

        train_dataset = KAISTDataset(
            rgb_dir=args.data1,
            th_dir=args.data2,
            rgb_transform=rgb_t,   
            th_transform=th_t,   
            mode=args.mode  
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.is_slurm_job, sampler=train_sampler, drop_last=True)

    print(f"[DEBUG] train_loader has {len(train_loader)} batches")
    print(f'[INFO] Start training...')
    print(f"[INFO] Training method: {args.method}" + (f", DenseCL stream: {args.densecl_stream}" if args.method == 'DenseCL' else ""))

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    stats = {}
    loss = None
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)
        for step, (y1, y2) in enumerate(train_loader, start=epoch * len(train_loader)):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            y1_1 = y1[0].to(device, non_blocking=True)
            y1_2 = y1[1].to(device, non_blocking=True)
            y2_1 = y2[0].to(device, non_blocking=True)
            y2_2 = y2[1].to(device, non_blocking=True)
                        
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                if args.method=='DenseDeCUR':
                    loss1,loss2,loss12,on_diag12_c = model.forward(y1_1, y1_2, y2_1, y2_2)
                    loss = (loss1 + loss2 + loss12) / 3
                elif args.method=='DeCUR':
                    loss1,loss2,loss12,on_diag12_c = model.forward(y1_1, y1_2, y2_1, y2_2)
                    loss = (loss1 + loss2 + loss12) / 3
                elif args.method=='DenseCL':
                    if args.densecl_stream == 'rgb':
                        im_q, im_k = y1_1, y1_2  # RGB
                    elif args.densecl_stream == 'thermal':
                        im_q, im_k = y2_1, y2_2  # Thermal
                    loss1, loss2, _ = model.forward(im_q, im_k) 
                    loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # pdb.set_trace()
            

            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, 
                                 step=step,
                                 #lr_weights=optimizer.param_groups[0]['lr'],
                                 #lr_biases=optimizer.param_groups[1]['lr'],
                                 #lr=optimizer.param_groups['lr'],
                                 loss=loss.item(),
                                 #loss_contra_single=loss_contra_single.item(),
                                 #loss_contra_dense=loss_contra_dense.item(),
                                 loss1=loss1.item(),
                                 loss2=loss2.item(),
                                 #loss12=loss12.item(),
                                 #on_diag12_c=on_diag12_c.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
    
        if args.rank == 0 and epoch%100==0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint_{:04d}.pth'.format(epoch))

            tb_writer.add_scalars('training log',stats,epoch)
            

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    w = 1
    if args.cos:
        w *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            w *= 0.1 if epoch >= milestone else 1.
    optimizer.param_groups[0]['lr'] = w * args.learning_rate_weights
    if optimizer.__class__.__name__ == 'LARS':
        optimizer.param_groups[1]['lr'] = w * args.learning_rate_biases 
     

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


if __name__ == '__main__':
    main()