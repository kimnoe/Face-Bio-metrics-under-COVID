import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from visualize import visualize

def main(args):
    # print("현재 경로 : ",os.getcwd())
    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK']) #for gpu parallel 설정
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]) #이거는 서버 설정하는 것인 듯
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"

    # dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size) #TCP initialization
    dist.init_process_group(backend='gloo', init_method=dist_url, rank=rank, world_size=world_size) #TCP initialization
    local_rank = args.local_rank #defait : 0
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank == 0: 
        os.makedirs(cfg.output)
    else:
        time.sleep(2)


    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank) #데이터셋 생성
    # print('데이터셋 : ',train_set)
    # print('데이터셋 중 하나 : ',train_set[0][0].shape) #(3,112,112)

    visualize(train_set,10,17,5) #시각화 visualize(데이터셋이름, 시작인덱스, 끝인덱스, 가로 행 수)


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True)
    train_loader = DataLoaderX( #데이터로더 생성
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
 
    dropout = 0.4 if cfg.dataset == "webface" else 0 #webface일때마 dropout
    backbone = get_model(args.network, dropout=dropout, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank) 
    #모델은 resnet계열로만 설정가능

    if args.resume: #학습하던거 불러와서 학습하려면
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    # for ps in backbone.parameters(): 
    #     dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    margin_softmax = losses.get_loss(args.loss)
    module_partial_fc = PartialFC( #PartialFC : distributed deep learning training framework for face recognition
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD( #백본의 optimizer
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD( #parallel optimizer
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR( #스케쥴러 설정
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(train_set) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(2000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch) #train_sampler : trainset을 distributedsampler 객체로 만들 것
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16: #fp16 : 16 bit floating point -> 계산량을 줄이기 위해 사용함(부동소수점)
                # ref : https://hoya012.github.io/blog/Mixed-Precision-Training/
                features.backward(grad_amp.scale(x_grad))
                grad_amp.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(opt_backbone)
                grad_amp.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_amp)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_pfc.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--loss', type=str, default='arcface', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)
