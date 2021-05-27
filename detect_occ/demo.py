# -*- coding:utf-8 –*- #
from __future__ import print_function, division
import time
import argparse
import os
import random
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import pdb
sys.path.append('..')
import models
from utils.utility import save_checkpoint, AverageMeter, kaiming_init
from utils.config import config, update_config
from utils.compute_loss import get_criterion_occ
from utils.metrics import MetricsEvaluator
from utils.visualizer import viz_and_log, viz_and_save, plot_train_metrics, plot_val_metrics
from utils.custom_transforms import get_data_transforms, resize_to_origin
from lib.logger.create_logger import create_logger
from lib.logger.print_and_log import print_and_log
from lib.dataset.occ_dataset import PIOD_Dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


def parse_args():
    parser = argparse.ArgumentParser(description='Train and val for occlusion edge/order detection')
    parser.add_argument('--config', default='',
                        required=False, type=str, help='experiment configure file name')
    args, rest = parser.parse_known_args()
    update_config(args.config)  # update params with experiment config file

    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--new_val', action='store_true', help='new val with resumed model, re-calculate val perf ')
    parser.add_argument('--out_dir', default='', type=str, metavar='PATH', help='res output dir(defaut: output/date)')
    parser.add_argument('--evaluate', action='store_true', help='test with best model in validation')
    parser.add_argument('--frequent', default=config.default.frequent, type=int, help='frequency of logging')
    parser.add_argument('--gpus', help='specify the gpu to be use', default='3', required=False, type=str)
    parser.add_argument('--cpu', default=False, required=False, type=bool, help='whether use cpu mode')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--vis', action='store_true', help='turn on visualization')
    parser.add_argument('--arch', '-a', metavar='ARCH', choices=model_names,
                        help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))

    parser.add_argument('--lst', default='', type=str, metavar='PATH')
    parser.add_argument('--datapath', default='', type=str, metavar='PATH')
    parser.add_argument('--outpath', default='', type=str, metavar='PATH')

    args = parser.parse_args()

    return args


args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
n_iter = 0  # train iter num
best_perf = 0

torch.cuda.set_device(int(args.gpus))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    global args, curr_path, best_perf
    print('Called with argument:', args)

    set_seed(1024)

    config.root_path = os.path.join(curr_path, '..')
    if args.out_dir:  # specify out dir
        output_path = os.path.join(config.root_path, config.output_path, args.out_dir)
    else:
        output_path = os.path.join(config.root_path, config.output_path)

    # logger
    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    logger, save_path = create_logger(output_path, args, config.dataset.image_set,
                                      curr_time, temp_file=False)
    logger.info('called with args {}'.format(args))
    logger.info('called with config {}'.format(config))

    print_and_log('=> will save everything to {}'.format(save_path), logger)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # copy curr cfg file to output dir
    cmd = "cp {0} {1}/{2}_{3}.yaml".format(args.config, save_path,
                                           args.config.split('/')[-1].split('.')[0], curr_time)
    print('=> created current config file: {}_{}.yaml'
          .format(args.config.split('/')[-1].split('.')[0], curr_time))
    os.system(cmd)

    # set tensorboard writer
    train_writer = SummaryWriter(os.path.join(save_path, 'train_vis'))
    val_writer   = SummaryWriter(os.path.join(save_path, 'val_vis'))
    train_vis_writers = []
    val_vis_writers = []

    for i in range(4):  # writers for n sample visual res
        train_vis_writers.append(SummaryWriter(os.path.join(save_path, 'train_vis', str(i))))
        val_vis_writers.append(SummaryWriter(os.path.join(save_path, 'val_vis', str(i))))




    # create model
    network_data = None  # no pretrained weights
    args.arch = config.network.arch
    model = models.__dict__[args.arch](config, network_data)
    print_and_log("=> created model '{}'".format(args.arch), logger)

    # gpu settings
    gpu_ids = list(args.gpus.replace(',', ''))
    args.gpus = [int(gpu_id) for gpu_id in gpu_ids]

    if args.gpus.__len__() == 1:
        # single GPU
        torch.cuda.set_device(args.gpus[0])
        model = model.cuda(args.gpus[0])
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    cudnn.benchmark = False  # True if fixed train/val input size
    print('=> cudnn.deterministic flag:', cudnn.deterministic)

    criterion = get_criterion_occ()  # define loss functions(criterion)

    # define optimizer and lr scheduler
    resnet = list(map(id, model.inconv.parameters()))
    resnet += list(map(id, model.encoder.parameters()))

    output = list(map(id, model.out0_edge.parameters()))
    output += list(map(id, model.out0_ori.parameters()))

    other_parts = filter(lambda p: id(p) not in resnet + output, model.parameters())

    # other_parts = filter(lambda p:id(p) not in resnet, model.parameters())

    assert (config.TRAIN.optimizer in ['adam', 'sgd'])
    print_and_log('=> setting {} solver'.format(config.TRAIN.optimizer), logger)
    if config.TRAIN.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.inconv.parameters(), 'lr': config.TRAIN.lr*0.1},
                                      {'params': model.encoder.parameters(), 'lr': config.TRAIN.lr*0.1},
                                      {'params': other_parts, 'lr': config.TRAIN.lr},
                                      {'params': model.out0_ori.parameters(), 'lr': config.TRAIN.lr*0.01},
                                      {'params': model.out0_edge.parameters(), 'lr': config.TRAIN.lr*0.01}
                                      ], config.TRAIN.lr, betas=config.TRAIN.betas)
    elif config.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.lr, momentum=config.TRAIN.momentum,
                                    weight_decay=config.TRAIN.weight_decay, nesterov=config.TRAIN.nesterov)

    print(output_path)
    if args.resume:  # optionally resume from a checkpoint with trained model or train from scratch
        model_path = os.path.join(output_path, args.resume)
        if os.path.isfile(model_path):
            print("=> loading resumed model '{}'".format(model_path))
            resumed_model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpus[0]))
            model.load_state_dict(resumed_model['state_dict'])
            optimizer.load_state_dict(resumed_model['optimizer'])
            config.TRAIN.begin_epoch = resumed_model['epoch']
            best_perf    = resumed_model['best_perf']
            train_losses = resumed_model['train_losses']
            date_time    = resumed_model['date_time']  # date of beginning
            print("=> loaded resumed model '{}' (epoch {})".format(args.resume, resumed_model['epoch']))
            if config.TRAIN.end_epoch > len(train_losses):  # resume training with more epochs
                train_losses = np.append(train_losses, np.zeros([config.TRAIN.end_epoch - len(train_losses)]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        train_losses = np.zeros(config.TRAIN.end_epoch)
        date_time = curr_time

    # setup learning schedule
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.milestones, gamma=0.1)
    # for epoch in range(0, config.TRAIN.begin_epoch):
    #     lr_scheduler.step()


    # --------------------------------- test on test set from resumed model ------------------------------------------ #
    if args.evaluate:
        print_and_log('=> evaluation mode', logger)
        _, _, test_co_transf = get_data_transforms(config)
        test_dataset = PIOD_Dataset(config, True, None, None, test_co_transf)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.workers,
                                                  pin_memory=True, shuffle=False)
        print_and_log('=> {} test samples'.format(len(test_dataset)), logger)

        # epoch = config.TRAIN.begin_epoch
        # test_vis_writers = []
        # for i in range(4):
        #     test_vis_writers.append(SummaryWriter(os.path.join(save_path, 'test_vis', str(i))))
        # results_vis_path = os.path.join(save_path, 'results_vis')
        # print('save visual results to {}'.format(results_vis_path))
        # if not os.path.exists(results_vis_path): os.makedirs(results_vis_path)

        save_path = args.outpath
        val_epoch(test_loader, model, save_path)


        return
    # ---------------------------------------------------------------------------------------------------------------- #





def val_epoch(val_loader, model, res_path):
    global args
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    # val_eval = MetricsEvaluator(config, isTrain=False)

    batch_num = len(val_loader)
    model.eval()
    out_path = os.path.join(res_path, 'results_vis')
    out_path_edgemap = os.path.join(res_path, 'result_vis_edgemap')
    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(out_path_edgemap): os.mkdir(out_path_edgemap)

    sample_lst_file = args.lst
    with open(sample_lst_file, 'r') as f:
        lines = f.readlines()
        test_lst = [os.path.join(args.datapath, line.rstrip() + '.png') for line in lines]
    print('test samples num: {}'.format(len(test_lst)))

    with torch.no_grad():
        end = time.time()

        from PIL import Image

        for batch_idx, img_path in enumerate(test_lst):
            data_time.update(time.time() - end)
            print(img_path)

            # load data
            img = Image.open(img_path, 'r')
            dummy_labels = [np.zeros((1,2,2)),np.zeros((1,2,2))]
            inputs = val_loader.dataset.co_transform({'image': img, 'label': dummy_labels})['image']
            inputs = torch.unsqueeze(inputs, 0)
            net_in  = inputs.cuda(args.gpus[0], non_blocking=True)
            # targets = [target.cuda(args.gpus[0], non_blocking=True) for target in targets]
            img_id = os.path.basename(img_path)[:-4]
            print(img_id)

            # forward
            net_out = model(net_in)

            edge_out, ori_out = torch.sigmoid(net_out[0]), net_out[1]
            edgemap = np.squeeze(edge_out.detach().cpu().numpy())
            orimap = np.squeeze(ori_out.detach().cpu().numpy())

            edge_ori = {}
            edge_ori['edge'] = edgemap  ##[H,W]
            edge_ori['ori'] = orimap
            # save to .mat
            cv2.imwrite(out_path_edgemap + '/' + img_id + '.png', edgemap * 255)
            sio.savemat(out_path + '/' + img_id + '.mat', {'edge_ori': edge_ori})

            batch_time.update(time.time() - end)
            end = time.time()

    return


if __name__ == '__main__':
    main()