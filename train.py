# --------------------------------------------------------
# Pytorch Diversify and Match
# Licensed under The MIT License [see LICENSE for details]
# Written by Taeykyung Kim based on codes from Jiasen Lu, Jianwei Yang, and Ross Girshick

# train command: python train.py --dataset clipart --net vgg16
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from roi_da_data_layer.create_loader import create_dataloader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.DivMatch_vgg16 import vgg16
from model.faster_rcnn.DivMatch_resnet import resnet
from test import evaluate
import shutil

step=80000##############################################################################################################
eval_interval=100######################################################################################################1000


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--steps', type=int, default=step, metavar='N',
                        help='maximum number of iterations '
                             'to train (default: 80000)')
    # parser.add_argument('--dataset', dest='dataset',
    #                     help='training dataset',
    #                     default="clipart", type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default="real", type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)######################################################################
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)##########################################################################
    parser.add_argument('--save_interval', dest='save_interval',
                        help='number of iterations to display',
                        default=1000, type=int)##########################################################################
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    # parser.add_argument('--cuda', dest='cuda',
    #                     help='whether use CUDA',
    #                     action='store_true')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True, type=bool)

    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)######################################################################
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=34, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=10022, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)

    parser.add_argument("--evaluation_interval", type=int, default=eval_interval, help="interval evaluations on validation set")#########################

    args = parser.parse_args()
    return args

def input2loss(data, need_backprop, dc_label, step, disp_interval):
    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(data[2].size()).copy_(data[2])
    num_boxes.data.resize_(data[3].size()).copy_(data[3])

    try:
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, DA_loss_dom = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                 need_backprop=need_backprop, dc_label=dc_label)
    except:
        # print("only return DA_loss_dom")
        DA_loss_dom = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                 need_backprop=need_backprop, dc_label=dc_label)

    if need_backprop.numpy():
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
               + DA_loss_dom.mean()

        if (step + 1) % disp_interval == 0:
            print('rpn_cls: ', '%0.5f' % rpn_loss_cls.cpu().data.numpy(),
                  ' | rpn_box: ', '%0.5f' % rpn_loss_box.cpu().data.numpy(),
                  ' | RCNN_cls: ', '%0.5f' % RCNN_loss_cls.cpu().data.numpy(),
                  ' | RCNN_bbox: ', '%0.5f' % RCNN_loss_bbox.cpu().data.numpy(),
                  ' | DA_dom: ', '%0.5f' % DA_loss_dom.cpu().data.numpy()
                  )
        return loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom
    else:
        loss = DA_loss_dom.mean()

        if (step + 1) % disp_interval == 0:
            print('rpn_cls: ', '%0.5f' % 0,
                  ' | rpn_box: ', '%0.5f' % 0,
                  ' | RCNN_cls: ', '%0.5f' % 0,
                  ' | RCNN_bbox: ', '%0.5f' % 0,
                  ' | DA_dom: ', '%0.5f' % DA_loss_dom.cpu().data.numpy()
            )
        return loss, DA_loss_dom


if __name__ == '__main__':

    # delete test output fodler
    delete_path1="test_output"
    if os.path.exists(delete_path1):
        shutil.rmtree(delete_path1)
    delete_path2=os.path.join("datasets", "real", "annotations_cache")
    if os.path.exists(delete_path2):
        shutil.rmtree(delete_path2)
    delete_path3=os.path.join("data", "cache")
    if os.path.exists(delete_path3):
        shutil.rmtree(delete_path3)

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.use_tfboard:
        # from model.utils.logger import Logger
        # # Set the logger
        # logger = Logger('./logs')

        from model.utils.logger_txx import Logger
        # Set the logger
        logger = Logger('./logs/divmatch_')

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if args.dataset in ["clipart", "watercolor", "comic"]:
    if args.dataset in ["real"]:
        print(args.dataset)
        # args.imdb_name = "voc_integrated_trainval"
        args.imdb_name_source = "source_trainval"
        # args.imdbval_name = args.dataset + "_train"
        args.imdb_name_target = "target_train"
        args.imdb_name_targetval = "targetval_test"
        # args.imdb_shifted1_name = args.dataset + "CP_trainval"
        # args.imdb_shifted2_name = args.dataset + "R_trainval"
        args.imdb_shifted3_name = "sourceCPR_trainval"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # elif args.dataset == "cityscapes":
    #     args.imdb_name = "cityscapes_train"
    #     args.imdbval_name = "foggy_cityscapes_val"
    #     args.imdbguide_name = "CityscapesCP_train"
    #     args.imdbguide2_name = "CityscapesR_train"
    #     args.imdbguide3_name = "CityscapesCPR_train"
    #     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ####################################################################################################################
    ################################################## load train set ##################################################
    ####################################################################################################################
    cfg.TRAIN.USE_FLIPPED = False#################################
    cfg.USE_GPU_NMS = args.cuda

    print(args.imdb_name_source, args.imdb_name_target, args.imdb_shifted3_name)
    print('---------------------------------------------------------------------------------')
    print(args.imdb_name_source)
    dataloader, train_size, imdb = create_dataloader(args.imdb_name_source, args)
    print('---------------------------------------------------------------------------------')
    print(args.imdb_name_target)
    dataloader2, train_size2, _ = create_dataloader(args.imdb_name_target, args ,training_flag=False)
    print('---------------------------------------------------------------------------------')
    # dataloader3, train_size3, _ = create_dataloader(args.imdb_shifted1_name, args)
    # print('---------------------------------------------------------------------------------')
    # dataloader4, train_size4, _ = create_dataloader(args.imdb_shifted2_name, args)
    # print('---------------------------------------------------------------------------------')
    print(args.imdb_shifted3_name)
    dataloader5, train_size5, _ = create_dataloader(args.imdb_shifted3_name, args)
    print('---------------------------------------------------------------------------------')

    ####################################################################################################################
    ########################################### initialize the tensor holder ###########################################
    ####################################################################################################################
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    dc_label = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        dc_label = dc_label.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    dc_label = Variable(dc_label)

    if args.cuda:
        cfg.CUDA = True

    ####################################################################################################################
    ################################################### load network ###################################################
    ####################################################################################################################
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'Dis' in key:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr *10 * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else :
                    params += [{'params': [value], 'lr': lr*10 , 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

            else:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                              'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()
    optimizer.zero_grad()
    fasterRCNN.zero_grad()

    iters_per_epoch = int(train_size / args.batch_size)
    iters_per_epoch2 = int(train_size2 / args.batch_size)
    first = 1
    count = 0
    train_end = False

    sum_loss=[0,0,0,0,0]#source,target,source_shift1,source_shift2,source_shift3
    sum_loss_rpn_cls=[0,0,0,0,0]
    sum_loss_rpn_box=[0,0,0,0,0]
    sum_loss_rcnn_cls=[0,0,0,0,0]
    sum_loss_rcnn_bbox=[0,0,0,0,0]
    sum_loss_DA_dom=[0,0,0,0,0]
    # dataset_name=["source","target","source_shift1","source_shift2","source_shift3",]
    dataset_name = ["source", "target", "source_shift3", ]
    epoch=1
    best_map=0

    for step in range(args.steps):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if (step+1) % iters_per_epoch == 0:####################################### epoch
            epoch+=1
            for i in [0,2,3,4]:
                sum_loss[i] = 0
                sum_loss_rpn_cls[i] = 0
                sum_loss_rpn_box[i] = 0
                sum_loss_rcnn_cls[i] = 0
                sum_loss_rcnn_bbox[i] = 0
                sum_loss_DA_dom[i] = 0

        if step % iters_per_epoch == 0:
            data_iter = iter(dataloader)
            # data_iter3 = iter(dataloader3)
            # data_iter4 = iter(dataloader4)
            data_iter5 = iter(dataloader5)

        if (step+1) % iters_per_epoch2 == 0:
            sum_loss[1] = 0
            sum_loss_rpn_cls[1] = 0
            sum_loss_rpn_box[1] = 0
            sum_loss_rcnn_cls[1] = 0
            sum_loss_rcnn_bbox[1] = 0
            sum_loss_DA_dom[1] = 0

        if step % iters_per_epoch2 == 0:
            data_target_iter = iter(dataloader2)

        # print("epoch:{} - step:{}/{}".format(epoch,step+1,iters_per_epoch))

        if (step + 1) % 3000 == 0:######################################################################################
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        if (step + 1) % args.disp_interval == 0:
            print('[{} iters  / {} iters]'.format(step+1, args.steps))

        # SOURCE
        data = next(data_iter)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()
        # print("*"*10)
        # print(type(loss),type(rpn_loss_cls),type(rpn_loss_box),type(RCNN_loss_cls),type(RCNN_loss_bbox),type(DA_loss_dom))
        # print(loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom)
        # print("*"*10)
        sum_loss[0] += loss.item()
        sum_loss_rpn_cls[0] += rpn_loss_cls.item()
        sum_loss_rpn_box[0] += rpn_loss_box.item()
        sum_loss_rcnn_cls[0] += RCNN_loss_cls.item()
        sum_loss_rcnn_bbox[0] += RCNN_loss_bbox.item()
        sum_loss_DA_dom[0] += DA_loss_dom.item()


        # TARGET
        data2 = next(data_target_iter)
        need_backprop = torch.from_numpy(np.zeros((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(np.zeros((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, DA_loss_dom = input2loss(data2, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()
        sum_loss[1] += loss.item()
        sum_loss_DA_dom[1] += DA_loss_dom.item()

        # guide1
        # data3 = next(data_iter3)
        # need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        # dc_label_tmp = torch.from_numpy(2 * np.ones((2000, 1), dtype=np.float32))
        # dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)
        #
        # loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data3, need_backprop, dc_label, step, args.disp_interval)
        # loss.backward()
        # sum_loss[2] += loss.item()
        # sum_loss_rpn_cls[2] += rpn_loss_cls.item()
        # sum_loss_rpn_box[2] += rpn_loss_box.item()
        # sum_loss_rcnn_cls[2] += RCNN_loss_cls.item()
        # sum_loss_rcnn_bbox[2] += RCNN_loss_bbox.item()
        # sum_loss_DA_dom[2] += DA_loss_dom.item()

        # guide2
        # data4 = next(data_iter4)
        # need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        # dc_label_tmp = torch.from_numpy(3 * np.ones((2000, 1), dtype=np.float32))
        # dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)
        #
        # loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data4, need_backprop, dc_label, step, args.disp_interval)
        # loss.backward()
        # sum_loss[3] += loss.item()
        # sum_loss_rpn_cls[3] += rpn_loss_cls.item()
        # sum_loss_rpn_box[3] += rpn_loss_box.item()
        # sum_loss_rcnn_cls[3] += RCNN_loss_cls.item()
        # sum_loss_rcnn_bbox[3] += RCNN_loss_bbox.item()
        # sum_loss_DA_dom[3] += DA_loss_dom.item()

        # guide3
        data5 = next(data_iter5)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(4 * np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data5, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()
        sum_loss[4] += loss.item()
        sum_loss_rpn_cls[4] += rpn_loss_cls.item()
        sum_loss_rpn_box[4] += rpn_loss_box.item()
        sum_loss_rcnn_cls[4] += RCNN_loss_cls.item()
        sum_loss_rcnn_bbox[4] += RCNN_loss_bbox.item()
        sum_loss_DA_dom[4] += DA_loss_dom.item()

        optimizer.step()
        optimizer.zero_grad()
        fasterRCNN.zero_grad()

        # log every 1000 step
        log_step=500
        if args.use_tfboard:
            if (step + 1) % log_step == 0:##################################################
                divide_value=(step+1)%iters_per_epoch
                if divide_value==0:
                    divide_value=iters_per_epoch
                # for i in [0,1,2,3,4]:
                for i in [0, 1, 2]:
                    logger_info = [
                        ('{}_loss'.format(dataset_name[i]), sum_loss[i]/divide_value),
                        ('{}_loss_rpn_cls'.format(dataset_name[i]), sum_loss_rpn_cls[i]/divide_value),
                        ('{}_loss_rpn_box'.format(dataset_name[i]), sum_loss_rpn_box[i]/divide_value),
                        ('{}_loss_rcnn_cls'.format(dataset_name[i]), sum_loss_rcnn_cls[i]/divide_value),
                        ('{}_loss_rcnn_box'.format(dataset_name[i]), sum_loss_rcnn_bbox[i]/divide_value),
                        ('{}_loss_DA_dom'.format(dataset_name[i]), sum_loss_DA_dom[i]/divide_value),
                    ]
                    logger.list_of_scalars_summary(logger_info, (step+1)//log_step+1)

        # every 1000 steps
        if (step+1) % args.evaluation_interval == 0:
        # if True:
        # # save model every 5000 steps
        # args.save_interval=iters_per_epoch
        # if (step + 1) % args.save_interval == 0:
        #     save_name = os.path.join(output_dir,
        #                            '{}_DivMatch_trainval_{}_{}.pth'.format(args.dataset, args.session, step + 1))##################################
            save_name = os.path.join(output_dir,
                                 'DivMatch_{}_{}_latest.pth'.format(args.dataset, args.net))  ##################################
            save_checkpoint({
              'session': args.session,
              'model': fasterRCNN.state_dict(),
              'optimizer': optimizer.state_dict(),
              'pooling_mode': cfg.POOLING_MODE,
              'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

            print("\n---- Evaluating Model ----")
            map,cls_name_list,ap_list=evaluate(args.dataset, args.net)#clipart,vgg16
            print("map={}".format(map))
            print("ap_list:")
            print(cls_name_list)
            print(["{:.4f}".format(i) for i in ap_list])
            logger.scalar_summary("test_map", map,step + 1)

            if map > best_map:
                print("updating best map...")
                save_name = os.path.join(output_dir,
                                       'divmatch_{}_{}_step{}.pth'.format(args.dataset, args.net, step + 1))##################################
                save_checkpoint({
                    'session': args.session,
                    'model': fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                best_map = map

                # save txt(map,ap)
                save_path_txt = os.path.join(output_dir, 'DivMatch_{}_{}.txt'.format(args.dataset,args.net))
                with open(save_path_txt, "a") as f:
                    f.write("step{:0>5}: mAP={} /best_mAP={}\n".format(step + 1, map, best_map))
                    for cls in cls_name_list:
                        f.write("{} ".format(cls))
                    f.write("\n")
                    for ap in ap_list:
                        f.write("{:.4f} ".format(ap))
                    f.write("\n")
                    f.write("*"*20+"\n")

            print("best_map={}".format(best_map))
            # print("\n")
        # end = time.time()
        # print(end - start)
