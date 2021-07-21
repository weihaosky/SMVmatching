from __future__ import print_function
import argparse, os, random, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
import cv2

from dataloader import datalist as ls
from dataloader import loader as DA
from dataloader import preprocess, readpfm
from models import basic, stackhourglass_prob
from unsuper_prob_utils import criterion1_prob, criterion2, criterion3, criterion4, \
                          evaluate, evaluate_kitti, predict,    \
                          WrappedModel

# set gpu id used
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"


def train(imgL, imgC, imgR, args, Test=False):
    if not Test:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if args.cuda:
        imgL, imgR, imgC = \
                imgL.cuda(), imgR.cuda(), imgC.cuda()

    imgC_rot = imgC.flip(2).flip(3)
    imgL_rot = imgL.flip(2).flip(3)

    if args.model == 'stackhourglass':
        if not Test:
            outputR_pred1, outputR_pred2, outputR, outputR_var = model(imgC, imgR, args.maxdisp)
            outputL_rot_pred1, outputL_rot_pred2, outputL_rot, outputL_var_rot = model(imgC_rot, imgL_rot, args.maxdisp)
            outputL = outputL_rot.flip(1).flip(2) 
            outputL_var = outputL_var_rot.flip(1).flip(2)  
        else:
            outputR, outputR_var, outputR_pred1, outputR_pred2 = model(imgC, imgR, 208)
            outputL_rot, outputL_var_rot, outputL_rot_pred1, outputL_rot_pred2 = model(imgC_rot, imgL_rot, 160)
            outputL = outputL_rot.flip(1).flip(2)
            outputL_var = outputL_var_rot.flip(1).flip(2) 
            
    elif args.model == 'basic':
        outputR = model(imgC, imgR, args.maxdisp)
        # outputR = torch.unsqueeze(outputR, 1)
        outputL_rot = model(imgC_rot, imgL_rot, args.maxdisp)
        # outputL_rot = torch.unsqueeze(outputL_rot, 1)
        outputL = outputL_rot.flip(1).flip(2)

    loss2 = criterion2(outputR, outputL, outputR_var, outputL_var, args)
    
    # appearance loss
    loss1, loss4, imgR2C, imgL2C, _, _ = criterion1_prob(
            imgC, imgR, imgL, outputR, outputR_var, outputL, outputL_var, args.maxdisp, args=args) 

    loss3 = (criterion3(outputR, imgC) + criterion3(outputL, imgC)) / 2


    loss1 = loss_w[0] * loss1
    loss2 = loss_w[1] * loss2
    loss3 = loss_w[2] * loss3
    # loss4 = loss_w[3] * loss4
    loss = loss1 + loss2 + loss3 + loss4

    if not Test:
        loss.backward()
        optimizer.step()

    return loss.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--trainpath', default='dataset/TRAIN/',
                        help='training data set path')
    parser.add_argument('--testpath', default='dataset/TEST/',
                        help='test data set path')
    parser.add_argument('--evalpath', default='dataset/EVAL/',
                        help='evaluate data set path')
    parser.add_argument('--model', default='stackhourglass',
                        help='select model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='maxium epochs')
    parser.add_argument('--maxdisp', type=int, default=208,
                        help='maxium disparity')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--savemodel', default='./trained',
                    help='save model')
    parser.add_argument('--resume', type=int, default=0, 
                    help='if resume from previous model (default: Non)')
    parser.add_argument('--resume_model', default=None, 
                    help='previous model to resume (default: None)')
    parser.add_argument('--name', default='1', 
                    help='name for saving log')
    parser.add_argument('--prob_mode', type=int, default=2, 
                    help='name for saving log')
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    log_path = 'log/' + args.name + '/'
    output_path = log_path + "output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + 'test/'):
        os.makedirs(output_path + 'test/')
    if not os.path.exists(output_path + 'eval/'):
        os.makedirs(output_path + 'eval/')

    if args.model == 'stackhourglass':
        model = stackhourglass_prob(args.maxdisp, args.prob_mode)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    else:
        print('no model')

    if args.cuda:
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.cuda()

    epoch_begin = 0
    if args.resume:   
        epoch_begin = args.resume
        if args.resume_model:
            if(args.cuda == False): model = WrappedModel(model)
            model.load_state_dict(torch.load(args.resume_model))
            # model.load_state_dict(torch.load(args.resume_model)['state_dict'])
            if(args.cuda == False): model = model.module
        else:
            model.load_state_dict(torch.load(log_path + 'model' + str(args.resume) + '.pth'))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    all_left_img, all_center_img, all_right_img = ls.dataloader(args.trainpath, mode='3frame')
    test_left_img, test_center_img, test_right_img = ls.dataloader(args.testpath, mode='3frame')

    IS_augmentation = False
    TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img, all_center_img, all_right_img, training=True, augment=IS_augmentation), 
         batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img, test_center_img, test_right_img, training=False, augment=False), 
         batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    print("Train dataset size:", len(all_left_img))
    print("Test dataset size:", len(test_left_img))
    print("Is augmentation:", IS_augmentation)

    processed1 = preprocess.get_transform(augment=False)

    # ========================= Optimization Setup ==============================
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # loss weights
    loss_w = [1, 0.01, 0.03, 0]

    print('loss_w:', loss_w)

    train_loss_record = []
    test_loss_record = []
    eval_record1 = []
    eval_record3 = []
    eval_record_kitti = []
    start_full_time = time.time()
    Test_epoch = 10

    for epoch in range(epoch_begin+1, args.epochs+1):
        total_train_loss = 0
        total_train_loss1 = 0
        total_train_loss2 = 0
        total_train_loss3 = 0
        total_train_loss4 = 0
        total_test_loss = 0
        total_test_loss1 = 0
        total_test_loss2 = 0
        total_test_loss3 = 0     
        total_test_loss4 = 0
        time_epoch = 0   
        # adjust_learning_rate(optimizer,epoch)

        ## ====================== training =======================
        for batch_idx, (imgL_crop, imgC_crop, imgR_crop) in enumerate(TrainImgLoader):
            start_time = time.time() 

            loss, loss1, loss2, loss3, loss4 = train(imgL_crop, imgC_crop, imgR_crop, args)
            # print('Iter %d training loss = %.3f , loss1 = %.3f, loss2 = %.3f, loss3 = %.3f, loss4 = %.3f, time = %.2f' 
            #           %(batch_idx, loss, loss1, loss2, loss3, loss4, time.time() - start_time))
            total_train_loss += loss
            total_train_loss1 += loss1
            total_train_loss2 += loss2
            total_train_loss3 += loss3
            total_train_loss4 += loss4
            time_epoch += time.time() - start_time

        loss_mean = [total_train_loss/len(TrainImgLoader), 
                    total_train_loss1/len(TrainImgLoader),
                    total_train_loss2/len(TrainImgLoader),
                    total_train_loss3/len(TrainImgLoader),
                    total_train_loss4/len(TrainImgLoader)]
        train_loss_record.append(loss_mean)
        print('epoch %d mean training loss = %.3f, loss1 = %.3f, loss2 = %.3f, loss3 = %.3f, loss4 = %.3f, time = %.2f'
                % (epoch, loss_mean[0], loss_mean[1], loss_mean[2], loss_mean[3], loss_mean[4], time_epoch/len(TrainImgLoader)) )
        with open(log_path + "loss.log", "a") as file:
            file.write('epoch %d mean training loss = %.3f, loss1 = %.3f, loss2 = %.3f, loss3 = %.3f, loss4 = %.3f, time = %.2f \n'
                % (epoch, loss_mean[0], loss_mean[1], loss_mean[2], loss_mean[3], loss_mean[4], time_epoch/len(TrainImgLoader)))
        
        # ======================== Test ==========================
        if epoch % Test_epoch == 0:
            for batch_idx, (imgL, imgC, imgR) in enumerate(TestImgLoader):
                with torch.no_grad():
                    test_loss, test_loss1, test_loss2, test_loss3, test_loss4 = train(imgL, imgC, imgR, args, Test=True)

                total_test_loss += test_loss
                total_test_loss1 += test_loss1
                total_test_loss2 += test_loss2
                total_test_loss3 += test_loss3
                total_test_loss4 += test_loss4
                torch.cuda.empty_cache()
            loss_mean = [total_test_loss/len(TestImgLoader), 
                        total_test_loss1/len(TestImgLoader),
                        total_test_loss2/len(TestImgLoader),
                        total_test_loss3/len(TestImgLoader),
                        total_test_loss4/len(TestImgLoader)]
            test_loss_record.append(loss_mean)
            print('epoch %d test loss = %.3f, loss1 = %.3f, loss2 = %.3f, loss3 = %.3f, loss4 = %.3f'
                % (epoch, loss_mean[0], loss_mean[1], loss_mean[2], loss_mean[3], loss_mean[4]) )
            with open(log_path + "loss.log", "a") as file:
                file.write('epoch %d test loss = %.3f, loss1 = %.3f, loss2 = %.3f, loss3 = %.3f, loss4 = %.3f \n'
                    % (epoch, loss_mean[0], loss_mean[1], loss_mean[2], loss_mean[3], loss_mean[4]))


        if epoch % 10 == 0:
            torch.save(model.state_dict(), log_path + 'model' + str(epoch) + '.pth')

            # --------------- evaluate kitti2015 ---------------------
            kitti_train = os.listdir(args.evalpath + 'KITTI2015/disp_occ_0/')
            eval_res = []
            for img_name in kitti_train:
                img1 = processed1(DA.default_loader(args.evalpath + 'KITTI2015/image_2/' + img_name)).numpy()
                img2 = processed1(DA.default_loader(args.evalpath + 'KITTI2015/image_3/' + img_name)).numpy()
                gt_noc = cv2.imread(args.evalpath + 'KITTI2015/disp_noc_0/' + img_name, cv2.IMREAD_ANYDEPTH)/256.0
                gt_occ = cv2.imread(args.evalpath + 'KITTI2015/disp_occ_0/' + img_name, cv2.IMREAD_ANYDEPTH)/256.0
                with torch.no_grad():
                    res, disp, var = evaluate_kitti(model, img1, img2, gt_occ, gt_noc, args, maxd=160)
                cv2.imwrite(output_path + 'eval/' + img_name, (disp*256).astype(np.uint16))
                cv2.imwrite(output_path + 'eval/' + img_name.split('.')[0] + '_var.png', var*255)
                eval_res.append(res)
            eval_res = np.array(eval_res).mean(0)
            eval_record_kitti.append(eval_res)
            print('epoch %d train occ bad3 = %.4f, noc bad3 = %.4f' % (epoch, eval_res[0], eval_res[1]) )
            with open(log_path + "loss.log", "a") as file:
                file.write( 'epoch %d train occ bad3 = %.4f, noc bad3 = %.4f \n' % (epoch, eval_res[0], eval_res[1]) )
            torch.cuda.empty_cache()


        