import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import time
import IPython, cv2

SSIM_WIN = 5


class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


def gradient_xy(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    
    return gx, gy


def warp_disp(x, disp, args):
    # result + flow(-disp) = x
    # warp back to result
    N, _, H, W = x.shape

    x_ = torch.arange(W).view(1, -1).expand(H, -1)
    y_ = torch.arange(H).view(-1, 1).expand(-1, W)
    grid = torch.stack([x_, y_], dim=0).float()
    if args.cuda:
        grid = grid.cuda()
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (W - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (H - 1) - 1
    # disp = 30*torch.ones(N, H, W).cuda()
    grid2 = grid.clone()
    grid2[:, 0, :, :] = grid[:, 0, :, :] + 2*disp/W
    grid2 = grid2.permute(0, 2, 3, 1)
    
    return F.grid_sample(x, grid2, padding_mode='zeros')


def Loss_prob(y, target, logvar):
    y = y*50
    target = target*50

    thresh = 10
    logvar = logvar.clamp(-50, 50)
    loss = 2**0.5 * F.smooth_l1_loss(y, target, reduction='none').mean(1) * (torch.exp(-logvar) ) + logvar
    loss = loss.clamp(-thresh, thresh).mean()
    return loss


# loss1
# appearance loss: the difference between reconstructed image and original image
def criterion1_prob(imgC, imgR, imgL, outputR, outputR_var, outputL, outputL_var, maxdisp, args):

    imgR2C = warp_disp(imgR, -outputR, args)
    imgL2C = warp_disp(imgL, outputL, args)
    imgR2C2 = warp_disp(imgR, -outputL, args)
    imgL2C2 = warp_disp(imgL, outputR, args)

    alpha = 0.85
    crop_edge = 200
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size = imgC.shape[2])

    # imgC_gx, imgC_gy = gradient_xy(imgC)
    # imgR2C_gx, imgR2C_gy = gradient_xy(imgR2C)
    # imgL2C_gx, imgL2C_gy = gradient_xy(imgL2C)

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgR2C2)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C2)) / 2.0
        # diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC, imgL2C, reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC, imgR2C2, reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC, imgL2C2, reduction='mean'))
        diff_L1 = Loss_prob(imgR2C, imgC, outputR_var) + \
                  Loss_prob(imgL2C, imgC, outputL_var) + \
                  Loss_prob(imgR2C2, imgC, outputL_var) + \
                  Loss_prob(imgL2C2, imgC, outputR_var)
    else:
        diff_ssim = (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge])) / 2.0
        # diff_L1 = (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:], reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge], reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:], reduction='mean')) + \
        #           (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge], reduction='mean'))
        diff_L1 = Loss_prob(imgR2C[:,:,:,crop_edge:], imgC[:,:,:,crop_edge:], outputR_var[:,:,crop_edge:]) + \
                  Loss_prob(imgL2C[:,:,:,:-crop_edge], imgC[:,:,:,:-crop_edge], outputL_var[:,:,:-crop_edge]) + \
                  Loss_prob(imgR2C2[:,:,:,crop_edge:], imgC[:,:,:,crop_edge:], outputL_var[:,:,crop_edge:]) + \
                  Loss_prob(imgL2C2[:,:,:,:-crop_edge], imgC[:,:,:,:-crop_edge], outputR_var[:,:,:-crop_edge])
    
    # loss1 = 1.0/4 * (alpha * diff_ssim + (1-alpha) * diff_L1 * 0.2)

    loss1 = 1.0/4 * diff_ssim

    varR_gx, varR_gy = gradient_xy(outputR_var.unsqueeze(1)*50)
    # varL_gx, varL_gy = gradient_xy(outputL_var.unsqueeze(1)*50)
    intensity_gx, intensity_gy = gradient_xy(imgC)
    weights_x = torch.exp(-10 * torch.abs(intensity_gx).mean(1).unsqueeze(1))
    weights_y = torch.exp(-10 * torch.abs(intensity_gy).mean(1).unsqueeze(1))
    smoothness_x = torch.abs(varR_gx) * weights_x #+ torch.abs(varL_gx) * weights_x
    smoothness_y = torch.abs(varR_gy) * weights_y #+ torch.abs(varL_gy) * weights_y
    var_smooth = smoothness_x.mean() + smoothness_y.mean()

    loss4 = 0.01 * diff_L1 + 0.001 * var_smooth
    # print('%.4f, %.4f, %.4f'%(loss1.item(), 0.005 * diff_L1.item(), 0.001 * var_smooth.item()))
    # loss4 = torch.Tensor([0]).cuda()

    return loss1, loss4, imgR2C, imgL2C, imgC, outputR
                

# loss2
# consistency loss the difference between left output and right output
def criterion2(R, L, R_var=None, L_var=None):
    tau = 10    # truncation for occluded region

    # ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    # ssimloss = (1 - ssim_loss(R.unsqueeze(1), L.unsqueeze(1))) / 2.0
    if(R_var is None):
        return F.smooth_l1_loss(R, L, reduction='none').clamp(min=0, max=tau).mean()

    thresh = 1  # 0.2

    mask = (R_var < thresh) & (L_var < thresh)
    R_mask = (R_var > thresh) & (L_var < thresh/2.0)
    L_mask = (L_var > thresh) & (R_var < thresh/2.0)

    l1 = F.smooth_l1_loss(R[mask], L[mask], reduction='none').clamp(min=0, max=tau)
    l2 = F.smooth_l1_loss(R[R_mask], L.detach()[R_mask], reduction='none').clamp(min=0, max=tau)
    l3 = F.smooth_l1_loss(R.detach()[L_mask], L[L_mask], reduction='none').clamp(min=0, max=tau)

    # L1loss = F.smooth_l1_loss(R, L, reduction='none').clamp(min=0, max=tau).mean()
    # print "%.2f, %.2f, %.2f" % (100.0*len(l1)/R.shape.numel(), 
    #                             100.0*len(l2)/R.shape.numel(), 
    #                             100.0*len(l3)/R.shape.numel())

    return torch.cat([l1, l2, l3]).mean()


# loss3
# smooth loss: force grident of intensity to be small
def criterion3(disp, img):
    disp = disp.unsqueeze(1)
    disp_gx, disp_gy = gradient_xy(disp)
    intensity_gx, intensity_gy = gradient_xy(img)

    weights_x = torch.exp(-10 * torch.abs(intensity_gx).mean(1).unsqueeze(1))
    weights_y = torch.exp(-10 * torch.abs(intensity_gy).mean(1).unsqueeze(1))

    disp_gx = torch.abs(disp_gx)
    gx = disp_gx.clone()
    gx[gx>0.5] = disp_gx[disp_gx>0.5] + 10

    disp_gy = torch.abs(disp_gy)
    gy = disp_gy.clone()
    gy[gy>0.5] = disp_gy[disp_gy>0.5] + 10

    smoothness_x = gx * weights_x
    smoothness_y = gy * weights_y

    return smoothness_x.mean() + smoothness_y.mean()

# loss4
# regularization term: 
def criterion4(disp, maxdisp):
    # r1 = disp.mean()
    # r = torch.exp(-1 / 5.0 * disp) + torch.exp(1 / 5.0 * (disp - 90))
    # r = torch.exp(-1 / 5.0 * disp)
    r = (disp*2/maxdisp - 1).pow(2)
    return r.mean()


def evaluate(model, imgL, imgC, imgR, gt, args, maxd):
    use_cuda = args.cuda
    # use_cuda = False
    height = imgL.shape[1]
    width = imgL.shape[2]
    maxdisp = maxd

    pad_h = (height // 32 + 1) * 32
    pad_w = (width // 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])
    if imgC is not None:
        imgC = np.reshape(imgC, [1, imgC.shape[0], imgC.shape[1], imgC.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    if imgC is not None:
        imgC = np.lib.pad(imgC, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)
    if imgC is not None:
        imgC = torch.from_numpy(imgC)

    model.eval()

    if imgC is not None:
        # multiscopic mode
        imgC_rot = imgC.flip(2).flip(3)
        imgL_rot = imgL.flip(2).flip(3)

        if use_cuda:
            imgL, imgR, imgC, imgC_rot, imgL_rot = \
                    imgL.cuda(), imgR.cuda(), imgC.cuda(), imgC_rot.cuda(), imgL_rot.cuda()
        
        if args.model == 'stackhourglass':
            outputR, outputR_var, _, _ = model(imgC, imgR, maxdisp)
            if args.cuda and (not use_cuda):
                outputR = outputR.cpu()
                outputR_var = outputR_var.cpu()
            outputL_rot, outputL_var_rot, _, _ = model(imgC_rot, imgL_rot, maxdisp)
            outputL = outputL_rot.flip(1).flip(2)
            outputL_var = outputL_var_rot.flip(2).flip(3)
            if args.cuda and (not use_cuda):
                outputL = outputL.cpu()
                outputL_var = outputL_var.cpu()
        elif args.model == 'basic':
            outputR = model(imgC, imgR, maxdisp)
            outputL_rot = model(imgC_rot, imgL_rot)
            outputL = outputL_rot.flip(1).flip(2)

        # mindisp = torch.min(torch.cat([outputR, outputL]), 0)[0]
        # diff = (outputR - outputL).squeeze()
        # outputR = outputR.squeeze()
        # outputL = outputL.squeeze()
        # outputR[diff>3] = mindisp[diff>3]

        disp = outputR.squeeze()[top_pad:, :-left_pad]
        var = outputR_var.squeeze()[top_pad:, :-left_pad]
    
    else:
        # stereo mode
        if use_cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        if args.model == 'stackhourglass':
            output, var, _, _ = model(imgL, imgR, maxdisp)
        elif args.model == 'basic':
            output = model(imgL, imgR, maxdisp)
        if args.cuda and (not use_cuda):
            output = output.cpu()
        disp = output.squeeze()[top_pad:, :-left_pad]
        var = var.squeeze()[top_pad:, :-left_pad]

    gt = torch.from_numpy(gt).float()
    if(use_cuda): gt = gt.cuda()
    mask = (gt != 0)

    diff = torch.abs(disp[mask] - gt[mask])
    avgerr = torch.mean(diff)
    rms = torch.sqrt( (diff**2).mean() ) 
    bad05 = len(diff[diff>0.5])/float(len(diff))
    bad1 = len(diff[diff>1])/float(len(diff))
    bad2 = len(diff[diff>2])/float(len(diff))
    bad3 = len(diff[diff>3])/float(len(diff))

    return [avgerr.data.item(), rms.data.item(), bad05, bad1, bad2, bad3], disp.cpu().numpy(), var.cpu().numpy()


def evaluate_kitti(model, imgL, imgR, gt_occ, gt_noc, args, maxd=160):
    height = imgL.shape[1]
    width = imgL.shape[2]
    maxdisp = maxd

    pad_h = (height / 32 + 1) * 32
    pad_w = (width / 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)

    model.eval()
    
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    if args.model == 'stackhourglass':
        output, var, _, _ = model(imgL, imgR, maxdisp)
    elif args.model == 'basic':
        output = model(imgL, imgR, maxdisp)

    disp = output.squeeze()[top_pad:, :-left_pad]
    var = var.squeeze()[top_pad:, :-left_pad]

    if gt_noc is None:
        return disp.cpu().numpy(), var.cpu().numpy()

    gt_occ = torch.from_numpy(gt_occ).float()
    gt_noc = torch.from_numpy(gt_noc).float()
    if args.cuda:
        gt_noc = gt_noc.cuda()
        gt_occ = gt_occ.cuda()
    mask_occ = (gt_occ != 0)
    mask_noc = (gt_noc != 0)

    mask_o = mask_occ.clone()
    mask_o[gt_noc != 0] = 0

    diff_occ = torch.abs(disp[mask_occ] - gt_occ[mask_occ])
    diff_noc = torch.abs(disp[mask_noc] - gt_noc[mask_noc])
    # bad3_occ = len(diff_occ[diff_occ>3])/float(len(diff_occ))
    # bad3_noc = len(diff_noc[diff_noc>3])/float(len(diff_noc))

    bad3_occ = torch.sum((diff_occ>3) & (diff_occ/gt_occ[mask_occ]>0.05)).float() / float(len(diff_occ))
    bad3_noc = torch.sum((diff_noc>3) & (diff_noc/gt_noc[mask_noc]>0.05)).float() / float(len(diff_noc))

    if(mask_o.sum()>0):
        diff_o = torch.abs(disp[mask_o] - gt_occ[mask_o])
        bad3_o = torch.sum((diff_o>3) & (diff_o/gt_occ[mask_o]>0.05)).float() / float(len(diff_o))
    else:
        diff_o = torch.Tensor([0]).cuda()
        bad3_o = torch.Tensor([0]).cuda().sum()

    return [bad3_occ, bad3_noc, bad3_o, diff_o.mean()], disp.cpu().numpy(), var.cpu().numpy()


def predict(model, imgL, imgR, args, maxd):
    height = imgL.shape[1]
    width = imgL.shape[2]

    pad_h = (height / 32 + 1) * 32
    pad_w = (width / 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)

    model.eval()

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
    
    if args.model == 'stackhourglass':
        output, _, _, _ = model(imgL, imgR, maxd)
        
    elif args.model == 'basic':
        output = model(imgL, imgR, maxd)

    disp = output.squeeze()[top_pad:, :-left_pad]

    return disp.cpu().numpy()