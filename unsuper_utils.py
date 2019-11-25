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

# loss1
# appearance loss: the difference between reconstructed image and original image
def criterion1(imgC, imgR, imgL, outputR, outputL, maxdisp, args, down_factor=1):
    if down_factor != 1:
        imgC = F.interpolate(imgC, scale_factor=1.0/down_factor, mode='bicubic')
        imgR = F.interpolate(imgR, scale_factor=1.0/down_factor, mode='bicubic')
        imgL = F.interpolate(imgL, scale_factor=1.0/down_factor, mode='bicubic')
        outputR = F.interpolate(outputR.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor
        outputL = F.interpolate(outputL.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor

        outputR = outputR.squeeze(1)
        outputL = outputL.squeeze(1)

    imgR2C = warp_disp(imgR, -outputR, args)
    imgL2C = warp_disp(imgL, outputL, args)
    imgR2C2 = warp_disp(imgR, -outputL, args)
    imgL2C2 = warp_disp(imgL, outputR, args)

    alpha2 = 0.85
    crop_edge = 200
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size = imgC.shape[2])

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgR2C2)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C2)) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgL2C, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgR2C2, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgL2C2, reduction='mean'))
    else:
        diff_ssim = (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge])) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge], reduction='mean'))
    
    loss1 = 1.0/4 * (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    
    return loss1, imgR2C, imgL2C, imgC, outputR

def criterion1_2frame(imgC, imgR, outputR, maxdisp, args, down_factor=1):
    if down_factor != 1:
        imgC = F.interpolate(imgC, scale_factor=1.0/down_factor, mode='bicubic')
        imgR = F.interpolate(imgR, scale_factor=1.0/down_factor, mode='bicubic')
        outputR = F.interpolate(outputR.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor
        
        outputR = outputR.squeeze(1)

    imgR2C = warp_disp(imgR, -outputR, args)

    alpha2 = 0.85
    crop_edge = 0
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size = imgC.shape[2])

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean'))
    else:
        diff_ssim = (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:])) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:], reduction='mean'))
    
    loss1 = (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    
    return loss1, imgR2C
                

# loss2
# consistency loss the difference between left output and right output
def criterion2(R, L):
    alpha1 = 0
    tau = 10    # truncation for occluded region

    L1loss = F.smooth_l1_loss(R, L, reduction='none').clamp(min=0, max=tau).mean()

    return L1loss

    # R = R.unsqueeze(1)
    # L = L.unsqueeze(1)
    # R_gx, R_gy = gradient_xy(R)
    # L_gx, L_gy = gradient_xy(L)
    # gxloss = F.smooth_l1_loss(R_gx, L_gx, reduction='none').clamp(min=0, max=tau).mean()
    # gyloss = F.smooth_l1_loss(R_gy, L_gy, reduction='none').clamp(min=0, max=tau).mean()
    # g1loss = 0.5 * (gxloss + gyloss)
    # R_gxx, R_gxy = gradient_xy(R_gx)
    # R_gyx, R_gyy = gradient_xy(R_gy)
    # L_gxx, L_gxy = gradient_xy(L_gx)
    # L_gyx, L_gyy = gradient_xy(L_gy)
    # gxxloss = F.smooth_l1_loss(R_gxx, L_gxx, reduction='none').clamp(min=0, max=tau).mean()
    # gyyloss = F.smooth_l1_loss(R_gyy, L_gyy, reduction='none').clamp(min=0, max=tau).mean()
    # g2loss = 0.5 * (gxxloss + gyyloss)

    # return 0.5 * (L1loss + (g1loss*10 + g2loss*10)/2.0 ) 


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
            outputR, outputR_prob, _, _ = model(imgC, imgR, maxdisp)
            if args.cuda and (not use_cuda):
                outputR = outputR.cpu()
                outputR_prob = outputR_prob.cpu()
            outputL_rot, outputL_prob_rot, _, _ = model(imgC_rot, imgL_rot, maxdisp)
            outputL = outputL_rot.flip(1).flip(2)
            outputL_prob = outputL_prob_rot.flip(2).flip(3)
            if args.cuda and (not use_cuda):
                outputL = outputL.cpu()
                outputL_prob = outputL_prob.cpu()
        elif args.model == 'basic':
            outputR = model(imgC, imgR, maxdisp)
            outputL_rot = model(imgC_rot, imgL_rot)
            outputL = outputL_rot.flip(1).flip(2)

        mindisp = torch.min(torch.cat([outputR, outputL]), 0)[0]
        diff = (outputR - outputL).squeeze()
        outputR = outputR.squeeze()
        outputL = outputL.squeeze()
        outputR[diff>3] = mindisp[diff>3]

        disp = outputL
        disp = disp[top_pad:, :-left_pad]
    
    else:
        # stereo mode
        if use_cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        if args.model == 'stackhourglass':
            output, _, _, _ = model(imgL, imgR, maxdisp)
        elif args.model == 'basic':
            output = model(imgL, imgR, maxdisp)
        if args.cuda and (not use_cuda):
            output = output.cpu()
        disp = output.squeeze()[top_pad:, :-left_pad]

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

    return [avgerr.data.item(), rms.data.item(), bad05, bad1, bad2, bad3], disp.cpu().numpy()


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
        output, _, _, _ = model(imgL, imgR, maxdisp)
    elif args.model == 'basic':
        output = model(imgL, imgR, maxdisp)

    disp = output.squeeze()[top_pad:, :-left_pad]

    if gt_noc.any() == None:
        return disp.cpu().numpy()

    gt_occ = torch.from_numpy(gt_occ).float()
    gt_noc = torch.from_numpy(gt_noc).float()
    if args.cuda:
        gt_noc = gt_noc.cuda()
        gt_occ = gt_occ.cuda()
    mask_occ = (gt_occ != 0)
    mask_noc = (gt_noc != 0)

    diff_occ = torch.abs(disp[mask_occ] - gt_occ[mask_occ])
    diff_noc = torch.abs(disp[mask_noc] - gt_noc[mask_noc])
    # bad3_occ = len(diff_occ[diff_occ>3])/float(len(diff_occ))
    # bad3_noc = len(diff_noc[diff_noc>3])/float(len(diff_noc))

    bad3_occ = torch.sum((diff_occ>3) & (diff_occ/gt_occ[mask_occ]>0.05)).float() / float(len(diff_occ))
    bad3_noc = torch.sum((diff_noc>3) & (diff_noc/gt_noc[mask_noc]>0.05)).float() / float(len(diff_noc))

    return [bad3_occ, bad3_noc], disp.cpu().numpy()


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