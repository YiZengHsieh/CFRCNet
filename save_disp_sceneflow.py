from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from dataset import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
from utilsx.experiment import tensor2float,save_scalars,save_images,AverageMeterDict,tensor2numpy

# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='msacvnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
parser.add_argument('--datapath', default="/media/cihci/0000678400004823/andy_master/code/", help='data path')
parser.add_argument('--testlist',default="./filenames/demo.txt", help='testing list')
# parser.add_argument('--testlist',default='./filenames/test_192_one.txt', help='testing list')
# parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
# parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt', default='./log_dir/20220808_2_30epoch/chekpoint_000029.ckpt',help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
# test_dataset = StereoDataset(args.datapath,args.datapath, args.testlist, False)

TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=24, drop_last=False)

# model, optimizer
# model = __models__[args.model](args.maxdisp, False, False)

model = __models__[args.model](args.maxdisp,False,False)
model = nn.DataParallel(model)
model.cuda()

#load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

save_dir = '/media/cihci/0000678400004823/andy_master/code/demo/output'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        left_filenames = sample["left_filename"]
        if left_filenames is not None:
            for disp_est, fn in zip(disp_est_np, left_filenames):
                assert len(disp_est.shape) == 2
                disp_est = np.array(disp_est, dtype=np.float32)
                print(fn)
                fn = fn.split("/left/")
                fn = fn[1]
                # fn = os.path.join(save_dir, fn.split('/')[-4]+fn.split('/')[-3]+fn.split('/')[-1])
                fn = os.path.join(save_dir, fn)

                print(fn)

                # fn = os.path.join(save_dir, fn.split('/')[-3]+fn.split('/')[-1])
                # print(fn)

                print("saving to", fn, disp_est.shape)
                disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
                # disp_est_uint = np.round(disp_est * 256).astype(np.uint8)
                # img = cv2.resize(disp_est_uint, dsize=(1920,1080), interpolation=cv2.INTER_CUBIC)
                skimage.io.imsave(fn, disp_est_uint)
                # cv2.imwrite(fn, img)

                cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))
                print("ok")
                # cv2.imwrite("./disp.png", cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR)



    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
