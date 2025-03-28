import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.dev2 import GeleNet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

from data import test_dataset
import time
import imageio

# torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# build models
model = GeleNet(channel=16).to(device)
def count_parameters(model):
    para =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(para) # 统计当前模型的参数量
count_parameters(model)

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# image_root = './dataset/train_dataset/ORSSD/train/image/'
# gt_root = './dataset/train_dataset/ORSSD/train/GT/'
image_root = './data/EORSSD/train-images/'
gt_root = './data/EORSSD/train-labels/'
image_root_test = './data/EORSSD/test-images/'
gt_root_test = './data/EORSSD/test-labels/'
model_save_path = './models/106.pth'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)


def evaluate(model):
    model.eval()
    print('evalution start!')
    test_loader = test_dataset(image_root_test, gt_root_test, 352)
    dataset_path = './data/'
    test_datasets = ['EORSSD']

    for dataset in test_datasets:
        # save_path = './models/GeleNet/' + dataset + '/'
        # save_path = './models/GeleNet/EORSSD1/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        time_sum = 0
        mae=0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)
            time_start = time.time()
            res, sal_sig = model(image)
            time_end = time.time()
            time_sum = time_sum+(time_end-time_start)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # imageio.imsave(save_path+name, res.astype('uint8'))
            # if i == test_loader.size-1:
            #     print('Running time {:.5f}'.format(time_sum/test_loader.size))
            #     print('FPS {:.5f}'.format(test_loader.size / time_sum))
            mae += np.mean(np.abs(res - gt))
        print('mae:',mae/test_loader.size)
        return mae/test_loader.size


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_train_loss = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.to(device)
        gts = gts.to(device)

        sal, sal_sig = model(images)
        loss = CE(sal, gts) + IOU(sal_sig, gts)

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        losses = loss.item()
        total_train_loss += losses
        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))
    # 打印平均loss
    avg_train_loss = total_train_loss / len(train_loader)
    print('average train loss:', avg_train_loss)

    # # save the model
    # save_path = 'models/GeleNet/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), save_path + 'GeleNet.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)



print("Let's go!")
min_mae = 1
min_epoch = 0
for epoch in range(1, opt.epoch+1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    # 监督模型
    t = evaluate(model)
    if t < min_mae:# 保存结果最好的模型
        min_mae = t
        min_epoch = epoch
        torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
    print("The best test_mae is {} in epoch {}".format(min_mae,min_epoch))

print("The best test_mae is {} in epoch {}".format(min_mae,min_epoch))