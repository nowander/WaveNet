import torch

import os
import cv2

from networks.Wavenet import WaveNet

import matplotlib.pyplot as plt
from config import opt
from rgbt_dataset_KD import test_dataset
from datetime import datetime

dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = WaveNet()
print('NOW USING:WaveNet')
model.load_state_dict(torch.load(''))

model.cuda()
model.eval()

#test

test_mae = []
test_datasets = ['VT800','VT1000', 'VT5000']

for dataset in test_datasets:

    mae_sum  = 0
    save_path = 'show/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/T/'

    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    prec_time = datetime.now()
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()

        res = model(image, depth)
        res = torch.sigmoid(res[0])
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_train = torch.sum((torch.abs(res - gt)) * 1.0 / (torch.numel(gt)))
        mae_sum = mae_train.item() + mae_sum
        predict = res.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name, )
        predict = cv2.resize(predict,(224,224),interpolation=cv2.INTER_LINEAR)
        plt.imsave(save_path + name, arr=predict, cmap='gray')

    cur_time = datetime.now()
    test_mae.append(mae_sum / len(test_loader))

print('Test_mae:', test_mae)
print('Test Done!')