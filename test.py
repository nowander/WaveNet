import torch
import torch.nn.functional as F
import sys
import numpy as np
import os
import cv2
# from SFFFF.SSFF7_Res34 import  SFNet
# from sunfanNet.MENet_VGG16_Res34_4_abstract_HIFM import    SFNet
# from Fourth.wave_ablation.wave2_ablation_DWT_384 import SF4_Stu
from Fourth.wave2_SF4_384_share import SF4_Stu#,SF4_Teach
# from Fourth.Uniformer_SF4_384 import SF4_Stu
import matplotlib.pyplot as plt
# from lr.复现的网络.BBS改 import BBSNet
from config import opt
from rgbd_dataset import test_dataset
from datetime import datetime
# from torch.cuda import amp
# from SFNet.SFNet6_Res_NDEM_abstract_MFB import SFNet
# from SFFFF.SSFF6_RES import  SFNet
# from SFFFF.SSFF7_VGG import   SFNet
dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = SF4_Stu()
print('NOW USING:SF4_Stu')
# ICNnet uses 180 epoch
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth/SSSFF6_RES0809/SSFFF6_RES_best_mae_test.pth'))
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth/SSSFF4/SSFFF4_VGG_best_mae_test.pth'))
model.load_state_dict(torch.load('/media/sunfan/date/Paper_4/Wave/PTH_RGBT/Ablation_KD_val1000_RGBT/RES34_1_epoch_30_test.pth'))
# model.load_state_dict(torch.load("/media/sunfan/date/Paper_4/Wave/PTH_RGBT/服务器4/Ablation_CSW_384_val1000_RGBT/RES34_1_epoch_60_test.pth"))
# model.load_pre("/media/sunfan/date/Paper_4/Wave/PTH_KD/Wave_SF4_Stu_RGBT_KD/RES34_1_best_mae_test.pth")
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth/SFNet6_Res_best_mae.pth'))
model.cuda()
model.eval()

#test

test_mae = []
test_datasets = ['VT800','VT1000','VT5000']
# test_datasets = ['RGBT']

for dataset in test_datasets:

    mae_sum  = 0
    save_path = '/media/sunfan/date/Paper_4/Wave/Salient_maps_RGBT/Ablation/Wave_Ablation_KD_30_RGBT/' + dataset + '/'
    # save_path = '/media/sunfan/date/Paper_4/Wave/Salient_maps_RGBT/Ablation/Ablation_SP_60_RGBT/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    # depth_root=dataset_path +dataset +'/parrllex/'
    depth_root = dataset_path + dataset + '/T/'
    #
    # image_root = dataset_path  + '/RGB/'
    # gt_root = dataset_path  + '/GT/'
    # # depth_root=dataset_path +dataset +'/parrllex/'
    # depth_root = dataset_path  + '/T/'

    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    prec_time = datetime.now()
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        # print(image,right,name,Gabor_l,Gabor_r)
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()
        # n, c, h, w = image.size()
        # depth = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        # depth = depth.transpose(3, 1)
        # depth = depth.transpose(3, 2)

        # n, c, h, w = image.size()  # batch_size, channels, height, weight
        # Gabor_l = Gabor_l.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_l = Gabor_l.transpose(3, 1)
        # Gabor_l = Gabor_l.transpose(3, 2)
        #
        # Gabor_r = Gabor_r.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_r = Gabor_r.transpose(3, 1)
        # Gabor_r = Gabor_r.transpose(3, 2)
        # with amp.autocast():
        res = model(image, depth)
        # res = torch.split(res, 1, 1)
        # res = torch.sigmoid(res)
        res = torch.sigmoid(res[0])
        # res = res[0]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_train = torch.sum((torch.abs(res - gt)) * 1.0 / (torch.numel(gt)))
        mae_sum = mae_train.item() + mae_sum
        # print(mae_sum)
        predict = res.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name, )
        predict = cv2.resize(predict,(224,224),interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(save_path + name, predict*255)
        plt.imsave(save_path + name, arr=predict, cmap='gray')

    cur_time = datetime.now()
    test_mae.append(mae_sum / len(test_loader))
h, remainder = divmod((cur_time - prec_time).seconds, 3600)
m, s = divmod(remainder, 60)
# fps = len(test_loader) / (m * 60 + s)  test_loader.size
fps = test_loader.size / (m * 60 + s)
time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print(time_str, 'fps: ', fps)

print('Test_mae:', test_mae)
print('Test Done!')