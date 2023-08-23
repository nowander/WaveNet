import cv2
import os
dst = '/media/sunfan/date/Paper_4/RGBD_DATA/test/Boundary/'
isExist = os.path.exists(dst)
if not isExist:
    os.makedirs(dst)
    print('folder created')
else:
    print('folder exist')
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os

gtTest = os.listdir('/media/sunfan/date/Paper_4/RGBD_DATA/test/GT')
gtTest = [os.path.join('/media/sunfan/date/Paper_4/RGBD_DATA/test/GT', gtimg) for gtimg in gtTest]

for img in gtTest:
    # data = Image.open(img)
    # arr = np.asarray(data)
    arr = cv2.imread(img, 0)
    sobelx = cv2.Laplacian(arr, cv2.CV_64FC3, ksize=3)
    sobely = cv2.Laplacian(arr, cv2.CV_64FC3, ksize=3)
    gm = cv2.sqrt(sobelx ** 2, sobely ** 2)
    print(gm.shape)
    name = img.split('/')[-1]
    cv2.imwrite(dst + name, gm)
