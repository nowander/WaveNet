from PIL import Image
import cv2
# # def rgb_loader(path):
# #     with open(path, 'rb') as f:
# #         img = Image.open(f)
# #         return img.convert('RGB')
root = '/home/sunfan/Desktop/colorization-master/colorization-master/imgs/ansel_adams3.jpg'
save_path = '/home/sunfan/Desktop/save1.png'
img_gray = cv2.imread(root, flags = 0)
img2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
cv2.imwrite(save_path, img2)
cv2.imshow('test', img2)
# #
# #
# # cv2.waitKey(0)
# #
# # cv2.destroyAllWindows()
#
# # img = Image.open(root)
# # rgb = img.convert('RGB')
#
# # cv2.imwrite(save_path,img2)
#
#
#
# import numpy as np
# import cv2
# root = '/home/sunfan/Desktop/colorization-master/colorization-master/imgs/ansel_adams3.jpg'
# root = '/home/sunfan/Downloads/newdata/train/DUT_NJUNLPR/depth/1_02-02-40.png'
#
# save_path = '/home/sunfan/Desktop/c.png'
# src_gray = cv2.imread(root,flags = 0)
# print(src_gray.shape)
# src = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)
# print(src.shape)
# # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
# B = src[:,:,0]
# G = src[:,:,1]
# R = src[:,:,2]
# # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
# g = src_gray[:]
#
# p = 1; q = 1; t = 1
# B_new = (g-p*R-q*G)/t
# G_new1 = (g-p*R+q*G)/t
# R_new = R
#
# # B_new = np.uint8(B_new)
# src_new = np.zeros((src.shape)).astype("uint8")
# src_new[:,:,0] = B_new
# src_new[:,:,1] = G_new1
# src_new[:,:,2] = R_new
# # 显示图像
# cv2.imshow("input", src_gray)
# cv2.imshow("output", src)
# cv2.imshow("result", src_new)
# cv2.imwrite(save_path, src_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

# Show images
