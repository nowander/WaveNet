import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label, depth,bound):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        bound = bound.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth,bound


def randomCrop(image, label, depth,bound):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region),bound.crop(random_region)


def randomRotation(image, label, depth,bound):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        bound = bound.rotate(random_angle, mode)
    return image, label, depth,bound


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, bound_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png') or f.endswith('.jpg')]
        self.bound = [bound_root + f for f in os.listdir(bound_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        # print(len(self.images))
        self.gts = sorted(self.gts)
        # print(len(self.gts))
        self.depths = sorted(self.depths)
        self.bound = sorted(self.bound)
        # print(len(self.depths))
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.bound_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # depth = self.binary_loader(self.depths[index])
        depth = self.rgb_loader(self.depths[index])
        bound = self.binary_loader(self.bound[index])
        image, gt, depth,bound = cv_random_flip(image, gt, depth,bound)
        image, gt, depth,bound = randomCrop(image, gt, depth,bound)
        image, gt, depth,bound = randomRotation(image, gt, depth,bound)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        bound = randomPeper(bound)
        # image, gt, depth = self.resize(image,gt, depth)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        bound = self.bound_transform(bound)
        depth = self.depths_transform(depth)

        return image, gt, depth,bound

    def filter_files(self):
        # print('len(self.images)',len(self.images))
        # print('len(self.depths)', len(self.depths))
        # print('len(self.images)', len(self.images))
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        bounds = []
        for img_path, gt_path, depth_path,bound_path in zip(self.images, self.gts, self.depths,self.bound):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            bound = Image.open(bound_path)
            # if img.size == depth.size:
                    # and gt.size == depth.size
            images.append(img_path)
            gts.append(gt_path)
            depths.append(depth_path)
            bounds.append(bound_path)
        self.images = images
        self.gts = gts
        self.depths = depths
        self.bound = bounds

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)


    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, depth_root, bound_root,batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, bound_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True
                                  )
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')or  f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        # depth = self.binary_loader(self.depths[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        # image, gt, depth = self.resize(image, gt, depth)
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        # assert img.size == gt.size and gt.size == depth.size
        h = self.testsize
        w = self.testsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)


    def __len__(self):
        return self.size
