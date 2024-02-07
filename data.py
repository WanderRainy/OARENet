"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
import random
import setting
random.seed(setting.setting['seed'])

def random_crop(image, crop_shape, mask=None):
    image_shape = image.shape
    image_shape = image_shape[0:2]
    ret = []
    if crop_shape[0]<image_shape[0]: # 当测试设定不裁剪时，略去裁剪步骤
        nh = np.random.randint(0, image_shape[0] - crop_shape[0])
        nw = np.random.randint(0, image_shape[1] - crop_shape[1])
        image = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        # while image.max()==0:
        #     # 训练集由大幅影像裁剪，部分图像中存在空白区域，剔除全白裁剪影像
        #     nh = np.random.randint(0, image_shape[0] - crop_shape[0])
        #     nw = np.random.randint(0, image_shape[1] - crop_shape[1])
        #     image = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    ret.append(image)
    if mask is not None:
        if crop_shape[0] < image_shape[0]:
            mask = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        ret.append(mask)
        return ret
    return ret[0]

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, root ,type=None):
    img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(root+'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    if type=='Test':
        np.random.seed(setting.setting['seed'])
    img, mask = random_crop(img, setting.setting['crop_size'], mask)
    if type=='Train':
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    #mask = abs(mask-1)
    return img, mask

class DPGlobe_Dataset(data.Dataset):

    def __init__(self, root, type=None):
        '''

        :param trainlist:
        :param root:
        '''
        imagelist = filter(lambda x: x.find('jpg') != -1, os.listdir(root))
        imglist = list(map(lambda x: x[:-8], imagelist))
        self.ids = imglist
        self.loader = default_loader
        self.root = root
        self.type = type
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root,type=self.type)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 1.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(list(self.ids))


class JHWV2_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/img')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.labellist = list(map(lambda x: x.replace('img', 'label'), self.imglist))
        self.type=type
        # self.mean = np.array([0.5181, 0.4532, 0.3734]).reshape(3, 1, 1)
        # self.std = np.array([0.0719, 0.0806, 0.1009]).reshape(3, 1, 1)
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index]))
        label = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
        label = label*255
        if type == 'Test':
            np.random.seed(setting.setting['seed'])
        img, mask = random_crop(img, self.shape, label)
        if type=='Train':
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        mask = np.expand_dims(mask, axis=2)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 1.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.imglist)


class Mass_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/image')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.labellist = list(map(lambda x: x[:-1], self.imglist))
        self.type=type
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/image', self.imglist[index]))
        label = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]))[:,:,0]
        if type == 'Test': #测试时，每次裁剪相同区域
            np.random.seed(setting.setting['seed'])
        img, mask = random_crop(img, self.shape, label)
        if type=='Train':
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        mask = np.expand_dims(mask, axis=2)
        # img = np.array(img, np.float32).transpose(2, 0, 1) / img.max() * 3.2 - 1.6
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0-self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 1.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.imglist)