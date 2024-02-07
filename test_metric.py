import os
import setting
os.environ["CUDA_VISIBLE_DEVICES"]= setting.setting['gpu']
import torch.utils.data as data

import time
from tqdm import tqdm
from thop import profile

from networks.unet import Unet
from sklearn.metrics import confusion_matrix
from networks.dunet import Dunet
from networks.testNet import SwinT_OAM
from data import *

NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
out_pred = setting.setting['out_pred']
dataroot = setting.setting['dataroot']
dataset = setting.setting['dataset']
model = setting.setting['model']
shape = setting.setting['crop_size']

target = './results/'+NAME+'_'+dataset+'/'

if dataset == 'DPGlobe':
    test_dir = os.path.join(dataroot,'test_label/')
    dataset = DPGlobe_Dataset(test_dir,type='Test')
if dataset == 'JHWV2':
    dataset = JHWV2_Dataset(dataroot,type='Test')
    # testset_indexs = torch.arange(len(dataset))[int(len(dataset)*0.8):]
    # dataset = torch.utils.data.Subset(dataset, testset_indexs)
if dataset == 'Mass':
    dataset = Mass_Dataset(dataroot,type='Test')
    trainset_indexs = torch.arange(len(dataset))[int(len(dataset)*0.8):]
    dataset = torch.utils.data.Subset(dataset, trainset_indexs)

mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


if not os.path.isdir(target):
    os.makedirs(target)


if model =='SwinT_OAM':
    net = SwinT_OAM()

net = net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.load_state_dict(torch.load('./weights/'+NAME+'.th'))

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4)

tic = time.time()
class SegmentationMetric():
    def __init__(self,numclass):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass,) * 2)

    def CM(self, mask, pred):
        mask_num = mask.numpy().astype('int')
        pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask_num.flatten()
        cm = confusion_matrix(mask,pred)
        self.confusionMatrix += cm
        # return cm

    def PA(self):
        # PA = acc
        acc = np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        return acc

    def CPA(self):
        # CPA = (tp)/tp+fp
        # precision
        cpa = np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0)
        return cpa

    def meanPA(self):
        cpa = self.CPA()
        meanpa = np.nanmean(cpa)
        return meanpa
    def CPR(self):
        # CPR:class recall
        cpr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return cpr

    def IOU(self):
        intersection = np.diag(self.confusionMatrix)  
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iou = intersection/union
        return iou

metric = SegmentationMetric(2)
# cpa1=[]
# cpr1=[]
# f11=[]
# iou1=[]
title = 0
net.eval()
with torch.no_grad():
    for data_loader_iter in tqdm(iter(data_loader)):
        img, mask = data_loader_iter
        img = img.cuda()
        # solver.set_input(img, mask)
        pred = net(img)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        if out_pred==True:
            for i in range(0,pred.shape[0]):
                maskout = mask[i].cpu()
                imgout = img[i].cpu()
                predout = pred[i].cpu()
                cv2.imwrite(target +str(title)+'_mask.png', (maskout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                cv2.imwrite(target + str(title) + '_img.png', (((imgout.numpy()*std)+mean)*255).astype(np.uint8).transpose(1,2,0))
                cv2.imwrite(target + str(title) + '_pred.png', (predout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                title += 1
        pred = pred.squeeze().cpu().data.numpy()
        # mask.shape(3,1,768,768) pred.shape(3,768,768)
        metric.CM(mask, pred)

cpa = metric.CPA()[1]
cpr = metric.CPR()[1]
f1 = 2*(cpa*cpr/(cpa+cpr))
iou = metric.IOU()[1]
# 写入实验结果
metric_file = open('./logs/' + NAME + '_metric.log', 'a')
# 写入实验条件
metric_file.write('--------%s--------\n' % str(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))))
for k,v in setting.setting.items():
    metric_file.write('{}:{}\n'.format(k, v))
metric_file.write('cpa:{}\n'.format(cpa))
metric_file.write('cpr:{}\n'.format(cpr))
metric_file.write('f1:{}\n'.format(f1))
metric_file.write('iou:{}\n'.format(iou))
print('Finish!')
metric_file.close()

