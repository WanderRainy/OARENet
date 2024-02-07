import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import osgeo.gdal as gdal
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from networks.testNet import SwinT_OAM
import cv2
import os
import pickle
import time

def base_predict(file_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    original = gdal.Open(file_path, gdal.GA_ReadOnly)
    img = original.ReadAsArray()[:3]
    row = img.shape[1]
    col = img.shape[2]

    # pad = np.pad(img, ((0, 0), (winsize, int(winsize + ((np.ceil(row / winsize)) * winsize) - row)),
    #                    (winsize, int(winsize + ((np.ceil(col / winsize)) * winsize) - col))), mode='symmetric')

    pad = np.pad(img, ((0, 0), (winsize, int(winsize + ((np.ceil(row / winsize)) * winsize) - row)+buffersize),
                       (winsize, int(winsize + ((np.ceil(col / winsize)) * winsize) - col)+buffersize)), mode='symmetric')

    nrow = pad.shape[1]
    ncol = pad.shape[2]
    result = torch.zeros([nrow, ncol]).cuda()
    print('image size:',pad.shape)
    with torch.no_grad():
        for r in range(int(row / winsize) + 1):
            print(r.__str__() + "/" + str(int(row / winsize) + 1))
            for c in range(int(col / winsize) + 1):
                tem = pad[:, int((r + 1) * winsize) - buffersize:int((r + 2) * winsize) + buffersize,
                      int((c + 1) * winsize) - buffersize:int((c + 2) * winsize) + buffersize]
                if (tem.max() == 0):
                    result[int((r + 1) * winsize):int((r + 2) * winsize),
                    int((c + 1) * winsize):int((c + 2) * winsize)] = 0

                input = (np.array(tem, np.float32) / 255.0 - mean)/std
                # input = (np.array(tem, np.float32) / 255.0 )*3.2-1.6
                tem = net(torch.Tensor([input]).cuda()).squeeze()
                tem[tem > 0.1] = 255
                tem[tem <= 0.1] = 0
                # tem = softmax(tem)[0].argmax(dim=0)
                result[int((r + 1) * winsize):int((r + 2) * winsize),
                int((c + 1) * winsize):int((c + 2) * winsize)] = tem[
                                                                 buffersize:buffersize+winsize,
                                                                 buffersize:buffersize+winsize]


    result = result[winsize:winsize + row, winsize:winsize + col]
    result = result.cpu().detach().numpy()
    # save the result
    print('Create image')
    cv2.imwrite(os.path.join(save_path,imgname+'_'+modelname+'.jpg'),result)
    return result/255

class SegmentationMetric():
    def __init__(self,numclass):
        self.numclass = numclass

    def CM(self, mask, pred):
        # mask_num = mask.numpy().astype('int')
        # pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask.flatten()
        cm = confusion_matrix(mask,pred)
        self.confusionMatrix = cm
        return cm

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
if __name__ == '__main__':
    winsize = 1024 #256
    buffersize = 64
    img_path = "/home/yry22/data/LSRV/Shanghai_img.jpg"  # Birmingham #Shanghai # Boston
    label_path = "/home/yry22/data/LSRV/Shanghai_road.png"
    img_path = "/home/yry22/data/jianghan_WV2/large/20130613.tif" # 20130613,20170423,20200225
    label_path = "/home/yry22/data/jianghan_WV2/large/20130613_label.tif"
    wuhan = False
    save_path = "./results/large/"
    os.makedirs(save_path,exist_ok=True)
    weight_file = "weights/SwinT_OAM_DP.th"
    modelname = os.path.basename(weight_file).split('.')[0]
    imgname = os.path.basename(img_path).split('.')[0]

    net = SwinT_OAM()

    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load(weight_file))
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    net.eval()
    print('Model Load Done')

    pred = base_predict(img_path, save_path)
    pred = np.array(pred, dtype=np.uint8)

    if 'jianghan' in imgname:
        import skimage.io as io
        label = io.imread(label_path)
    else:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label.max()==255:
        label = label/255
    # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = np.array(label,dtype=np.uint8)
    metric = SegmentationMetric(2)
    cm = metric.CM(label, pred)
    cpa = metric.CPA()[1]
    cpr = metric.CPR()[1]
    f1 = 2 * (cpa * cpr / (cpa + cpr))
    iou = metric.IOU()[1]
    # 写入实验结果
    metric_file = open(os.path.join(save_path,imgname+'_metric.log'), 'a')
    # 写入实验条件
    metric_file.write('--------%s--------\n' % str(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))))
    metric_file.write(weight_file+'\n')
    metric_file.write('cpa:{}\n'.format(cpa))
    metric_file.write('cpr:{}\n'.format(cpr))
    metric_file.write('f1:{}\n'.format(f1))
    metric_file.write('iou:{}\n'.format(iou))
    metric_file.write('win_size:{},buffersize:{}\n'.format(winsize,buffersize))
    print('Finish!')
    metric_file.close()




