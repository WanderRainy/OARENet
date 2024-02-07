import setting
import os
os.environ["CUDA_VISIBLE_DEVICES"]=setting.setting['gpu']
from tqdm import tqdm
from time import time
import torch
import torch.utils.data as data
from networks.testNet import SwinT_OAM
from framework import MyFrame
from loss import dice_bce_loss
from data import *

dataset = setting.setting['dataset']
SHAPE = setting.setting['crop_size']
dataroot = setting.setting['dataroot']
model = setting.setting['model']
NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
total_epoch = setting.setting['epoch']

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if model =='SwinT_OAM':
    solver = MyFrame(SwinT_OAM, dice_bce_loss, 2e-4)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
if dataset == 'DPGlobe':
    train_dir = os.path.join(dataroot,'train/')
    dataset = DPGlobe_Dataset(train_dir,type='Train')
if dataset == 'Mass':
    dataset = Mass_Dataset(dataroot,type='Train')
    trainset_indexs = torch.arange(len(dataset))[:int(len(dataset)*0.8)]
    dataset = torch.utils.data.Subset(dataset, trainset_indexs)

if os.path.exists('weights/' + NAME + '.th'):# checkpoint
    solver.load('weights/' + NAME + '.th')

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

mylog = open('logs/'+NAME+'.log','w')# checkpoint
tic = time()
no_optim = 0
train_epoch_best_loss = 100.
for epoch in range(0, total_epoch + 1): # checkpoint
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in tqdm(data_loader_iter):
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss.item(), file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss.item())
    print('SHAPE:', SHAPE)
    

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
        
    if epoch==int(total_epoch/2):
        #solver.load('weights/'+NAME+'.th')
        solver.save('weights/'+NAME+str(epoch)+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
        
    if epoch==int(total_epoch*0.65):
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        #solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    
    if epoch==int(total_epoch*0.8):
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        #solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
        
    mylog.flush()
    
print('Finish!', file=mylog)
print('Finish!')
mylog.close()
