setting ={
    'exper_name':'SwinT_OA_DP',# 模型-训练集
    'crop_size':(1280,1280),#(1024,1024)#1280，1024 #1472
    'dataset':'JHWV2', # 'DPGlobe' or 'JHWV2' or'Mass'
    'dataroot':'/data1/yry22/data/jianghan_WV2/', # '/home/yry22/data/dg_road/'or'/home/yry22/data/jianghan_WV2/'or'/home/yry22/data/Massachusetts/'
    'model':'SwinT_OAM', 
    'batch_size_card': 4,
    'gpu':'1',
    'seed':197,
    'epoch':150,
    'out_pred':True,
    'remarks':'',
}

