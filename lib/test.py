import torch
import pickle

# pthfile = r'/data/lijiaxin/myWSOD/Outputs/vgg16_voc2007_ours/Feb27-21-19-13_amax_step/ckpt/model_step1.pth'  # .pth文件的路径
# model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
# # print('type:')
# print(type(model))  # 查看模型字典长度
# print('length:')
# print(len(model))
# print('key:')
# for k in model.keys():  # 查看模型字典里面的key
#     print(k)
# print('value:')
# for k in model:  # 查看模型字典里面的value
#     print(k, model[k])
with open("/data/lijiaxin/myWSOD/data/cache/voc_2007_trainval_me_roidb.pkl","rb") as f:
    data = pickle.load(f)
    print(data['labels'])
