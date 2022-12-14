from cProfile import label
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import src.model as model
import src.util_openimages as util
from config.config_openimages import opt
from src.util_openimages import LoadMetaData
import numpy as np
import random
import time
import os
import socket
from torch.utils.data import DataLoader
import h5py
import pickle
import logging
import csv
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pdb
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import cv2

print(opt)

#### Load Meta Data ####
seen_labelmap_path = os.path.join(opt.src, '2022_01', 'classes-trainable.txt')
unseen_labelmap_path = os.path.join(opt.src, '2022_01', 'unseen_labels.pkl')
dict_path = os.path.join(opt.src, '2022_01', 'class-descriptions.csv')
seen_labelmap, unseen_labelmap, label_dict = LoadMetaData(seen_labelmap_path, unseen_labelmap_path, dict_path)

df_top_unseen = pd.read_csv(os.path.join(opt.src, '2022_01', 'top_400_unseen.csv'), header=None)
idx_top_unseen = df_top_unseen.values[:, 0]
########################
figure_dir = opt.figure_dir
os.makedirs(figure_dir, exist_ok=True)
num_k = 5
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf", 30)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

data = util.DATA_LOADER(opt) ### INTIAL DATALOADER ###

model_vgg = None
model_vgg = model.VGG_Global_Feature_Extractor()
model_test = model.BiAM(opt, dim_feature=[196,512])
print(model_test)

if torch.cuda.is_available():
    device_ids = [i for i in range(torch.cuda.device_count())]
    device_main = torch.device(f"cuda:{device_ids[0]}")
    if len(device_ids) > 1:
        model_joint_emb = nn.DataParallel(model_test, device_ids=device_ids).cuda()
        model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
        # model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
    else:
        # model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
        # model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
        # model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
        model_joint_emb = model_test.to(device_main)
        model_vgg = model_vgg.to(device_main)
        # model_fe = model_fe.to(device_main)
        data.vecs_400 = data.vecs_400.to(device_main)
        data.vecs_7186 = data.vecs_7186.to(device_main)
    



test_start_time = time.time()


model_path = opt.save_path
pdb.set_trace()
print(model_path)
model_test.load_state_dict(torch.load(model_path))
model_test.eval()

src = opt.src
test_loc = os.path.join(src, 'test_features', 'OPENIMAGES_TEST_CONV5_4_NO_CENTERCROP.h5')
test_features = h5py.File(test_loc, 'r')
test_feature_keys = list(test_features.keys())
image_names = np.unique(np.array([m.split('-')[0] for m in test_feature_keys]))
ntest = len(image_names)
test_batch_size = opt.test_batch_size

assert len(idx_top_unseen) == 400

print('===> total TEST samples')
print(ntest)
logging.info('===> total TEST samples')
logging.info(ntest)

prediction_400 = torch.empty(ntest,400)
prediction_7586 = torch.empty(ntest,7586)
prediction_7186 = torch.empty(ntest,7186)
lab_400 = torch.empty(ntest,400)
lab_7586 = torch.empty(ntest,7586)
lab_7186 = torch.empty(ntest,7186)

if model_vgg is not None:
    model_vgg.eval()

for m in range(0, ntest, test_batch_size):
    strt = m
    endt = min(m+test_batch_size, ntest)
    bs = endt-strt
    c=m
    c+=bs
    features, labels_7186, labels_2594 = np.empty((bs,512,196)), np.empty((bs,7186)), np.empty((bs,2594))
    for i, key in enumerate(image_names[strt:endt]):
        features[i,:,:] = np.float32(test_features.get(key+'-features'))
        labels_7186[i,:] =  np.int32(test_features.get(key+'-seenlabels'))
        labels_2594[i,:] =  np.int32(test_features.get(key+'-unseenlabels'))
    
    features = torch.from_numpy(features).float()
    labels_7186 = torch.from_numpy(labels_7186).long()
    labels_400 = torch.from_numpy(labels_2594).long()[:,idx_top_unseen]
    labels_7586 = torch.cat((labels_7186,labels_400),1)
    
    with torch.no_grad():
        vgg_4096 = model_vgg(features.cuda()) if model_vgg is not None else None
        logits_400 = model_test(features.cuda(), data.vecs_400, vgg_4096)
        logits_7586  = model_test(features.cuda(), gzsl_vecs, vgg_4096) ##seen-unseen
        logits_7186 = model_test(features.cuda(), data.vecs_7186, vgg_4096) ##seen-unseen
    
    att_maps = model_test.get_att_map(features.cuda(), data.vecs_400, vgg_4096)
    # att_map = att_map.reshape([bs, 14, 14, 400])

    ###################### JW Visualization of image and prediction ###############################################################
    jw_image_names = image_names[strt:endt]
    jw_image_paths = [os.path.join("/root/jwssd/datasets/OpenImages/test", file_name) for file_name in jw_image_names]
    if m > 320:
        pdb.set_trace()
    os.makedirs(os.path.join(figure_dir, f"batch_{m}"), exist_ok=True)
    for i in range(bs):
        fig_label = torch.clamp(labels_400[i],0,1)
        att_map = att_maps[i]
        if not fig_label.nonzero().shape[0] == 0:    
            for nonzero_i in range(fig_label.nonzero().shape[0]):
                fig_img = Image.open(jw_image_paths[i])
                unseen_id = fig_label.nonzero()[nonzero_i].item()
                unseen_class = unseen_labelmap[idx_top_unseen[unseen_id]]
                # pdb.set_trace()
                class_name = label_dict[unseen_class]
                draw = ImageDraw.Draw(fig_img)
                print(f"{m}_{i}")
                # print(f"{m}_{i}: {unseen_id}/{class_name}")
                # if m==320 and i==21:
                #     pdb.set_trace()
                if fig_img.mode == "L":
                    draw.text((0, 0), f"{class_name}", fill=255, font=font)    
                else:
                    draw.text((0, 0), f"{class_name}", fill=(255, 255, 255), font=font)
                # draw.text((0, 0), f"{class_name}", fill=255, font=font)
                
                fig_att_map = att_map[unseen_id]
                fig_att_map -= fig_att_map.min()
                fig_att_map /= fig_att_map.max()
                fig_att_map = to_pil_image(fig_att_map)
                fig_att_map = fig_att_map.resize(fig_img.size)
                
                numpy_att_map = np.array(fig_att_map)
                opencv_att_map = cv2.cvtColor(numpy_att_map, cv2.COLOR_RGB2BGR)
                heatmap = cv2.applyColorMap(opencv_att_map, cv2.COLORMAP_JET)
                
                numpy_img = np.array(fig_img)
                opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                
                img_with_heatmap = cv2.addWeighted(heatmap, 0.7, opencv_img, 0.3, 0)
                
                cv2.imwrite(os.path.join(figure_dir, f"batch_{m}", f"label_{i}_nonzero{nonzero_i}.png"), img_with_heatmap)
            
            
            fig_pred = logits_400[i]
            _, topk_idx = torch.topk(fig_pred, num_k)
            
            for topk_i in range(num_k):
                fig_img = Image.open(jw_image_paths[i])
                unseen_id = topk_idx[topk_i]
                unseen_class = unseen_labelmap[idx_top_unseen[unseen_id]]
                class_name = label_dict[unseen_class]
                draw = ImageDraw.Draw(fig_img)
                if fig_img.mode == "L":
                    draw.text((0, 0), f"{class_name}", fill=255, font=font)    
                else:
                    draw.text((0, 0), f"{class_name}", fill=(255, 255, 255), font=font)
                
                fig_att_map = att_map[unseen_id]
                fig_att_map -= fig_att_map.min()
                fig_att_map /= fig_att_map.max()
                fig_att_map = to_pil_image(fig_att_map)
                fig_att_map = fig_att_map.resize(fig_img.size)
                
                numpy_att_map = np.array(fig_att_map)
                opencv_att_map = cv2.cvtColor(numpy_att_map, cv2.COLOR_RGB2BGR)
                heatmap = cv2.applyColorMap(opencv_att_map, cv2.COLORMAP_JET)
                
                numpy_img = np.array(fig_img)
                opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                
                img_with_heatmap = cv2.addWeighted(heatmap, 0.3, opencv_img, 0.7, 0)
                
                cv2.imwrite(os.path.join(figure_dir, f"batch_{m}", f"pred_{i}_topk{topk_i}.png"), img_with_heatmap)
    
                