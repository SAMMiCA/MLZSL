import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import src.model as model
import src.util_openimages as util
from config.config_openimages import opt
from src.util_openimages import LoadMetaData, SaveResultFigure, FilterSeenResults, VisClassSeenTopK
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
from PIL import Image, ImageDraw
import pdb
from src.dataset_custom import DatasetCustom




if __name__ == '__main__':
    
    raw_img_dir = '/root/datasets/ETRI/ADE'
    custom_dataset = DatasetCustom(raw_img_dir)
    num_k = 5
    fig_dir = './figures/220420_etri/nus_top_k'
    os.makedirs(fig_dir, exist_ok=True)

    ### load metadata ###
    #### Load Meta Data ####
    seen_labelmap_path = os.path.join(opt.src, '2022_01', 'classes-trainable.txt')
    unseen_labelmap_path = os.path.join(opt.src, '2022_01', 'unseen_labels.pkl')
    dict_path = os.path.join(opt.src, '2022_01', 'class-descriptions.csv')
    seen_labelmap, unseen_labelmap, label_dict = LoadMetaData(seen_labelmap_path, unseen_labelmap_path, dict_path)

    df_top_unseen = pd.read_csv(os.path.join(opt.src, '2022_01', 'top_400_unseen.csv'), header=None)
    idx_top_unseen = df_top_unseen.values[:, 0]
    
    pdb.set_trace()

    
    ### load models ###
    model_fe = model.VGG_Feature_Extractor()
    model_vgg = model.VGG_Global_Feature_Extractor()
    model_joint_emb = model.BiAM(opt, vecs=vecs_seen)

    model_vgg = model.VGG_Global_Feature_Extractor()
    model_test = model.BiAM(opt, dim_feature=[196,512])
    # print(model_test)

    name='OPENIMAGES_{}'.format(opt.SESSION)
    opt.save_path += '/'+name


    # # vecs_925 = torch.from_numpy(normalize(src_att[0][seen_cls_idx]))
    # vecs_seen = torch.from_numpy(normalize(src_att[0][seen_cls_idx]))
    # # vecs_81 = torch.from_numpy(normalize(src_att[1]))
    # vecs_unseen = torch.from_numpy(normalize(src_att[1]))
    # scales = opt.scales
    
    # ### load models ###
    # model_fe = model.VGG_Feature_Extractor()
    # model_vgg = model.VGG_Global_Feature_Extractor()
    # model_joint_emb = model.MSBiAM(opt, vecs=vecs_seen)
    
    # if torch.cuda.is_available():
    #     device_ids = [i for i in range(torch.cuda.device_count())]
    #     device_main = torch.device(f"cuda:{device_ids[0]}")
    #     if len(device_ids) > 1:
    #         model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
    #         model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
    #         model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
    #     else:
    #         # model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
    #         # model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
    #         # model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
    #         model_joint_emb = model_joint_emb.to(device_main)
    #         model_vgg = model_vgg.to(device_main)
    #         model_fe = model_fe.to(device_main)
    
    # model_path = opt.save_path
    # model_joint_emb.load_state_dict(torch.load(model_path))
    
    # model_joint_emb.eval()
    # model_vgg.eval()
    # model_fe.eval()

    # dataloader = DataLoader(dataset=custom_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
    
    # num_test = len(custom_dataset)
    
    
    # for i, batch_data in enumerate(tqdm(dataloader)):
    #     batch_imgs = batch_data[0].cuda()
    #     batch_img_paths = batch_data[1]
    #     # pdb.set_trace()
    #     with torch.no_grad():
    #         batch_features = model_fe(batch_imgs)

    #         vgg_4096 = model_vgg(batch_features) if model_vgg is not None else None
            
    #         # model_joint_emb.module.vecs = vecs_925
    #         model_joint_emb.vecs = vecs_seen
    #         logits_925 = model_joint_emb(batch_features, vgg_4096)
            
    #         # model_joint_emb.module.vecs = vecs_81
    #         model_joint_emb.vecs = vecs_unseen
    #         logits_81 = model_joint_emb(batch_features, vgg_4096)

    #     _, topk_idx_unseen = torch.topk(logits_81, 5)
    #     _, topk_idx_seen = torch.topk(logits_925, 5)

    #     for fig_i, img_path in enumerate(batch_img_paths):
    #         seen_fig_dir = os.path.join(fig_dir, 'seen')
    #         unseen_fig_dir = os.path.join(fig_dir, 'unseen')
    #         os.makedirs(seen_fig_dir, exist_ok=True)
    #         os.makedirs(unseen_fig_dir, exist_ok=True)

    #         seen_fig_path = os.path.join(seen_fig_dir, os.path.basename(img_path))
    #         unseen_fig_path = os.path.join(unseen_fig_dir, os.path.basename(img_path))

    #         fig_path = os.path.join(fig_dir, os.path.basename(img_path))
            
            
    #         fig_labels_seen = tag925[topk_idx_seen.cpu()[fig_i]]
    #         fig_labels_unseen = tag81[topk_idx_unseen.cpu()[fig_i]]
            
    #         vis_img_labels(img_path, seen_fig_path, fig_labels_seen)
    #         vis_img_labels(img_path, unseen_fig_path, fig_labels_unseen)
    #         # pdb.set_trace()