import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from config.config_nus import opt
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from src.dataset_nus import get_seen_unseen_classes, DatasetNUS
import pickle
from sklearn.preprocessing import normalize
import src.model as model
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.util_nus import compute_AP, compute_F1
from src.vis import vis_img_labels
import pdb
from src.dataset_custom import DatasetCustom


if __name__ == '__main__':
    print(opt)
    # raw_img_dir = '/root/datasets/ETRI/ADE'
    names = ['cafe', 'dining_room', 'face', 'food', 'fruit', 'kitchen', 'market', 'office', 'restaurant', 'shopping_mall']
    for name in names:
        print(f'{name}')
        raw_img_dir = f'/root/datasets/NUS_WIDE/Flickr/{name}'

        custom_dataset = DatasetCustom(raw_img_dir)
        num_k = 5
        # fig_dir = './figures/220420_etri/nus_pre_top_k'
        fig_dir = f'./figures/220420_etri/nus_pre_top_k_{name}'
        
        os.makedirs(fig_dir, exist_ok=True)

        ### load metadata ###
        src = opt.src
        att_path = os.path.join(src,'wiki_contexts','NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
        file_tag1k = os.path.join(src,'NUS_WID_Tags','TagList1k.txt')
        file_tag81 = os.path.join(src,'ConceptsList','Concepts81.txt')
        seen_cls_idx, unseen_cls_idx, tag1k, tag81 = get_seen_unseen_classes(file_tag1k, file_tag81)
        tag925 = tag1k[seen_cls_idx]
        src_att = pickle.load(open(att_path, 'rb'))
        
        # vecs_925 = torch.from_numpy(normalize(src_att[0][seen_cls_idx]))
        vecs_seen = torch.from_numpy(normalize(src_att[0][seen_cls_idx]))
        # vecs_81 = torch.from_numpy(normalize(src_att[1]))
        vecs_unseen = torch.from_numpy(normalize(src_att[1]))
        scales = opt.scales
        
        ### load models ###
        model_fe = model.VGG_Feature_Extractor()
        model_vgg = model.VGG_Global_Feature_Extractor()
        # model_joint_emb = model.MSBiAM(opt, vecs=vecs_seen)
        model_joint_emb = model.BiAM(opt, dim_feature=[196,512])

        if torch.cuda.is_available():
            device_ids = [i for i in range(torch.cuda.device_count())]
            device_main = torch.device(f"cuda:{device_ids[0]}")
            if len(device_ids) > 1:
                model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
                model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
                model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
            else:
                # model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
                # model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
                # model_fe = nn.DataParallel(model_fe, device_ids=device_ids).cuda()
                model_joint_emb = model_joint_emb.to(device_main)
                model_vgg = model_vgg.to(device_main)
                model_fe = model_fe.to(device_main)
        
        model_path = opt.save_path
        model_joint_emb.load_state_dict(torch.load(model_path))
        
        model_joint_emb.eval()
        model_vgg.eval()
        model_fe.eval()

        dataloader = DataLoader(dataset=custom_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
        
        num_test = len(custom_dataset)
        
        
        for i, batch_data in enumerate(tqdm(dataloader)):
            batch_imgs = batch_data[0].cuda()
            batch_img_paths = batch_data[1]
            # pdb.set_trace()
            with torch.no_grad():
                batch_features = model_fe(batch_imgs)

                vgg_4096 = model_vgg(batch_features) if model_vgg is not None else None
                
                # model_joint_emb.module.vecs = vecs_925
                model_joint_emb.vecs = vecs_seen
                logits_925 = model_joint_emb(batch_features, vecs_seen.cuda(), vgg_4096)
                
                # model_joint_emb.module.vecs = vecs_81
                model_joint_emb.vecs = vecs_unseen
                logits_81 = model_joint_emb(batch_features, vecs_unseen.cuda(), vgg_4096)

            _, topk_idx_unseen = torch.topk(logits_81, 5)
            _, topk_idx_seen = torch.topk(logits_925, 5)

            for fig_i, img_path in enumerate(batch_img_paths):
                seen_fig_dir = os.path.join(fig_dir, 'seen')
                unseen_fig_dir = os.path.join(fig_dir, 'unseen')
                os.makedirs(seen_fig_dir, exist_ok=True)
                os.makedirs(unseen_fig_dir, exist_ok=True)

                seen_fig_path = os.path.join(seen_fig_dir, os.path.basename(img_path))
                unseen_fig_path = os.path.join(unseen_fig_dir, os.path.basename(img_path))

                fig_path = os.path.join(fig_dir, os.path.basename(img_path))
                
                
                fig_labels_seen = tag925[topk_idx_seen.cpu()[fig_i]]
                fig_labels_unseen = tag81[topk_idx_unseen.cpu()[fig_i]]
                
                vis_img_labels(img_path, seen_fig_path, fig_labels_seen)
                vis_img_labels(img_path, unseen_fig_path, fig_labels_unseen)
                # pdb.set_trace()