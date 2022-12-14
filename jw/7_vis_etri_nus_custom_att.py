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
from torchvision.transforms.functional import resize
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid


font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf", 30)

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
        fig_dir = f'./figures/220421_etri/nus_pre_top_k_{name}'
        
        os.makedirs(fig_dir, exist_ok=True)

        ### load metadata ###
        src = opt.src
        att_path = os.path.join(src,'wiki_contexts','NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
        file_tag1k = os.path.join(src,'NUS_WID_Tags','TagList1k.txt')
        file_tag81 = os.path.join(src,'ConceptsList','Concepts81.txt')
        seen_cls_idx, unseen_cls_idx, tag1k, tag81 = get_seen_unseen_classes(file_tag1k, file_tag81)
        tag925 = tag1k[seen_cls_idx]
        src_att = pickle.load(open(att_path, 'rb'))
        pdb.set_trace()
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
        # pdb.set_trace()
        
        for i, batch_data in enumerate(tqdm(dataloader)):
            batch_imgs = batch_data[0].cuda()
            batch_img_paths = batch_data[1]
            if i == 5:
                pdb.set_trace()
            with torch.no_grad():
                batch_features = model_fe(batch_imgs)
                batch_features_x2 = model_fe(resize(batch_imgs, (448,448)))
                
                vgg_4096 = model_vgg(batch_features) if model_vgg is not None else None
                
                # model_joint_emb.module.vecs = vecs_925
                model_joint_emb.vecs = vecs_seen
                logits_925 = model_joint_emb(batch_features, vecs_seen.cuda(), vgg_4096)
                att_925 = {}
                att_925[1] = model_joint_emb.get_att_map(batch_features, vecs_seen.cuda(), vgg_4096)
                att_925[2] = model_joint_emb.get_att_map(batch_features_x2, vecs_seen.cuda(), vgg_4096)
                
                # model_joint_emb.module.vecs = vecs_81
                model_joint_emb.vecs = vecs_unseen
                logits_81 = model_joint_emb(batch_features, vecs_unseen.cuda(), vgg_4096)
                att_81 = {}
                att_81[1] = model_joint_emb.get_att_map(batch_features, vecs_unseen.cuda(), vgg_4096)
                att_81[2] = model_joint_emb.get_att_map(batch_features_x2, vecs_unseen.cuda(), vgg_4096)
            # pdb.set_trace()
            _, topk_idx_unseen = torch.topk(logits_81, 5)
            _, topk_idx_seen = torch.topk(logits_925, 5)

            for fig_i, img_path in enumerate(batch_img_paths):
                seen_fig_dir = os.path.join(fig_dir, 'seen')
                os.makedirs(seen_fig_dir, exist_ok=True)
                
                fig_labels_seen = tag925[topk_idx_seen.cpu()[fig_i]]
                for topk_i in range(num_k):
                    img_name = os.path.basename(img_path).split('.')[0]
                    fig_path = os.path.join(seen_fig_dir, f'{img_name}_top{topk_i}_{fig_labels_seen[topk_i]}.png')
                    fig_img_list = []
                    for scale in [1,2]:
                        fig_img = Image.open(img_path)
                        class_name = fig_labels_seen[topk_i]
                        draw = ImageDraw.Draw(fig_img)
                        if fig_img.mode == "L":
                            draw.text((0, 0), f"{class_name}", fill=255, font=font)
                        else:
                            draw.text((0, 0), f"{class_name}", fill=(255, 255, 255), font=font)
                        # pdb.set_trace()
                        seen_id = topk_idx_seen.cpu()[fig_i][topk_i]
                        fig_att_map = att_925[scale][fig_i][seen_id]
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
                        
                        img_with_heatmap = cv2.cvtColor(img_with_heatmap, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_with_heatmap)
                        pil_img = to_tensor(pil_img)
                    
                        fig_img_list.append(pil_img)
                    figures = torch.stack(fig_img_list)
                    grid = make_grid(figures)
                    to_pil_image(grid).save(fig_path)
                

                unseen_fig_dir = os.path.join(fig_dir, 'unseen')
                os.makedirs(unseen_fig_dir, exist_ok=True)

                fig_labels_unseen = tag81[topk_idx_unseen.cpu()[fig_i]]
                for topk_i in range(num_k):
                    img_name = os.path.basename(img_path).split('.')[0]
                    fig_path = os.path.join(unseen_fig_dir, f'{img_name}_top{topk_i}_{fig_labels_unseen[topk_i]}.png')
                    fig_img_list = []
                    for scale in [1,2]:
                        fig_img = Image.open(img_path)
                        class_name = fig_labels_unseen[topk_i]
                        draw = ImageDraw.Draw(fig_img)
                        if fig_img.mode == "L":
                            draw.text((0, 0), f"{class_name}", fill=255, font=font)
                        else:
                            draw.text((0, 0), f"{class_name}", fill=(255, 255, 255), font=font)
                        # pdb.set_trace()
                        unseen_id = topk_idx_unseen.cpu()[fig_i][topk_i]
                        fig_att_map = att_81[scale][fig_i][unseen_id]
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
                        
                        img_with_heatmap = cv2.cvtColor(img_with_heatmap, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_with_heatmap)
                        pil_img = to_tensor(pil_img)
                    
                        fig_img_list.append(pil_img)
                    figures = torch.stack(fig_img_list)
                    grid = make_grid(figures)
                    to_pil_image(grid).save(fig_path)
                
                

                