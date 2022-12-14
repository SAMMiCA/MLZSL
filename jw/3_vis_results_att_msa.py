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
from src.dataset import OpenImagesMSDatasetTest
from tqdm import tqdm
from sklearn.preprocessing import normalize
from src.vis import vis_ms_att

if __name__ == '__main__':
    print(opt)
    # model_path = os.path.join(opt.save_path, "model_latest_1300_10.pth")
    figure_dir = opt.figure_dir
    num_k = 5
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf", 30)
    model_path = opt.save_path
    ### Load Meta Data ###
    seen_labelmap_path = os.path.join(opt.src, '2022_01', 'classes-trainable.txt')
    unseen_labelmap_path = os.path.join(opt.src, '2022_01', 'unseen_labels.pkl')
    dict_path = os.path.join(opt.src, '2022_01', 'class-descriptions.csv')
    att_path = os.path.join(opt.src, 'wiki_contexts', 'OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl')
    test_h5_path = os.path.join(opt.src, 'ms_features', 'test', 'OPENIMAGES_test.h5')

    seen_labelmap, unseen_labelmap, label_dict = LoadMetaData(seen_labelmap_path, unseen_labelmap_path, dict_path)
    df_top_unseen = pd.read_csv(os.path.join(opt.src, '2022_01', 'top_400_unseen.csv'), header=None)
    idx_top_unseen = df_top_unseen.values[:, 0]
    assert len(idx_top_unseen) == 400
    src_att = pickle.load(open(att_path, 'rb'))
    vecs_7186 = torch.from_numpy(normalize(src_att[0]))
    vecs_400 = torch.from_numpy(normalize(src_att[1][idx_top_unseen,:]))
    if opt.cuda:
        vecs_7186 = vecs_7186.cuda()
        vecs_400 = vecs_400.cuda()
    gzsl_vecs = torch.cat([vecs_7186, vecs_400],0)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ### Load Model ###
    model_vgg = model.VGG_Global_Feature_Extractor()
    model_test = model.MSAV1(opt, dim_feature=[196,512], vecs=vecs_7186)
    # pdb.set_trace()
    model_test.load_state_dict(torch.load(model_path))
    
    if opt.cuda:
        model_vgg = model_vgg.cuda()
        model_test = model_test.cuda()
    # if opt.cuda:
    #     # device_ids = [i for i in range(torch.cuda.device_count())]
    #     device_ids = [0, 1]
    #     model_vgg = model_vgg.cuda()
    #     if len(device_ids) > 1:
    #         model_test = nn.DataParallel(model_test, device_ids=device_ids).cuda()
    #     else:
    #         model_test = model_test.cuda()
    model_vgg.eval()
    model_test.eval()
    print(model_test)

    ### Load Dataset ###
    test_dataset = OpenImagesMSDatasetTest(test_h5_path, opt.scales)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
    
    ntest = len(test_dataset)
    print('===> total TEST samples')
    print(ntest)
    
    prediction_400 = torch.empty(ntest, 400)
    prediction_7586 = torch.empty(ntest, 7586)
    prediction_7186 = torch.empty(ntest, 7186)
    lab_400 = torch.empty(ntest, 400)
    lab_7586 = torch.empty(ntest, 7586)
    lab_7186 = torch.empty(ntest, 7186)
    image_names = np.empty(ntest, dtype='<U20')
    
    ### Compute Test Results ###
    for i, batch_data in enumerate(tqdm(test_loader)):
        strt = i*opt.test_batch_size
        endt = min(strt + opt.test_batch_size, ntest)
        
        ms_test_inputs = {scale: batch_data[scale] for scale in opt.scales}
                
        batch_image_names = np.array(batch_data['image_names'])
        # batch_features = ms_test_inputs[1]
        batch_lab_7186 = batch_data['seenlabels'].long()
        batch_lab_400 = batch_data['unseenlabels'].long()[:, idx_top_unseen]
        batch_lab_7586 = torch.cat((batch_lab_7186, batch_lab_400), 1)
        if opt.cuda:
            # batch_features = batch_features.cuda()
            batch_lab_7186 = batch_lab_7186.cuda()
            batch_lab_400 = batch_lab_400.cuda()
            batch_lab_7586 = batch_lab_7586.cuda()
            for scale in opt.scales:
                ms_test_inputs[scale] = ms_test_inputs[scale].cuda()
        # pdb.set_trace()
        ### Network Output ###
        with torch.no_grad():
            vgg_4096 = model_vgg(ms_test_inputs[1]) if model_vgg is not None else None
            # model_test.vecs = vecs_7186
            # logits_7186 = model_test(ms_test_inputs, vgg_4096)
            logits_7186 = model_test.eval_with_vecs(ms_test_inputs, vgg_4096, vecs_7186)
            
            # model_test.vecs = vecs_400
            # logits_400 = model_test(ms_test_inputs, vgg_4096)
            logits_400 = model_test.eval_with_vecs(ms_test_inputs, vgg_4096, vecs_400)
            
            # model_test.vecs = gzsl_vecs
            # logits_7586  = model_test(ms_test_inputs, vgg_4096) ##seen-unseen
            logits_7586 = model_test.eval_with_vecs(ms_test_inputs, vgg_4096, gzsl_vecs)

   
        att_maps = model_test.get_att_map(ms_test_inputs, vgg_4096, vecs_400)
        # att_map = att_map.reshape([bs, 14, 14, 400])

        ###################### JW Visualization of image and prediction ###############################################################
        jw_image_names = batch_image_names
        jw_image_paths = [os.path.join("/root/jwssd/datasets/OpenImages/test", file_name) for file_name in jw_image_names]
        
        if strt>320:
            pdb.set_trace()
        # pdb.set_trace()
        
        
        bs = att_maps[1].shape[0]
        
        
        
        for i in range(bs):
            fig_label = torch.clamp(batch_lab_400[i],0,1)
            if not fig_label.nonzero().shape[0] == 0:    
                for nonzero_i in range(fig_label.nonzero().shape[0]):
                    for scale in opt.scales:
                        os.makedirs(os.path.join(figure_dir, f"{scale}", f"batch_{strt}"), exist_ok=True)
                        att_map = att_maps[scale][i]
                
                        # pdb.set_trace()
                        fig_img = Image.open(jw_image_paths[i])
                        unseen_id = fig_label.nonzero()[nonzero_i].item()
                        unseen_class = unseen_labelmap[idx_top_unseen[unseen_id]]
                        class_name = label_dict[unseen_class]
                        
                        # pdb.set_trace()
                        # a = {s: att_maps[s][i][unseen_id] for s in opt.scales}
                        # vis_ms_att(jw_image_paths[i], './test.png', opt.scales, a, class_name)
                        
                        draw = ImageDraw.Draw(fig_img)
                        print(f"{strt}_{i}")
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
                        
                        img_with_heatmap = cv2.addWeighted(heatmap, 0.3, opencv_img, 0.7, 0)
                        
                        cv2.imwrite(os.path.join(figure_dir, f"{scale}", f"batch_{strt}", f"label_{i}_nonzero{nonzero_i}.png"), img_with_heatmap)
                    
                    
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
                            
                            cv2.imwrite(os.path.join(figure_dir, f"{scale}", f"batch_{strt}", f"pred_{i}_topk{topk_i}.png"), img_with_heatmap)
        
                    