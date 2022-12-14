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

import pdb

if __name__ == '__main__':
    print(opt)

    ### setting up seeds ###
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)
    torch.set_default_tensor_type('torch.FloatTensor')
    cudnn.benchmark = True  # For speed i.e, cudnn autotuner
    
    ### load metadata ###
    src = opt.src
    att_path = os.path.join(src,'wiki_contexts','NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
    file_tag1k = os.path.join(src,'NUS_WID_Tags','TagList1k.txt')
    file_tag81 = os.path.join(src,'ConceptsList','Concepts81.txt')
    seen_cls_idx, _, _, _ = get_seen_unseen_classes(file_tag1k, file_tag81)
    src_att = pickle.load(open(att_path, 'rb'))
    
    vecs_925 = torch.from_numpy(normalize(src_att[0][seen_cls_idx]))
    vecs_81 = torch.from_numpy(normalize(src_att[1]))
    scales = opt.scales
    
    ### load models ###
    model_vgg = model.VGG_Global_Feature_Extractor()
    model_joint_emb = model.MSBiAM(opt, vecs=vecs_925)
    
    if torch.cuda.is_available():
        # pdb.set_trace()
        device_ids = [i for i in range(torch.cuda.device_count())]
        # device_ids = [2, 3]
        device_main = torch.device(f"cuda:{device_ids[0]}")
        if len(device_ids) > 1:
            model_joint_emb = nn.DataParallel(model_joint_emb, device_ids=device_ids).cuda()
            model_vgg = nn.DataParallel(model_vgg, device_ids=device_ids).cuda()
        else:
            model_joint_emb = model_joint_emb.to(device_main)
            model_vgg = model_vgg.to(device_main)
           
    model_path = opt.save_path
    # pdb.set_trace()
    model_joint_emb.load_state_dict(torch.load(model_path))
    
    model_joint_emb.eval()
    model_vgg.eval()
        
    test_h5_path = '/root/datasets/NUS_WIDE/ms_features/NUS_WIDE_test.h5'
    dataset = DatasetNUS(test_h5_path, scales)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
    
    num_test = len(dataset)
    
    pred_81 = torch.empty(num_test, 81)
    pred_925 = torch.empty(num_test, 925)
    lab_81 = torch.empty(num_test, 81)
    lab_925 = torch.empty(num_test, 925)
    
    start_t = 0
    for i, batch_data in enumerate(tqdm(dataloader)):
        seen_labels = batch_data['seenlabels']
        unseen_labels = batch_data['unseenlabels']
        bs = seen_labels.shape[0]

        seen_labels = seen_labels*2 - 1
        unseen_labels = unseen_labels*2 - 1
        ms_test_inputs = {scale: batch_data[scale] for scale in scales}

        if torch.cuda.is_available:
            seen_labels = seen_labels.to(device_main)
            unseen_labels = unseen_labels.to(device_main)
            for scale in opt.scales:
                ms_test_inputs[scale] = ms_test_inputs[scale].to(device_main)
        
        with torch.no_grad():
            vgg_4096 = model_vgg(ms_test_inputs[1]) if model_vgg is not None else None
            
            model_joint_emb.module.vecs = vecs_925
            # model_joint_emb.vecs = vecs_925
            logits_925 = model_joint_emb(ms_test_inputs[1], vgg_4096)
            
            model_joint_emb.module.vecs = vecs_81
            # model_joint_emb.vecs = vecs_81
            logits_81 = model_joint_emb(ms_test_inputs[1], vgg_4096)

            # logits_925 = model_joint_emb(ms_test_inputs[1], vgg_4096)
        
        lab_925[start_t:start_t+bs] = seen_labels
        lab_81[start_t:start_t+bs] = unseen_labels
        pred_925[start_t:start_t+bs] = logits_925
        pred_81[start_t:start_t+bs] = logits_81

        start_t += bs
    

    ap_81 = compute_AP(pred_81.cuda(), lab_81.cuda())
    F1_3_81,P_3_81,R_3_81 = compute_F1(pred_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
    print(f"mAP: {torch.mean(ap_81).item()}")
    print(f"F1 at Top-3: {F1_3_81.item()}")
    pdb.set_trace()
        # ### remove empty label images ###
        # temp_label = torch.clamp(train_labels,0,1)
        # temp_seen_labels = temp_label.sum(1)
        
        # ms_train_inputs = {scale: batch_data[scale][temp_seen_labels>0] for scale in scales}
        # train_labels = train_labels[temp_seen_labels>0]
        # ##### TO DO !!!! extract features -> labels -1 and 1
        # train_labels = train_labels*2 - 1
        # ### if available, to device ###
        # if torch.cuda.is_available:
        #     train_labels = train_labels.to(device_main)
        #     for scale in opt.scales:
        #         ms_train_inputs[scale] = ms_train_inputs[scale].to(device_main)
        
        # loss = model.ranking_lossT(logits, train_labels.float())
        