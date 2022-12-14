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

# import model as model
# # import util_nus as util



# import time
# import os
# import socket
# import pickle
# import logging
# from warmup_scheduler import GradualWarmupScheduler
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

    os.makedirs(opt.save_path, exist_ok=True)
    
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
    model_vgg.eval()
    
        
    ## load optimizer ###
    optimizer = torch.optim.Adam(model_joint_emb.parameters(), opt.lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))
    start_epoch = 1
    num_epochs = opt.nepoch+1

    if opt.cosinelr_scheduler:
        print("------------------------------------------------------------------")
        print("USING LR SCHEDULER")
        print("------------------------------------------------------------------")
        ######### Scheduler ###########
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=opt.lr_min)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    
    
    train_h5_path = '/root/datasets/NUS_WIDE/ms_features/NUS_WIDE_train.h5'
    dataset = DatasetNUS(train_h5_path, scales)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        mean_loss = 0
        # dataset[33247]
        # for i in range(129*256, 130*256):
            
        #     print(i, dataset[i]['seenlabels'])
        # pdb.set_trace()
        for i, batch_data in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            train_labels = batch_data['seenlabels']
            # pdb.set_trace()
            ### remove empty label images ###
            temp_label = torch.clamp(train_labels,0,1)
            temp_seen_labels = temp_label.sum(1)
            
            ms_train_inputs = {scale: batch_data[scale][temp_seen_labels>0] for scale in scales}
            train_labels = train_labels[temp_seen_labels>0]
            ##### TO DO !!!! extract features -> labels -1 and 1
            # train_labels = train_labels*2 -1
            ### if available, to device ###
            if torch.cuda.is_available:
                train_labels = train_labels.to(device_main)
                for scale in opt.scales:
                    ms_train_inputs[scale] = ms_train_inputs[scale].to(device_main)
            
            vgg_4096 = model_vgg(ms_train_inputs[1]) if model_vgg is not None else None
            logits = model_joint_emb(ms_train_inputs[1], vgg_4096)
            loss = model.ranking_lossT(logits, train_labels.float())
            
            mean_loss += loss.item()
            if torch.isnan(loss) or loss.item() > 100:
                print('Unstable/High Loss:', loss)
                pdb.set_trace()
            
            loss.backward()
            optimizer.step()
            print(f'{i} || {loss.item()}')
            # pdb.set_trace()
        mean_loss /= len(dataset)
        
        if opt.cosinelr_scheduler:
            learning_rate = scheduler.get_lr()[0]
        else:
            learning_rate = opt.lr

        print("------------------------------------------------------------------")
        print(f"Epoch:{epoch}/{opt.nepoch} \tLoss: {mean_loss} \tLearningRate {learning_rate}")
        print("------------------------------------------------------------------")

        torch.save(model_joint_emb.module.state_dict(), os.path.join(opt.save_path, f"model_best_train_full_{epoch}.pth"))
        
        if opt.cosinelr_scheduler:
            scheduler.step()
            
            
            