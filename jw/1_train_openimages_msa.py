from config.config_openimages import opt
from src.dataset import LoadOpenImagesChunkDatset
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import src.model as model
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import os 
from src.util_openimages import LoadMetaData
import pandas as pd
import pickle
from sklearn.preprocessing import normalize
import torch.nn as nn
from tqdm import tqdm
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
    ## load metadata ###
    
    seen_labelmap_path = os.path.join(opt.src, opt.version, 'classes-trainable.txt')
    unseen_labelmap_path = os.path.join(opt.src, opt.version, 'unseen_labels.pkl')
    dict_path = os.path.join(opt.src, opt.version, 'class-descriptions.csv')
    top_unseen_path = os.path.join(opt.src, opt.version, 'top_400_unseen.csv')
    att_path = os.path.join(opt.src, 'wiki_contexts', 'OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl')
    seen_labelmap, unseen_labelmap, label_dict = LoadMetaData(seen_labelmap_path, unseen_labelmap_path, dict_path)
    df_top_unseen = pd.read_csv(top_unseen_path, header=None)
    idx_top_unseen = df_top_unseen.values[:, 0]
    assert len(idx_top_unseen) == 400
    src_att = pickle.load(open(att_path, 'rb'))
    vecs_7186 = torch.from_numpy(normalize(src_att[0]))
    vecs_400 = torch.from_numpy(normalize(src_att[1][idx_top_unseen,:]))
    if opt.cuda:
        # vecs_7186 = vecs_7186.cuda()
        # vecs_400 = vecs_400.cuda()
        vecs_7186 = vecs_7186.to("cuda:1")
        vecs_400 = vecs_400.to("cuda:1")
    gzsl_vecs = torch.cat([vecs_7186, vecs_400],0)
    
    ### load models ###
    model_vgg = model.VGG_Global_Feature_Extractor()
    model_BiAM = model.MSAV1(opt, dim_feature=[196,512], vecs=vecs_7186)
    if opt.cuda:
        # model_vgg = model_vgg.cuda()
        model_vgg = model_vgg.to("cuda:1")
        
        # device_ids = [i for i in range(torch.cuda.device_count())]
        device_ids = [1, 2]
        
        if len(device_ids) > 1:
            # model_BiAM = nn.DataParallel(model_BiAM, device_ids=device_ids).cuda()
            # model_BiAM = nn.DataParallel(model_BiAM, device_ids=device_ids)
            model_BiAM = model_BiAM.to("cuda:1")
        else:
            model_BiAM = model_BiAM.cuda()
    model_vgg.eval()
    
    ### load optimizer ###
    optimizer = torch.optim.Adam(model_BiAM.parameters(), opt.lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))
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
    
    ### train ###
    chunk_datasets = LoadOpenImagesChunkDatset(opt)
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        for chunk_id in range(len(chunk_datasets)):
            chunk_dataset = chunk_datasets[chunk_id]
            chunk_dataloader = DataLoader(dataset=chunk_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=False)
            
            mean_loss = 0
            for i, batch_data in enumerate(chunk_dataloader):
                optimizer.zero_grad()
                train_labels = batch_data['seenlabels'].long()
                temp_label = torch.sum(train_labels > 0, 1) > 0 # remove images that do not have positive label
                train_labels = train_labels[temp_label]
                ms_train_inputs = {scale: batch_data[scale][temp_label] for scale in opt.scales}
                ## Train with images containing 40 or less than 40 labels
                _train_labels = train_labels[torch.clamp(train_labels,0,1).sum(1)<=40]
                for scale in opt.scales:
                    ms_train_inputs[scale] = ms_train_inputs[scale][torch.clamp(train_labels,0,1).sum(1)<=40]
                if torch.cuda.is_available:
                    # _train_labels = _train_labels.cuda()
                    _train_labels = _train_labels.to("cuda:1")
                    for scale in opt.scales:
                        # ms_train_inputs[scale] = ms_train_inputs[scale].cuda()
                        ms_train_inputs[scale] = ms_train_inputs[scale].to("cuda:1")

                vgg_4096 = model_vgg(ms_train_inputs[1]) if model_vgg is not None else None
                # logits = model_BiAM(ms_train_inputs[1], vecs_7186, vgg_4096)

                logits = model_BiAM(ms_train_inputs, vgg_4096)
                # pdb.set_trace()
                loss = model.ranking_lossT(logits, _train_labels.float())

                mean_loss += loss.item()

                if torch.isnan(loss) or loss.item() > 100:
                    print('Unstable/High Loss:', loss)
                    import pdb; pdb.set_trace()

                loss.backward()
                optimizer.step()
            mean_loss /= len(chunk_datasets) / opt.batch_size

            if opt.cosinelr_scheduler:
                learning_rate = scheduler.get_lr()[0]
            else:
                learning_rate = opt.lr
            
            print("------------------------------------------------------------------")
            print(f"Epoch:{epoch}/{opt.nepoch} \tBatch: {chunk_id}/{len(chunk_datasets)} \tLoss: {mean_loss} \tLearningRate {learning_rate}")
            print("------------------------------------------------------------------")
            
            if (chunk_id > 3 and chunk_id % 100 == 0):
                torch.save(model_BiAM.state_dict(), os.path.join(opt.save_path, f"model_latest_{chunk_id}_{epoch}.pth"))
                                                                 
        if opt.cosinelr_scheduler:
            scheduler.step()

