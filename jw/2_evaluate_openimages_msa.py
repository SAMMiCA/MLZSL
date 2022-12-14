import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model as model
from config.config_openimages import opt
from src.util_openimages import LoadMetaData, FilterSeenResults, FilterUnseenResults, FilterBothResults, compute_AP, compute_F1
from src.dataset import OpenImagesMSDatasetTest

import pdb

if __name__ == '__main__':
    print(opt)
    # model_path = os.path.join(opt.save_path, "model_latest_1300_10.pth")
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
        
        ### All Results ###
        prediction_7186[strt:endt,:] = logits_7186
        prediction_400[strt:endt,:] = logits_400
        prediction_7586[strt:endt,:] = logits_7586
        
        lab_7186[strt:endt,:] = batch_lab_7186
        lab_400[strt:endt,:] = batch_lab_400
        lab_7586[strt:endt,:] = batch_lab_7586
        
        image_names[strt:endt] = batch_image_names
    
    eval_device_id = 1
    eval_device = torch.device(f"cuda:{eval_device_id}")
    ### Evaluaion for Seen Classes ###
    if opt.cuda:
        lab_7186 = lab_7186.to(eval_device)
        prediction_7186 = prediction_7186.to(eval_device)
    
    map_lab_7186, map_prediction_7186, map_seen_labelmap, map_image_names = FilterSeenResults(lab_7186, prediction_7186, seen_labelmap, image_names)
    imgs_per_label = torch.clamp(map_lab_7186, 0, 1).sum(0)
    ### AP ###
    ap_7186 = compute_AP(map_prediction_7186, map_lab_7186, eval_device)
    print('SEEN AP on 4728 classes', torch.mean(ap_7186).item())
    ### Weighted AP ###
    weighted_map_7186 = (imgs_per_label.float() * ap_7186).sum()/imgs_per_label.sum().float()
    print('WEIGHTED SEEN AP on 4728 classes', weighted_map_7186.item())
    ### Delete for CUDA Memory
    del weighted_map_7186, ap_7186, imgs_per_label, lab_7186, prediction_7186
    torch.cuda.empty_cache()
    ### Top-K F1 Score ###
    for k in [20, 10, 5, 1]:
        map_prediction_7186.clone()
        F1_7186, P_7186, R_7186 = compute_F1(map_prediction_7186, map_lab_7186, 'overall', k_val=k, device=eval_device)
        print(f'g_{k}={k}',torch.mean(F1_7186).item(), torch.mean(P_7186).item(), torch.mean(R_7186).item())    
    del map_prediction_7186, map_lab_7186, F1_7186, P_7186, R_7186
    torch.cuda.empty_cache()
    
    ### Evaluation for Unseen Classes (ZSL) ###
    if opt.cuda:
        lab_400 = lab_400.to(eval_device)
        prediction_400 = prediction_400.to(eval_device)
    map_lab_400, map_prediction_400, map_unseen_labelmap, map_image_names = FilterUnseenResults(lab_400, prediction_400, unseen_labelmap, idx_top_unseen, image_names)
    imgs_per_label = torch.clamp(map_lab_400,0,1).sum(0)
    ### AP ###
    ap_400 = compute_AP(prediction_400, lab_400, eval_device)
    print('ZSL AP', torch.mean(ap_400).item())
    ### Weighted AP ###
    weighted_map_400 = (imgs_per_label.float() * ap_400).sum()/imgs_per_label.sum().float()
    print('WEIGHTED ZSL AP',torch.mean(weighted_map_400).item())  
    ### Delete for CUDA Memory
    del weighted_map_400, ap_400, imgs_per_label, lab_400, prediction_400
    torch.cuda.empty_cache()
    ### Top-K F1 Score ###
    for k in [20, 10, 5, 1]:
        map_prediction_400.clone()
        F1_400, P_400, R_400 = compute_F1(map_prediction_400, map_lab_400, 'overall', k_val=k, device=eval_device)
        print(f'g_{k}={k}',torch.mean(F1_400).item(), torch.mean(P_400).item(), torch.mean(R_400).item())
    del map_prediction_400, map_lab_400, F1_400, P_400, R_400
    # pdb.set_trace()
    
    ### Evaluation for Seen and Unseen Classes (GZSL) ###
    if opt.cuda:
        lab_7586 = lab_7586.to(eval_device)
        prediction_7586 = prediction_7586.to(eval_device)    
    map_lab_7586, map_prediction_7586, map_both_labelmap, map_image_names = FilterBothResults(lab_7586, prediction_7586, seen_labelmap, unseen_labelmap, idx_top_unseen, image_names)
    imgs_per_label = torch.clamp(map_lab_7586,0,1).sum(0)
    ### AP ###
    ap_7586 = compute_AP(map_prediction_7586, map_lab_7586, eval_device)
    print('GZSL AP on 4728+400 classes',torch.mean(ap_7586).item())
    ### Weighted AP ###
    weighted_map_7586 = (imgs_per_label.float() * ap_7586).sum()/imgs_per_label.sum().float()
    print('WEIGHTED GZSL AP on 4728+400 classes', weighted_map_7586.item())
    ### Delete for CUDA Memory
    del weighted_map_7586, ap_7586, imgs_per_label, lab_7586, prediction_7586
    torch.cuda.empty_cache()
    ### Top-K F1 Score ###
    for k in [20, 10, 5, 1]:
        map_prediction_7586.clone()
        F1_7586, P_7586, R_7586 = compute_F1(map_prediction_7586, map_lab_7586, 'overall', k_val=k, device=eval_device)
        print(f'g_{k}={k}',torch.mean(F1_7586).item(), torch.mean(P_7586).item(), torch.mean(R_7586).item())
    del map_prediction_7586, map_lab_7586, F1_7586, P_7586, R_7586
    pdb.set_trace()
    