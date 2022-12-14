import json
import numpy as np
import pdb
from src.dataset_nus import LoadAllDataNUS, DatasetExtract
from src.vis import vis_img_labels
import os
from PIL import Image
import pickle
from src.model import VGG_Feature_Extractor
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

# from config.feature_extraction_nus import opt


if __name__ == '__main__':
    ### Load NUS-WIDE Info ###
    data_mode = ['train', 'test']
    feature_dir = '/root/datasets/NUS_WIDE/ms_features'
    os.makedirs(feature_dir, exist_ok=True)
    scales = [1, 2]
    batch_size = 64
    file_tag1k = '/root/datasets/NUS_WIDE/NUS_WID_Tags/TagList1k.txt'
    file_tag81 = '/root/datasets/NUS_WIDE/ConceptsList/Concepts81.txt'
    
    id_filename = '/root/datasets/NUS_WIDE/ImageList/Imagelist.txt'
    label1k_filename = '/root/datasets/NUS_WIDE/NUS_WID_Tags/AllTags1k.txt'
    label81_human_filename = '/root/datasets/NUS_WIDE/AllLabels/'
    
    img_list, tag925, tag81, label925_imgs, label81_imgs = LoadAllDataNUS(file_tag1k, file_tag81, id_filename, label1k_filename, label81_human_filename)
    
    train_filename = '/root/datasets/NUS_WIDE/ImageList/TrainImagelist.txt'
    test_filename = '/root/datasets/NUS_WIDE/ImageList/TestImagelist.txt'
    
    all_nus_lab925_dict = {}
    all_nus_lab81_dict = {}
    for idx, img_id in enumerate(img_list):
        all_nus_lab925_dict[img_id] = label925_imgs[idx]
        all_nus_lab81_dict[img_id] = label81_imgs[idx]
    # #### ####
    # test = []
    # for i in img_list:
    #     a = os.path.join('/root/datasets/NUS_WIDE/Flickr', i)
    #     if os.path.isfile(a):
    #         test.append(i)
    #         print(i)
    # pdb.set_trace()
    # #### ####
    
    ### Load Models ###
    model = VGG_Feature_Extractor()
    model.eval()
    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        # device_ids = [0]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = model.cuda()
    
    ### Feature Extraction ###
    for mode in data_mode:
        if mode == 'train':
            with open(train_filename, "r") as file:
                train_img_list = file.readlines()
                train_img_list = [img_name.rstrip().replace('\\','/') for img_name in train_img_list]
            mode_img_list = train_img_list
            fn = os.path.join(feature_dir, f'NUS_WIDE_train.h5')

        elif mode == 'test':
            with open(test_filename, "r") as file:
                test_img_list = file.readlines()
                test_img_list = [img_name.rstrip().replace('\\','/') for img_name in test_img_list]
            mode_img_list = test_img_list
            fn = os.path.join(feature_dir, f'NUS_WIDE_test.h5')
            
        mode_label925_imgs = [all_nus_lab925_dict[mode_img_id] for mode_img_id in mode_img_list]
        mode_label81_imgs = [all_nus_lab81_dict[mode_img_id] for mode_img_id in mode_img_list]
        
        mode_dataset = DatasetExtract(mode_img_list, mode_label925_imgs, mode_label81_imgs, raw_img_dir='/root/datasets/NUS_WIDE/Flickr')
        mode_loader = DataLoader(dataset=mode_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        
        with h5py.File(fn, mode='w') as h5f:
            for i, data in enumerate(tqdm(mode_loader), 0):
                _img_id, imgs, label925, label81 = data
                imgs = imgs.cuda()
                label925, label81 = label925*2 - 1, label81*2 - 1
                ms_outs = {}
                for scale in scales:
                    imgs = resize(imgs, list(np.asarray(imgs.shape[2:])*scale))
                    with torch.no_grad():
                        outs = model(imgs)
                    outs = np.float32(outs.cpu().numpy())
                    ms_outs[scale] = outs
                    
                seen_label = np.int8(label925.numpy())
                unseen_label = np.int8(label81.numpy())
                bs = seen_label.shape[0]
                for m in range(bs):
                    # pdb.set_trace()
                    img_id = _img_id[m].replace('/', '__')
                    for scale in scales:
                        h5f.create_dataset(f'{img_id}-features{scale}', data=ms_outs[scale][m], dtype=np.float32, compression="gzip")    
                    h5f.create_dataset(f'{img_id}-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                    h5f.create_dataset(f'{img_id}-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")
                    
                    
    ################################# Visualize image-label ###################################
    # data_path= '/root/datasets/NUS_WIDE/Flickr/'			#path to imgs data  
    # fig_dir = './figures/220405_nus_test'
    # os.makedirs(fig_dir, exist_ok=True)
    # for i in range(2000, 2100):
    #     img_id = img_list[i]
    #     img_path = os.path.join(data_path, img_id)
    #     fig_path = os.path.join(fig_dir, f"{i}.png")
    #     # pdb.set_trace()
        
    #     img_labels = tag925[np.nonzero(label925_imgs[i])[0]]
        
    #     if  os.path.exists(img_path):
    #         x = Image.open(img_path).convert('RGB')
    #         vis_img_labels(img_path, fig_path, img_labels)
    
    
    ########################################################################
    
    