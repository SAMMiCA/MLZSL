from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from src.model import VGG_Feature_Extractor
from src.util_openimages import LoadMetaData
import os
from config.feature_extraction_train import opt
import pdb


def get_label(file, partition_idx, partition_df, num_classes, seen_labelmap):
    img_id = file.split('.')[0]
    df_img_label = partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    label = np.zeros(num_classes, dtype=np.int32)
    for index, row in df_img_label.iterrows():
        if row['LabelName'] not in seen_labelmap:
            continue #not trainable classes
        idx=seen_labelmap.index(row['LabelName'])
        label[idx] = 2*row['Confidence']-1
    return label


class DatasetExtract(Dataset):
    def __init__(self, data_dir, transform, files, partition_idxs, partition_df, num_classes, seen_labelmap):
        super(DatasetExtract, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.files = files
        self.partition_idxs = partition_idxs
        self.partition_df = partition_df
        self.num_classes = num_classes
        self.seen_labelmap = seen_labelmap

    def __len__(self):
        return len(files)

    def __getitem__(self, index):
        
        file, partition_idx = self.files[index], self.partition_idxs[index]
    
        seen_label = get_label(file, partition_idx, self.partition_df, self.num_classes, self.seen_labelmap)
        file_path = os.path.join(self.data_dir, file)
        try:
            img = Image.open(file_path).convert('RGB')
            img = self.transform(img)
            flag=0
        except:
            img = torch.zeros(3,224,224)
            flag=1
        return file.encode("ascii", "ignore"), img, seen_label, flag


if __name__ == '__main__':
    ### Load Meta Data ###
    seen_labelmap_path = os.path.join(opt.src, opt.version, 'classes-trainable.txt')
    unseen_labelmap_path = os.path.join(opt.src, opt.version, 'unseen_labels.pkl')
    dict_path = os.path.join(opt.src, opt.version, 'class-descriptions.csv')
    seen_labelmap, unseen_labelmap, label_dict = LoadMetaData(seen_labelmap_path, unseen_labelmap_path, dict_path) # Meta Data
    label_path = os.path.join(opt.src, opt.version, opt.mode, f'{opt.mode}-annotations-human-imagelabels.csv')
    
    ### Load Settings ###
    data_dir = os.path.join(opt.data_dir, opt.mode) # Raw image directory
    feature_dir = os.path.join(opt.src, 'ms_features', opt.mode)
    os.makedirs(feature_dir, exist_ok=True)
    
    ### Load Models ###
    model = VGG_Feature_Extractor()
    model.eval()
    if torch.cuda.is_available():
        # device_ids = [i for i in range(torch.cuda.device_count())]
        device_ids = [0, 1]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = model.cuda()
    
    ### Partition ###
    df_label = pd.read_csv(label_path)
    capacity = opt.capacity
    t = len(df_label)//capacity
    partition_df = [df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity] for idx_cut in range(t)]
    partition_df.append(df_label.iloc[t*capacity:])
    
    ### Reading files from the local OpenImages folder ###
    files = []
    partition_idxs = []
    for idx_partition, partition in enumerate(partition_df):
        file_partition = [f'{img_id}.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(os.path.join(data_dir, f'{img_id}.jpg'))]
        files.extend(file_partition)
        partition_idxs.extend([idx_partition]*len(file_partition))
    
    n_samples = len(files)
    num_classes = len(seen_labelmap)
    print(f'number of sample: {n_samples} | dataset: {opt.mode}')
    print(f'number of unique sampels: {len(np.unique(files))}')
    
    ### Load Dataset ###
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_extract = DatasetExtract(data_dir, transform, files, partition_idxs, partition_df, num_classes, seen_labelmap)
    loader = DataLoader(dataset=dataset_extract, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=False)
    
    ### Multi-Scale Feature Extraction and Save with h5py ###
    import collections
    duplicate_files = [item for item, count in collections.Counter(files).items() if count > 1]
    feat = {key: [] for key in duplicate_files}
            
    # feature_scale_dir = os.path.join(feature_dir, f'{scale}')
    # os.makedirs(feature_scale_dir, exist_ok=True)
    
    # feature_size = 196*scale*scale
    # chunk_imgs, chunk_features, chunk_seenlabels = np.empty((num_chunk_data), dtype=np.dtype('U20')), np.empty((num_chunk_data,512,feature_size)), np.empty((num_chunk_data,7186))
    num_chunk_data = opt.batch_size * opt.num_save_iter
    chunk_imgs, chunk_seenlabels = np.empty((num_chunk_data), dtype=np.dtype('U20')), np.empty((num_chunk_data,7186))
    chunk_ms_features = {scale: np.empty((num_chunk_data,512,196*scale*scale)) for scale in opt.scales}

    for i, data in enumerate(tqdm(loader), 0):
        _file, imgs, seen_label, flag = data
        keep = []
        for j,f in enumerate(flag):
            if f == 0:
                keep.append(j)
        _file, imgs, seen_label = np.array(_file)[keep], imgs[keep], seen_label[keep]
        
        imgs = imgs.cuda()
        ms_outs = {}
        for scale in opt.scales:
            imgs = resize(imgs, list(np.asarray(imgs.shape[2:])*scale))
            with torch.no_grad():
                outs = model(imgs)
            outs = np.float32(outs.cpu().numpy())
            ms_outs[scale] = outs
            
        seen_label = np.int8(seen_label.numpy())
        
        ### Make chunk ###
        start_t = i%opt.num_save_iter * opt.batch_size 
        end_t = start_t + _file.shape[0]
        chunk_imgs[start_t:end_t,] = _file
        for scale in opt.scales:
            chunk_ms_features[scale][start_t:end_t,:] = ms_outs[scale]
        # chunk_features[start_t:end_t,:] = outs
        chunk_seenlabels[start_t:end_t,:] = seen_label

        if i%opt.num_save_iter == opt.num_save_iter - 1:
            fn = os.path.join(feature_dir, f'OPENIMAGES_{opt.mode}_CHUNK_{i+1}.h5')
            chunk_bs = chunk_imgs.shape[0]
            with h5py.File(fn, mode='w') as h5f:
                for m in range(chunk_bs):
                    dict_files = chunk_imgs[m]
                    if dict_files in duplicate_files:
                        if len(feat[dict_files]) == 1:
                            chunk_seenlabels[m] = chunk_seenlabels[m] + feat[dict_files]
                            for scale in opt.scales:
                                # h5f.create_dataset(dict_files+'-features', data=chunk_features[m], dtype=np.float32, compression="gzip")
                                h5f.create_dataset(f'{dict_files}-features-x{scale}', data=chunk_ms_features[scale][m], dtype=np.float32, compression="gzip")
                            # h5f.create_dataset(dict_files+'-seenlabels', data=chunk_seenlabels[m], dtype=np.int8, compression="gzip")
                            h5f.create_dataset(f'{dict_files}-seenlabels', data=chunk_seenlabels[m], dtype=np.int8, compression="gzip")
                        else:
                            feat[dict_files].append(chunk_seenlabels[m])
                    else:
                        for scale in opt.scales:
                                # h5f.create_dataset(dict_files+'-features', data=chunk_features[m], dtype=np.float32, compression="gzip")
                                h5f.create_dataset(f'{dict_files}-features-x{scale}', data=chunk_ms_features[scale][m], dtype=np.float32, compression="gzip")
                        h5f.create_dataset(f'{dict_files}-seenlabels', data=chunk_seenlabels[m], dtype=np.int8, compression="gzip")
            chunk_imgs, chunk_seenlabels = np.empty((num_chunk_data), dtype=np.dtype('U20')), np.empty((num_chunk_data,7186))
            chunk_ms_features = {scale: np.empty((num_chunk_data,512,196*scale*scale)) for scale in opt.scales}
    
    ## Load Saved Feature
    saved_features = h5py.File(fn, 'r')
    feature_keys = list(saved_features.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in feature_keys]))
    print(len(image_names))
    
    # ### Multi-Scale Feature Extraction and Save with h5py ###
    # for scale in opt.scales:
    #     print(f"Scale: x{scale}")

    #     import collections
    #     duplicate_files = [item for item, count in collections.Counter(files).items() if count > 1]
    #     feat = {key: [] for key in duplicate_files}
                
    #     feature_scale_dir = os.path.join(feature_dir, f'{scale}')
    #     os.makedirs(feature_scale_dir, exist_ok=True)
        
    #     num_chunk_data = opt.batch_size * opt.num_save_iter
    #     feature_size = 196*scale*scale
    #     chunk_imgs, chunk_features, chunk_seenlabels = np.empty((num_chunk_data), dtype=np.dtype('U20')), np.empty((num_chunk_data,512,feature_size)), np.empty((num_chunk_data,7186))
        
    #     for i, data in enumerate(tqdm(loader), 0):
    #         _file, imgs, seen_label, flag = data
    #         keep = []
    #         for j,f in enumerate(flag):
    #             if f == 0:
    #                 keep.append(j)
    #         _file, imgs, seen_label = np.array(_file)[keep], imgs[keep], seen_label[keep]
            
    #         imgs = imgs.cuda()
    #         imgs = resize(imgs, list(np.asarray(imgs.shape[2:])*scale))
    #         with torch.no_grad():
    #             outs = model(imgs)
            
    #         outs = np.float32(outs.cpu().numpy())
    #         seen_label = np.int8(seen_label.numpy())
            
    #         ### Make chunk ###
    #         start_t = i%opt.num_save_iter * opt.batch_size 
    #         end_t = start_t + _file.shape[0]
    #         chunk_imgs[start_t:end_t,] = _file
    #         chunk_features[start_t:end_t,:] = outs
    #         chunk_seenlabels[start_t:end_t,:] = seen_label

    #         if i%opt.num_save_iter == opt.num_save_iter - 1:
    #             fn = os.path.join(feature_scale_dir, f'OPENIMAGES_CONV5_4_NO_CENTERCROP_{opt.mode}_CHUNK_{i+1}.h5')
    #             chunk_bs = chunk_imgs.shape[0]
    #             with h5py.File(fn, mode='w') as h5f:
    #                 for m in range(chunk_bs):
    #                     dict_files = chunk_imgs[m]
    #                     if dict_files in duplicate_files:
    #                         if len(feat[dict_files]) == 1:
    #                             chunk_seenlabels[m] = chunk_seenlabels[m] + feat[dict_files]
    #                             h5f.create_dataset(dict_files+'-features', data=chunk_features[m], dtype=np.float32, compression="gzip")
    #                             h5f.create_dataset(dict_files+'-seenlabels', data=chunk_seenlabels[m], dtype=np.int8, compression="gzip")
    #                         else:
    #                             feat[dict_files].append(chunk_seenlabels[m])
    #                     else:
    #                         h5f.create_dataset(dict_files+'-features', data=chunk_features[m], dtype=np.float32, compression="gzip")
    #                         h5f.create_dataset(dict_files+'-seenlabels', data=chunk_seenlabels[m], dtype=np.int8, compression="gzip")
    #             chunk_imgs, chunk_features, chunk_seenlabels = np.empty((num_chunk_data), dtype=np.dtype('U20')), np.empty((num_chunk_data,512,feature_size)), np.empty((num_chunk_data,7186))
