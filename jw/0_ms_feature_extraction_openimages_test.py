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
from config.feature_extraction_test import opt
import pdb


def get_label(file, partition_idx, partition_df, num_seen_classes, seen_labelmap, num_unseen_classes, unseen_labelmap):
    img_id = file.split('.')[0] #file.decode('utf-8').split('.')[0]
    df_img_label = partition_df[partition_idx].query('ImageID=="{}"'.format(img_id))
    seen_label = np.zeros(num_seen_classes, dtype=np.int32)
    unseen_label = np.zeros(num_unseen_classes, dtype=np.int32)
    for index, row in df_img_label.iterrows():
        if row['LabelName'] in seen_labelmap:
            idx = seen_labelmap.index(row['LabelName'])
            seen_label[idx] = 2*row['Confidence']-1
        if row['LabelName'] in unseen_labelmap:
            idx = unseen_labelmap.index(row['LabelName'])
            unseen_label[idx] = 2*row['Confidence']-1
    return seen_label, unseen_label

class DatasetExtract(Dataset):
    def __init__(self, data_dir, transform, files, partition_idxs, partition_df, num_seen_classes, seen_labelmap, num_unseen_classes, unseen_labelmap):
        super(DatasetExtract, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.files = files
        self.partition_idxs = partition_idxs
        self.partition_df = partition_df
        self.num_seen_classes = num_seen_classes
        self.seen_labelmap = seen_labelmap
        self.num_unseen_classes = num_unseen_classes
        self.unseen_labelmap = unseen_labelmap

    def __len__(self):
        return len(files)

    def __getitem__(self, index):
        file, partition_idx = self.files[index], self.partition_idxs[index]
    
        seen_label, unseen_label = get_label(file, partition_idx, self.partition_df, self.num_seen_classes, self.seen_labelmap, self.num_unseen_classes, self.unseen_labelmap)
        file_path = os.path.join(self.data_dir, file)
        img = Image.open(file_path).convert('RGB')
        img = transform(img)
        return file.encode("ascii", "ignore"), img, seen_label, unseen_label


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
    num_seen_classes = len(seen_labelmap)
    num_unseen_classes = len(unseen_labelmap)
    print(f'number of sample: {n_samples} | dataset: {opt.mode}')
    print(f'number of unique sampels: {len(np.unique(files))}')
    print(f'number of seen classes: {num_seen_classes} | number of unseen classes: {num_unseen_classes}')
    
    ### Load Dataset ###
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_extract = DatasetExtract(data_dir, transform, files, partition_idxs, partition_df, num_seen_classes, seen_labelmap, num_unseen_classes, unseen_labelmap)
    loader = DataLoader(dataset=dataset_extract, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=False)
    # pdb.set_trace()
    
    ### Multi-Scale Feature Extraction and Save with h5py ###
    import collections
    duplicate_files = [item for item, count in collections.Counter(files).items() if count > 1]
    feat = {key: [] for key in duplicate_files}
    
    fn = os.path.join(feature_dir, f'OPENIMAGES_{opt.mode}.h5')
    
    with h5py.File(fn, mode='w') as h5f:
        for i, data in enumerate(tqdm(loader), 0):
            _file, imgs, seen_label, unseen_label = data
            imgs = imgs.cuda()
            
            ms_outs = {}
            for scale in opt.scales:
                imgs = resize(imgs, list(np.asarray(imgs.shape[2:])*scale))
                with torch.no_grad():
                    outs = model(imgs)
                outs = np.float32(outs.cpu().numpy())
                ms_outs[scale] = outs
                
            seen_label = np.int8(seen_label.numpy())
            unseen_label = np.int8(unseen_label.numpy())
            
            bs = seen_label.shape[0]
            for m in range(bs):
                file = _file[m].decode("utf-8")
                if file in duplicate_files:
                    if len(feat[file]) == 2:
                        seen_label[m] = seen_label[m] + feat[file][0]
                        unseen_label[m] = unseen_label[m] + feat[file][1]
                        
                        for scale in opt.scales:
                            h5f.create_dataset(f'{file}-features-x{scale}', data=ms_outs[scale][m], dtype=np.float32, compression="gzip")    
                        h5f.create_dataset(f'{file}-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                        h5f.create_dataset(f'{file}-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")
                    else:
                        feat[file].append(seen_label[m])
                        feat[file].append(unseen_label[m])
                else:
                    for scale in opt.scales:
                        h5f.create_dataset(f'{file}-features-x{scale}', data=ms_outs[scale][m], dtype=np.float32, compression="gzip")    
                    h5f.create_dataset(f'{file}-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                    h5f.create_dataset(f'{file}-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")
                    # h5f.create_dataset(file+'-features', data=outs[m], dtype=np.float32, compression="gzip")
                    # h5f.create_dataset(file+'-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
                    # h5f.create_dataset(file+'-unseenlabels', data=unseen_label[m], dtype=np.int8, compression="gzip")
    
    ## Load Saved Feature
    saved_features = h5py.File(fn, 'r')
    feature_keys = list(saved_features.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in feature_keys]))
    print(len(image_names))
        
            