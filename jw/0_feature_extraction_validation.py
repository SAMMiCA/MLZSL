from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from src.model import VGG_Feature_Extractor
import os

from config.feature_extraction_val import opt
import pdb


def LoadLabelMap(labelmap_path, dict_path):
    f_labelmap = open(labelmap_path, 'r')
    labelmap = [line.rstrip() for line in f_labelmap.readlines()]
    # labelmap = [line.rstrip() for line in tf.io.gfile.GFile(labelmap_path)]
        
    label_dict = {}
    f_dict = open(dict_path, 'r', encoding="utf-8")
    for line in f_dict.readlines():
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]
    # label_dict = {}
    # for line in tf.io.gfile.GFile(dict_path):
    #     words = [word.strip(' "\n') for word in line.split(',', 1)]
    #     label_dict[words[0]] = words[1]
    return labelmap, label_dict


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict  


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
    ## Load Settings ##
    version = '2022_01'
    root_dir = '/root/datasets/OpenImages'
    mode = 'validation'
    label_dir = os.path.join(root_dir, version)
    label_path = os.path.join(label_dir, mode, f'{mode}-annotations-human-imagelabels.csv') # Human-verified labels
    labelmap_path = os.path.join(label_dir, 'classes-trainable.txt') # Metadata
    dict_path = os.path.join(label_dir, 'class-descriptions.csv') # Metadata
    data_dir = os.path.join('/root/jwssd/datasets/OpenImages', mode) # Raw images
    
    batch_size = 32
    feature_save_dir = os.path.join(root_dir, f'{mode}_features_jw')
    os.makedirs(feature_save_dir, exist_ok=True)
    
    
    ## Load Models ##
    model = VGG_Feature_Extractor()
    model.eval()
    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = model.cuda()
    
    
    ## Partition
    df_label = pd.read_csv(label_path)
    capacity = 40000
    partition_df = []
    t = len(df_label)//capacity
    for idx_cut in range(t):
        partition_df.append(df_label.iloc[idx_cut*capacity:(idx_cut+1)*capacity])
    partition_df.append(df_label.iloc[t*capacity:])
    
    
    ## Reading files from the local OpenImages folder
    files = []
    partition_idxs = []
    for idx_partition, partition in enumerate(partition_df):
        file_partition = [f'{img_id}.jpg' for img_id in partition['ImageID'].unique() if os.path.isfile(os.path.join(data_dir, f'{img_id}.jpg'))]
        files.extend(file_partition)
        partition_idxs.extend([idx_partition]*len(file_partition))
    
    n_samples = len(files)
    print(f'number of sample: {n_samples} | dataset: {mode}')
    print(f'number of unique sampels: {len(np.unique(files))}')

    predictions_eval = 0
    predictions_eval_resize = 0
    
    seen_labelmap, label_dict = LoadLabelMap(labelmap_path, dict_path)
    num_classes = len(seen_labelmap)

    ## Load Dataset ##
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_extract = DatasetExtract(data_dir, transform, files, partition_idxs, partition_df, num_classes, seen_labelmap)
    loader = DataLoader(dataset=dataset_extract, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    
    
    ## Feature Extraction and Save with h5py
    fn = os.path.join(feature_save_dir, f'OPENIMAGES_CONV5_4_NO_CENTERCROP_{mode}.h5')
    xx = {}
    with h5py.File(fn, mode='w') as h5f:
        for i, data in enumerate(tqdm(loader), 0):
            _file, imgs, seen_label, flag = data
            keep = []
            for j,f in enumerate(flag):
                if f == 0:
                    keep.append(j)
            _file, imgs, seen_label = np.array(_file)[keep], imgs[keep], seen_label[keep]
            imgs = imgs.cuda()
            pdb.set_trace()
            with torch.no_grad():
                outs = model(imgs)
            outs = np.float32(outs.cpu().numpy())
            seen_label = np.int8(seen_label.numpy())
            bs = outs.shape[0]
            
            for m in range(bs):
                file = _file[m].decode("utf-8")
                print(file)
                if file in xx.keys():
                    with h5py.File(fn, mode='a') as h5f_1:
                        del h5f_1[file+'-features']
                        del h5f_1[file+'-seenlabels']
                    h5f_1.close()
                    seen_label[m] = seen_label[m] + xx[file]['seen_label']

                xx[file] = {}
                xx[file]['seen_label'] = seen_label[m]

                h5f.create_dataset(file+'-features', data=outs[m], dtype=np.float32, compression="gzip")
                h5f.create_dataset(file+'-seenlabels', data=seen_label[m], dtype=np.int8, compression="gzip")
        
    pdb.set_trace()
    
    
    ## Load Saved Feature
    saved_features = h5py.File(fn, 'r')
    feature_keys = list(saved_features.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in feature_keys]))
    print(len(image_names))
    
    a = h5py.File('/root/datasets/OpenImages/jw_validation_features/OPENIMAGES_VAL_CONV5_4_NO_CENTERCROP.h5')
    b = list(a.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in b]))
    print(len(image_names))