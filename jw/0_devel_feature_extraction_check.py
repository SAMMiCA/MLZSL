import h5py
import numpy as np
import os

import pdb


if __name__ == '__main__':
    root_dir = '/root/datasets/OpenImages'
    feature_save_dir = os.path.join(root_dir, 'val_features_jw')
    
    
    ## Feature Extraction and Save with h5py
    fn = os.path.join(feature_save_dir, 'OPENIMAGES_VAL_CONV5_4_NO_CENTERCROP.h5')
    
    ## Load Saved Feature
    saved_features = h5py.File(fn, 'r')
    feature_keys = list(saved_features.keys())
    image_names = np.unique(np.array([m.split('-')[0] for m in feature_keys]))
    print(len(image_names))
    
    original_features = h5py.File('/root/datasets/OpenImages/val_features/OPENIMAGES_VAL_CONV5_4_NO_CENTERCROP.h5')
    
    # np.float32(saved_features.get('ffff21932da3ed01.jpg-features')) == np.float32(original_features.get('ffff21932da3ed01.jpg-features'))
    
    # a = np.around(np.float32(saved_features.get('ffff21932da3ed01.jpg-features')), 3)
    # b = np.around(np.float32(original_features.get('ffff21932da3ed01.jpg-features')), 3)
    
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         print(f'{a[i,j]} | {b[i,j]}')
    
    # a = np.array(saved_features['ffff21932da3ed01.jpg-seenlabels'])
    # b = np.array(saved_features['ffff21932da3ed01.jpg-seenlabels'])
    x = []
    for i in image_names:
        a = np.array(saved_features[f'{i}-seenlabels'])
        b = np.array(original_features[f'{i}-seenlabels'])
        # print(np.all(a==b))
        x.append(a == b)