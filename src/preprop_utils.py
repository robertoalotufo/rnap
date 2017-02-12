from __future__ import print_function

import os
import cv2
import numpy as np
import nibabel as nib
import glob
import os,time
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from natsort import natsorted
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

out = '/home/adessowiki/Development/Oeslle/rnpi-notebooks'
#nifiti_path = os.path.join(out,'CC359/Original/*.nii.gz')
#staple_path = os.path.join(out,'CC359/STAPLE')
nifiti_path = os.path.join(out,'Original-Nifti/*/*/*.nii.gz')
staple_path = os.path.join(out,'STAPLE8')

nifiti_imgs = np.sort(glob.glob(nifiti_path))


n_base = 250 # 250 images for training, 109 images for test
max_patches = 100
n_imgs = 4
t = 0.9

# Configurações da u-net
img_rows = 64
img_cols = 80
patch_size = (img_rows,img_cols) # unet input size 

def create_data(folder):

    data_path = os.path.join(out,folder)  
    images = natsorted(os.listdir(data_path))
    total = len(images) / 2
    imgs = np.ndarray((total, 1, patch_size[0], patch_size[1]), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, patch_size[0], patch_size[1]), dtype=np.uint8)

    i = 0

    for image_name in images:
        if 'staplepatch' in image_name:
            continue
        image_mask_name = image_name.split('-')[0] + '-staplepatch.tif'

        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Data created.')

    return imgs, imgs_mask

def save_data(imgs, imgs_mask, imgs_name, imgs_mask_name):

    np.save(os.path.join(out,imgs_name), imgs)
    np.save(os.path.join(out,imgs_mask_name), imgs_mask)
    
    print('Saved data to .npy.')
    
def get_mid_sample(img_path):

    name = img_path.split('/')[-1].split('.')[0]
    staple_img = os.path.join(staple_path,name + '_staple.nii.gz') 

    data = nib.load(img_path).get_data()
    staple_data = nib.load(staple_img).get_data() > t

    H,W,Z = data.shape
    Hc,Wc,Zc = H/2,W/2,Z/2

    sag = data[Hc,:,:]
    staple = staple_data[Hc,:,:]

    return sag, staple, name, staple_img

def save_test_images(out_path):
    
    for img in nifiti_imgs[n_base:]: 
        sag, staple, name, _ = get_mid_sample(img)
        save_2d_slices(out_path,name,sag,staple)

def save_2d_slices(dst_path,name,img,mask):

    cv2.imwrite(os.path.join(dst_path,name + '_staple.tif'),mask.astype(np.uint8)*255)
    scipy.misc.imsave(os.path.join(dst_path,name +'.tif'),img)
    
def save_2d_samples(dst_path,name,img,mask):

    for i in range(img.shape[0]):
        cv2.imwrite(os.path.join(dst_path,name + '_' + str(i+1) + '-staplepatch.tif'),
                    mask[i].astype(np.uint8)*255)
        scipy.misc.imsave(os.path.join(dst_path,name + '_' + str(i+1) +'-sagpatch.tif'),img[i])
    
def sample_2d_patches(folder, opt):

    if (opt == 'train'):
        print ('Sampling Train Images ...')
       
        for img in nifiti_imgs[:n_base]: 

            sag, staple,name,_ = get_mid_sample(img)           
            sag_patches = extract_patches_2d(sag, patch_size, max_patches, random_state = 1)
            staple_patches = extract_patches_2d(staple, patch_size, max_patches, random_state = 1)
            
            print ('Saving image:', name)

            save_2d_samples(out + '/train_patches', name, sag_patches,staple_patches)


    if (opt == 'test'):
        print ('Sampling Test Images ...')

        for img in nifiti_imgs[n_base:n_base+n_imgs]: 

            sag,staple,name,_ = get_mid_sample(img)

            sag_patches = extract_patches_2d(sag, patch_size)
            staple_patches = extract_patches_2d(staple, patch_size)

            print ('Saving image:', name)
            
            dirPath = os.path.join(out,'test_patches')
            dirPath = os.path.join(dirPath,name)
 
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            
            save_2d_samples(dirPath, name, sag_patches,staple_patches)

            sag_patches = 0
            staple_patches = 0
            
def preprocess(imgs):
    
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def get_mean_std_train(imgs,mask):

    imgs_train, imgs_mask_train = load_train_data(imgs,mask)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    return mean,std

def read_prep_test(src_path,mean,std):
        
    #imgs_test = np.load(src_path) 
    #imgs_test = preprocess(imgs_test)
    
    imgs_test = preprocess(src_path)
    
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    return imgs_test

def load_train_data(imgs,mask):
    
    imgs_train = np.load(imgs)
    imgs_mask_train = np.load(mask)
    return imgs_train, imgs_mask_train

def reconstruct_2d_sample(dst_path, data_path, root_folder):

    pred_imgs_maks_test = np.load(data_path)
    n_patches = pred_imgs_maks_test.shape[0]

    pred_imgs_maks_test = np.reshape(pred_imgs_maks_test,(n_patches,patch_size[0],patch_size[1]))

    pred_imgs_maks_test[pred_imgs_maks_test >= 0.5] = 1
    pred_imgs_maks_test[pred_imgs_maks_test < 0.5] = 0

    name = data_path.split('/')[-1].split('-')[0]
    orig_mask = cv2.imread(os.path.join(root_folder,name + '_staple.tif'), cv2.IMREAD_GRAYSCALE)
    orig_size = orig_mask.shape
     
    del(orig_mask)
    rec_image_mask = reconstruct_from_patches_2d(pred_imgs_maks_test,orig_size)
    cv2.imwrite(os.path.join(dst_path,name + '-pred.tif'),rec_image_mask.astype(np.uint8)*255)