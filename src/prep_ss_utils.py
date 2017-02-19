
# coding: utf-8

# Funções necessárias para o pré-processsamento das imagens antes de inseri-lás na CNN U-NET.

# In[2]:

import os
import cv2
import numpy as np
import glob
import os,time
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from natsort import natsorted
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

n_base = 250 # 250 imagens para o treinamento
max_patches = 100 # número máximo de patches em cada imagem utilizada no treino.
n_imgs = 4 # número de imagens para o teste.

# Configuracoes da u-net
img_rows = 64
img_cols = 80
patch_size = (img_rows,img_cols) # unet input size 

# Criação dos dados em formato de array .py  a partir dos patches extraídos.
def create_data(folder):

    images = natsorted(os.listdir(folder))
    total = len(images) / 2
    imgs = np.ndarray((total, 1, patch_size[0], patch_size[1]), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, patch_size[0], patch_size[1]), dtype=np.uint8)

    i = 0

    for image_name in images:
        if 'staplepatch' in image_name:
            continue
        image_mask_name = image_name.split('-')[0] + '-staplepatch.tif'

        img = cv2.imread(os.path.join(folder, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(folder, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Data created.')

    return imgs, imgs_mask

# salva array contendo os patches no formato .npy
def save_data(imgs, imgs_mask, imgs_name, imgs_mask_name, out):

    np.save(os.path.join(out,imgs_name), imgs)
    np.save(os.path.join(out,imgs_mask_name), imgs_mask)
    
    print('Saved data to .npy.')

# salva imagens no formata .tif
def save_2d_samples(dst_path,name,img,mask):

    for i in range(img.shape[0]):
        cv2.imwrite(os.path.join(dst_path,name + '_' + str(i+1) + '-staplepatch.tif'),
                    mask[i].astype(np.uint8)*255)
        scipy.misc.imsave(os.path.join(dst_path,name + '_' + str(i+1) +'-sagpatch.tif'),img[i])

# Extrai patches a partir de imagens 2D. A extração dos patches é realizada com a função do sklearn 'extract_patches_2d' 
# (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html)
def sample_2d_patches(srcDir, dstDir, opt):
    
    mid_sag_samples = np.sort(glob.glob(os.path.join(srcDir, '*_staple.tif')))
    
    if (opt == 'train'):
        print ('Sampling Train Images ...')
       
        for img in mid_sag_samples:
            name = img.split('/')[-1].split('_staple.tif')[0]
            
            sag = cv2.imread(os.path.join(srcDir, name + '.tif'), cv2.IMREAD_GRAYSCALE)
            staple = cv2.imread(os.path.join(srcDir, name + '_staple.tif'), cv2.IMREAD_GRAYSCALE)
                      
            sag_patches = extract_patches_2d(sag, patch_size, max_patches, random_state = 1)
            staple_patches = extract_patches_2d(staple, patch_size, max_patches, random_state = 1)
            
            print ('Saving train image patches:', name)
            save_2d_samples(dstDir, name, sag_patches,staple_patches)
            
    if (opt == 'test'):
        print ('Sampling Test Images ...')

        for img in mid_sag_samples[:n_imgs]:
            name = img.split('/')[-1].split('_staple.tif')[0]
            
            sag = cv2.imread(os.path.join(srcDir, name + '.tif'), cv2.IMREAD_GRAYSCALE)
            staple = cv2.imread(os.path.join(srcDir, name + '_staple.tif'), cv2.IMREAD_GRAYSCALE)
                      
            sag_patches = extract_patches_2d(sag, patch_size)
            staple_patches = extract_patches_2d(staple, patch_size)

            print ('Saving test image patches:', name)           
            out = os.path.join(dstDir,name)
            print (out)
            
            if not os.path.exists(out):
                os.makedirs(out)
            
            save_2d_samples(out, name, sag_patches,staple_patches)

# Faz resize das imagens para o tamanho de entrada da U-NET
def preprocess(imgs):
    
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

# Extrai a média e o desvio padrão dos dados de treino para normalizar o teste
def get_mean_std_train(imgs,mask):

    imgs_train, imgs_mask_train = load_train_data(imgs,mask)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    return mean,std

# Ler os dados de teste e faz o resize mais a normalização
def read_prep_test(img,mean,std):
        
    imgs_test = preprocess(img)
    
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    return imgs_test

# Carrega os dados de treino em formato .py 
def load_train_data(imgs,mask):
    
    imgs_train = np.load(imgs)
    imgs_mask_train = np.load(mask)
    
    return imgs_train, imgs_mask_train

# Reconstroi as imagens a partir dos patches preditos da saída da CNN. A função utilizada par aisso é do sklean
# 'reconstruct_from_patches_2d' 
# (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html)
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


# In[3]:

get_ipython().system(u'ipython nbconvert prep_ss_utils.py')


# In[ ]:



