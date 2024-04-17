import cv2
import h5py
import os
from random import shuffle
import numpy as np

#img_dir = '/data/i5O/nyudepthv2/train/'
#img_folders = os.listdir(img_dir)
#h5_list = []

#for folder in img_folders:
#    for h5_file in os.listdir(img_dir+folder):
#        h5_list.append(img_dir+folder+'/'+h5_file)
        
#print(h5_list)		
#shuffle(h5_list)

#i = 0
#for file in h5_list:
#    i += 1
#    with h5py.File(file, "r") as f:
#        depth = f['depth']
#        depth = (depth[:]).astype('float')
#        np.save('/data/i5O/nyudepthv2_data/train/depth/'+file.split('/')[-2]+'_'+file.split('/')[-1].replace('.h5','.npy'), depth)
		
#        img = np.transpose(f['rgb'], (1, 2, 0))
#        img = np.array(img, dtype = np.uint8)
#        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#        cv2.imwrite('/data/i5O/nyudepthv2_data/train/image/'+file.split('/')[-2]+'_'+file.split('/')[-1].replace('.h5','.jpg'), img)
    
#    if i > 20000:
#        break

img_dir_test = '/data/i5O/nyudepthv2/val/official/'
img_folders_test = os.listdir(img_dir_test)
h5_list_test = []

for folder in img_folders_test:
    #for h5_file in os.listdir(img_dir_test+folder):
    h5_list_test.append(img_dir_test+folder+'/'+h5_file)
        
#print(h5_list_test)		
shuffle(h5_list_test)

j = 0
for file in h5_list_test:
    j += 1
    with h5py.File(file, "r") as f:
        depth = f['depth']
        depth = (depth[:]).astype('float')
        np.save('/data/i5O/nyudepthv2_data/val/depth/'+file.split('/')[-1].replace('.h5','.npy'), depth)
		
        img = np.transpose(f['rgb'], (1, 2, 0))
        img = np.array(img, dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/data/i5O/nyudepthv2_data/val/image/'+file.split('/')[-1].replace('.h5','.jpg'), img)

		

		
