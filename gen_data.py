import cv2
import h5py
import os
from random import shuffle
import numpy as np

img_dir = 'C:/Hatem/nyudepthv2/val/'
img_folders = os.listdir(img_dir)
h5_list = []

for folder in img_folders:
    for h5_file in os.listdir(img_dir+folder):
        h5_list.append(img_dir+folder+'/'+h5_file)
        
print(h5_list)		
shuffle(h5_list)

i = 0
for file in h5_list:
    i += 1
    with h5py.File(file, "r") as f:
        depth = f['depth']
        depth = (depth[:]).astype('float')
        np.save('C:/Hatem/nyudepthv2/img_dataset/test/depth/'+file.split('/')[-1].replace('.h5','.npy'), depth)
		
        img = np.transpose(f['rgb'], (1, 2, 0))
        img = np.array(img, dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('C:/Hatem/nyudepthv2/img_dataset/test/image/'+'_'+file.split('/')[-1].replace('.h5','.jpg'), img)
    
    if i > 20000:
        break

		

		
