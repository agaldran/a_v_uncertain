import os
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset


class ArteryVein(Dataset):
    """
        The RITE dataset contains 20 training and 20 test retinal images .png.
        We have further splitted it into 15 training and 5 validation images.
        RITE contains ground-truth for artery/vein classification.
    """

    def __init__(self, root_dir='../data/RITE_prepared/train/', im_dir='images/',
                 gt_dir='av/', fov_dir='masks/', od_dir='od/', transforms=None):
        """
        Arguments:
        @param root_dir: path to the folder that contains the train/val/test folder
        @param im_dir: subfolder containing the RGB retinal images
        @param gt_dir: subfolder containing the ground-truth (artery/vein)
        @param fov_dir: subfolder containing the FOV masks
        @param od_dir: subfolder containing OD segmentations
        @transforms: triplet transformations on the data
        """

        # initialize variables
        self.root_dir = root_dir
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.fov_dir = fov_dir
        self.od_dir = od_dir
        self.transforms = transforms

        self.im_names = sorted(os.listdir(self.root_dir + im_dir))
        self.gt_names = sorted(os.listdir(self.root_dir + gt_dir))
        self.fov_names = sorted(os.listdir(self.root_dir + fov_dir))
        self.od_names = sorted(os.listdir(self.root_dir + od_dir))

    def __getitem__(self, index):
        """
        Arguments:
        @param index:
        @return: tuple (img, target, mask) with the input data, its labels, and Region of Interest
        """
        im_name = os.path.join(self.root_dir, self.im_dir, self.im_names[index])
        gt_name = os.path.join(self.root_dir, self.gt_dir, self.gt_names[index])
        fov_name = os.path.join(self.root_dir, self.fov_dir, self.fov_names[index])
        od_name = os.path.join(self.root_dir, self.od_dir, self.od_names[index])

        im = Image.open(im_name)  # PIL image in [0,255], 3 channels
        gt = Image.open(gt_name)  # PIL image in [0,255], 1/3 channel/s

        gt_np = np.array(gt)
        gt_indexes = np.zeros(gt_np.shape[:-1], dtype = int)
        gt_indexes[np.where(gt_np[:,:,0])]=128
        gt_indexes[np.where(gt_np[:,:,2])]=255
        gt = Image.fromarray(gt_indexes.astype('uint8'))

        fov = Image.open(fov_name)  # PIL image in [0,255], 1 channel
        od = Image.open(od_name)  # PIL image in [0,255], 1 channel

        # Build Region of Interest
        roi = fov.copy()

        # If transforms are provided
        if self.transforms is not None:
            im, gt, roi = self.transforms(im, gt, roi)
        else:
            print('Warning: At least you should provide a ToTensor() if you want to train with this Dataset')

        return im_name, im, gt, roi

    def __len__(self):
        """
        @return: number of elements in the dataset
        """
        return len(self.im_names)


class VesselsBack(Dataset):
    """
        The RITE dataset contains 20 training and 20 test retinal images .png. 
        We have further splitted it into 15 training and 5 validation images.
        RITE contains ground-truth for artery/vein classification.
    """

    def __init__(self, root_dir='../data/RITE_prepared/train/', im_dir = 'images/', 
                 gt_dir = 'av/', fov_dir = 'masks/', transforms = None):
        """
        Arguments:
        @param root_dir: path to the folder that contains the train/val/test folder
        @param im_dir: subfolder containing the RGB retinal images
        @param gt_dir: subfolder containing the ground-truth (artery/vein)
        @param fov_dir: subfolder containing the FOV masks
        @transforms: triplet transformations on the data
        """
        
        # initialize variables
        self.root_dir = root_dir
        self.im_dir = im_dir
        self.gt_dir = gt_dir        
        self.fov_dir = fov_dir
        
        self.transforms = transforms
        
        self.im_names = sorted(os.listdir(self.root_dir + im_dir))
        self.gt_names = sorted(os.listdir(self.root_dir + gt_dir))
        self.fov_names = sorted(os.listdir(self.root_dir + fov_dir))



    def __getitem__(self, index):
        """
        Arguments:
        @param index:
        @return: tuple (img, target) with the input data and its labels
        """
        im_name = os.path.join(self.root_dir, self.im_dir, self.im_names[index])
        gt_name = os.path.join(self.root_dir, self.gt_dir, self.gt_names[index])
        fov_name = os.path.join(self.root_dir, self.fov_dir, self.fov_names[index])
                
        im = Image.open(im_name) # PIL image in [0,255], 3 channels
        vessels = Image.open(gt_name) # PIL image in [0,255], 1/3 channel/s
        fov = Image.open(fov_name)
    
        gt_np = np.array(fov)
        vessels_np = np.array(vessels)
        vessels_indexes = np.zeros(vessels_np.shape[:-1], dtype = int)
        vessels_indexes[np.where(vessels_np[:,:,0])]=1
        vessels_indexes[np.where(vessels_np[:,:,2])]=1
        
        gt_np[vessels_indexes==1]=125

        gt = Image.fromarray(gt_np.astype('uint8'))


        
        # If transforms are provided
        if self.transforms is not None:
            im, gt = self.transforms(im, gt)
        else:
            print('Warning: At least you should provide a ToTensor() if you want to train with this Dataset')
            
        return im, gt

    def __len__(self):
        """
        @return: number of elements in the dataset
        """
        return len(self.im_names)
