

# Torch imports
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Other package imports
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import utils



class CustomSmokeDataset(Dataset):
    def __init__(self, 
                 image_data_path=None,
                 prev_image_data_path=None,
                 labels_path=None, 
                 optical_flow_path=None,                 
                 image_dimensions = (480,640),
                 resize_dimensions = (480,640),
                 n_tiles_size=None):
    
        self.image_data_path = image_data_path
        self.prev_image_data_path = prev_image_data_path
        self.labels_path = labels_path
        self.optical_flow_path = optical_flow_path
        self.image_dimensions = image_dimensions
        self.resize_dimensions = resize_dimensions

        self.filenames = os.listdir(self.image_data_path)
        
        self.num_tiles_height, self.num_tiles_width = n_tiles_size
        self.tiles_dimensions = (int((resize_dimensions[0])/(self.num_tiles_height)),int((resize_dimensions[1])/(self.num_tiles_width)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
    
        ### Load Images ###
        x = []
        # print(image_name)
 
        if Path(self.prev_image_data_path+'/'+image_name).exists():
            img = cv2.imread(self.image_data_path+'/'+image_name)
            original_shape = img.shape
            img=cv2.resize(img,(self.resize_dimensions[1],self.resize_dimensions[0]))
            img=img/255
            img = utils.tile_image(img,self.num_tiles_height,self.num_tiles_width,self.resize_dimensions,self.tiles_dimensions,0)
            x.append(img)
        else:
            print(self.image_data_path+'/'+image_name +" do not exist")
            exit()

        if Path(self.prev_image_data_path+'/'+image_name).exists():
            img = cv2.imread(self.prev_image_data_path+'/'+image_name)
            img=cv2.resize(img,(self.resize_dimensions[1],self.resize_dimensions[0]))
            img=img/255
            img = utils.tile_image(img,self.num_tiles_height,self.num_tiles_width,self.resize_dimensions,self.tiles_dimensions,0)
            x.append(img)
        else:
            print(self.prev_image_data_path+'/'+image_name +" do not exist")
            exit()

        # if Path(self.optical_flow_path+'/'+image_name).exists():
        #     img = cv2.imread(self.optical_flow_path+'/'+image_name)
        #     img=cv2.resize(img,(self.resize_dimensions[1],self.resize_dimensions[0]))
        #     img=img/255
        #     img = utils.tile_image(img,self.num_tiles_height,self.num_tiles_width,self.resize_dimensions,self.tiles_dimensions,0)
        #     x.append(img)
        # else:
        #     print(self.optical_flow_path+'/'+image_name +" do not exist")
        #     exit()
                

        if Path(self.labels_path+'/'+image_name+".txt").exists():
            boxes = []
            with open(self.labels_path+'/'+image_name+".txt",'r') as f:
                lines = f.read().splitlines()
                for line in lines:  
                    boxes.append(np.array(line.split(",")).astype(int))
            tile_label, ground_thuth = utils.tile_labels(boxes,self.num_tiles_height,self.num_tiles_width, original_shape[0:2],0.3)
        else:
            print(self.labels_path+'/'+image_name+".txt")
            exit()

        x=np.array(x)
        x=np.transpose(np.stack(x),(1,0,4,2,3))
        tile_label=np.array(tile_label)
        
        return image_name, x, tile_label,ground_thuth



class SmokeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 image_data_path=None, 
                 prev_image_data_path=None, 
                 labels_path=None, 
                 optical_flow_path=None,

                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,

                 batch_size=1, 
                 num_workers=0, 
                 
                 original_dimensions = (480, 480),
                 resize_dimensions = (480, 480),
                 n_tile_size = None):
        
        super().__init__()
        
     
           
        self.image_data_path = image_data_path
        self.prev_image_data_path = prev_image_data_path
        self.labels_path = labels_path
        self.optical_flow_path = optical_flow_path
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
     
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.n_tile_size=n_tile_size
        
        self.has_setup = False
        
    def setup(self, stage=None, log_dir=None):
        """
        Args:
            - log_dir (str): logging directory to save train/val/test splits 
        """
        if self.has_setup: return
        print("Setting Up Data...")

        
        self.has_setup = True
        print("Setting Up Data Complete.")
    
    
    def train_dataloader(self):
        train_dataset = CustomSmokeDataset(image_data_path=self.train_split_path+'/'+self.image_data_path,
                                           prev_image_data_path=self.train_split_path+'/'+self.prev_image_data_path,
                                          labels_path=self.train_split_path+'/'+self.labels_path, 
                                          optical_flow_path=self.train_split_path+'/'+self.optical_flow_path,
                                          image_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          n_tiles_size=self.n_tile_size
                                          )
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = train_dataset = CustomSmokeDataset(image_data_path=self.val_split_path+'/'+self.image_data_path,
                                           prev_image_data_path=self.val_split_path+'/'+self.prev_image_data_path,
                                          labels_path=self.val_split_path+'/'+self.labels_path, 
                                          optical_flow_path=self.val_split_path+'/'+self.optical_flow_path,
                                          image_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          n_tiles_size=self.n_tile_size
                                          )
        val_loader = DataLoader(val_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True)
        return val_loader

    def test_dataloader(self):
        test_dataset = CustomSmokeDataset(image_data_path=self.test_split_path+'/'+self.image_data_path,
                                           prev_image_data_path=self.test_split_path+'/'+self.prev_image_data_path,
                                          labels_path=self.test_split_path+'/'+self.labels_path, 
                                          optical_flow_path=self.test_split_path+'/'+self.optical_flow_path,
                                          image_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          n_tiles_size=self.n_tile_size
                                          )
        test_loader = DataLoader(test_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True)
        return test_loader

if __name__ == "__main__":
    image_data_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/train/frame"
    prev_image_data_path="data/image_dataset/train/prev_frame"
    labels_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/train/bbox_annotation"
    optical_flow_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/train/opt_flow_frame"

    dataset = CustomSmokeDataset(
        image_data_path=image_data_path,
        prev_image_data_path=prev_image_data_path,
        labels_path=labels_path,
        optical_flow_path=optical_flow_path,
        image_dimensions = (1080,1920),
        n_tiles_size=(10,10))

    print(dataset.__getitem__(3))
    print("Fim")


