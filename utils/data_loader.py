import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from utils import data_handling as dh
from torchvision import transforms
import nibabel as nib
import numpy as np
import os, sys
import random
import torch


class train(Dataset):
    """ Class used to load the images for training, performing augmentations and data handling
        based on the configuration """

    def __init__(self, niftiPaths, niftiPathsGT, sliceIdxs, config):
        """Initialization"""
        self.niftiPaths = niftiPaths
        self.niftiPathsGT = niftiPathsGT
        self.sliceIdxs = sliceIdxs
        self.config = config
        
    def __getitem__(self, index):
        """Generates one sample of data"""
        niftiPath = self.niftiPaths[index]
        gtPath = self.niftiPathsGT[index]
        sliceIdx = self.sliceIdxs[index]
        
        #Load nifti volume
        volume = np.flip(nib.load(niftiPath).get_fdata(),0)
        if self.config["plane"] == 0: img = np.float32(volume[sliceIdx,:,:].T[::-1])
        if self.config["plane"] == 1: img = np.float32(volume[:,sliceIdx,:].T[::-1])
        if self.config["plane"] == 2: img = np.float32(volume[:,:,sliceIdx].T[::-1])

        #Load nifti annotation volume
        gt = np.flip(nib.load(gtPath).get_fdata(),0)
        if self.config["plane"] == 0: gtImg = np.float32(gt[sliceIdx,:,:].T[::-1])
        if self.config["plane"] == 1: gtImg = np.float32(gt[:,sliceIdx,:].T[::-1])
        if self.config["plane"] == 2: gtImg = np.float32(gt[:,:,sliceIdx].T[::-1])

        img, gtImg = self.pairedTransformations(img, gtImg, self.config)

        if self.config["heads"]:
            if "T1" in gtPath: 
                headImg = torch.zeros(img.shape)
                img = torch.cat((img, headImg))

            if "T2" in gtPath:
                headImg = torch.zeros(img.shape)
                img = torch.cat((headImg, img))

        return img, gtImg

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.niftiPaths)    

    def pairedTransformations(self, img, gtImg, config):

        # Convert to PIL image
        img = TF.to_pil_image(img)
        gtImg = TF.to_pil_image(gtImg)
        
        # Resize images
        img = TF.resize(img, size=(config["imgSize"], config["imgSize"]), interpolation=2)
        gtImg = TF.resize(gtImg, size=(config["imgSize"], config["imgSize"]), interpolation=0)

        # Rotate images
        if config["rotate"] & random.choice([True, False]):
            angle = random.randint(-10, 10)
            img = TF.rotate(img, angle)
            gtImg = TF.rotate(gtImg, angle)

        # Randomly crop images
        if config["crop"] & random.choice([True, False]):
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1), ratio=(0.75, 1))
            img = TF.resized_crop(img, i, j, h, w, size=(config["imgSize"], config["imgSize"]), interpolation=2)
            gtImg = TF.resized_crop(gtImg, i, j, h, w, size=(config["imgSize"], config["imgSize"]), interpolation=0)

        img = TF.to_tensor(img)
        if config["normalize"]: img = TF.normalize(img, mean=[config["normVals"][0]], std=[config["normVals"][1]])
        gtImg = torch.from_numpy(np.expand_dims(np.array(gtImg), 0))

        return img, gtImg


class validate(Dataset):
    """ Class used to load the images for validation """

    def __init__(self, niftiPaths, niftiPathsGT, sliceIdxs, config):
        """Initialization"""
        self.niftiPaths = niftiPaths
        self.niftiPathsGT = niftiPathsGT
        self.sliceIdxs = sliceIdxs
        self.config = config
        
    def __getitem__(self, index):
        """Generates one sample of data"""
        niftiPath = self.niftiPaths[index]
        gtPath = self.niftiPathsGT[index]
        sliceIdx = self.sliceIdxs[index]
        
        #Load nifti volume
        volume = np.flip(nib.load(niftiPath).get_fdata(),0)
        if self.config["plane"] == 0: img = np.float32(volume[sliceIdx,:,:].T[::-1])
        if self.config["plane"] == 1: img = np.float32(volume[:,sliceIdx,:].T[::-1])
        if self.config["plane"] == 2: img = np.float32(volume[:,:,sliceIdx].T[::-1])

        #Load nifti annotation volume
        gt = np.flip(nib.load(gtPath).get_fdata(),0)
        if self.config["plane"] == 0: gtImg = np.float32(gt[sliceIdx,:,:].T[::-1])
        if self.config["plane"] == 1: gtImg = np.float32(gt[:,sliceIdx,:].T[::-1])
        if self.config["plane"] == 2: gtImg = np.float32(gt[:,:,sliceIdx].T[::-1])

        img, gtImg = self.pairedTransformations(img, gtImg, self.config)

        if self.config["heads"]:
            if "T1" in gtPath: 
                headImg = torch.zeros(img.shape)
                img = torch.cat((img, headImg))

            if "T2" in gtPath:
                headImg = torch.zeros(img.shape)
                img = torch.cat((headImg, img))        

        numberIdx = [(i,c) for i,c in enumerate(niftiPath) if c.isdigit()]
        imgPath = os.path.join(self.config["modelDir"], "Outputs",niftiPath[numberIdx[0][0]:-7] + "_"+ str(sliceIdx) +".png")

        return img, gtImg, imgPath

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.niftiPaths)    

    def pairedTransformations(self, img, gtImg, config):

        # Convert to PIL image
        img = TF.to_pil_image(img)
        gtImg = TF.to_pil_image(gtImg)
        
        # Resize images
        img = TF.resize(img, size=(config["imgSize"], config["imgSize"]), interpolation=2)
        gtImg = TF.resize(gtImg, size=(config["imgSize"], config["imgSize"]), interpolation=0)

        img = TF.to_tensor(img)
        if config["normalize"]: img = TF.normalize(img, mean=[config["normVals"][0]], std=[config["normVals"][1]])
        gtImg = torch.from_numpy(np.expand_dims(np.array(gtImg), 0))

        return img, gtImg


class inference(Dataset):
    """ Class used to set perform inference on test images without annotations """

    def __init__(self, niftiPaths, sliceIdxs, config):
        """Initialization"""
        self.niftiPaths = niftiPaths
        self.sliceIdxs = sliceIdxs
        self.config = config
        
    def __getitem__(self, index):
        """Generates one sample of data"""
        niftiPath = self.niftiPaths[index]
        sliceIdx = self.sliceIdxs[index]
        
        #Load nifti volume
        volume = np.flip(nib.load(niftiPath).get_fdata(),0)
        if self.config["plane"] == 0: img = np.float32(volume[sliceIdx,:,:].T[::-1])
        if self.config["plane"] == 1: img = np.float32(volume[:,sliceIdx,:].T[::-1])
        if self.config["plane"] == 2: img = np.float32(volume[:,:,sliceIdx].T[::-1])

        img = self.pairedTransformations(img, self.config)

        numberIdx = [(i,c) for i,c in enumerate(niftiPath) if c.isdigit()]
        imgPath = os.path.join(self.config["outputFolder"], niftiPath[numberIdx[0][0]:-7] + "_"+ str(sliceIdx) +".png")

        return img, imgPath

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.niftiPaths)    

    def pairedTransformations(self, img, config):

        # Convert to PIL image
        img = TF.to_pil_image(img)
        
        # Resize images
        img = TF.resize(img, size=(config["imgSize"], config["imgSize"]), interpolation=2)
        img = TF.to_tensor(img)

        return img