"""""
Load related header files
"""""
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
import math
"""""
Define loading face image data
"""""
class Face_Dataset(Dataset):
    def __init__(self, imageFolderDataset, should_transform=True):
        self.imageFolderDataset = imageFolderDataset
        self.should_transform = should_transform
        self.transform = transforms.Compose([transforms.Resize((100,100)),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])
                                             ])

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

############# Read image path and convert to gray and white image ###########
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        mean_0 = np.mean(img0)
        gamma_val_0 = math.log10(0.5) / math.log10(mean_0 / 255)  # Formula for calculating gamma
        img0 = np.array(img0)
        fI = img0 / 255.0
        Oc = np.power(fI, gamma_val_0)
        data = Oc * 255.0
        img0 = Image.fromarray(data.astype('uint8')).convert('L')

        mean_1 = np.mean(img1)
        gamma_val_1 = math.log10(0.5) / math.log10(mean_1 / 255)  # Formula for calculating gamma
        img1 = np.array(img1)
        fI = img1 / 255.0
        Oc = np.power(fI, gamma_val_1)
        data = Oc * 255.0
        img1 = Image.fromarray(data.astype('uint8')).convert('L')
############################### END #########################################

######################Perform related image filtering####################
        img0 = img0.filter(ImageFilter.DETAIL)
        img0 = img0.filter(ImageFilter. MedianFilter(size=3))
        img1 = img1.filter(ImageFilter.DETAIL)
        img1 = img1.filter(ImageFilter.MedianFilter(size=3))
############################### END  ####################################

        if self.should_transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        else:
            img0=img0
            img1=img1

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)