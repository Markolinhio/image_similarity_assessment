import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class ProductDataset(Dataset):
    def __init__(self, image_label_dict, transform=None):
        self.image_label_dict = image_label_dict
        self.transform = transform
        
    def __getitem__(self,index):
        if len(list(self.image_label_dict.keys())) == 1:
            print(self.image_label_dict)
            image_path_1 = list(self.image_label_dict.keys())[0]
            image_1 = cv2.imread(image_path_1, cv2.IMREAD_UNCHANGED)
            label_1 = self.image_label_dict[image_path_1]
            if self.transform is not None:
                image_1 = self.transform(image_1)
            return image_1, torch.empty((0, 0), dtype=torch.float32), torch.from_numpy(np.array([label_1])), torch.empty((0, 0), dtype=torch.float32)
        else:
            #We need to approximately 50% of images to be in the same class
            should_get_same_class = random.randint(0,1) 
            image_path_1 = random.choice(list(self.image_label_dict.keys()))
            label_1 = self.image_label_dict[image_path_1]
            if should_get_same_class:
                same_label_list = [image_path for image_path, label in self.image_label_dict.items()
                                   if image_path != image_path_1 and label == label_1]
                image_path_2 = random.choice(same_label_list) 
                label_2 = self.image_label_dict[image_path_2]
            else:
                opposite_label_list = [image_path for image_path, label in self.image_label_dict.items()
                                   if image_path != image_path_1 and label != label_1]
                image_path_2 = random.choice(opposite_label_list) 
                label_2 = self.image_label_dict[image_path_2]
    
            image_1 = cv2.imread(image_path_1, cv2.IMREAD_UNCHANGED)
            image_2 = cv2.imread(image_path_2, cv2.IMREAD_UNCHANGED)
    
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
            
            return image_1, image_2, torch.from_numpy(np.array([label_1, label_2])), torch.from_numpy(np.array([int(label_1 != label_2)], dtype=np.float32))
    
    def __len__(self):
        return len(self.image_label_dict.keys())


class SiameseNetwork(nn.Module):

    def __init__(self, n_classes):
        super(SiameseNetwork, self).__init__()

        self.n_classes = n_classes
        # Setting up model with resnet18 backbone
        resnet = torchvision.models.resnet18()
        # Modify the top linear layer
        resnet.fc = nn.Linear(512, 1024)

        # Make a 
        self.cnn = nn.Sequential(resnet,
                      nn.ReLU(True),
                      nn.Linear(1024, 256),
                      nn.ReLU(True),
                      nn.Linear(256, self.n_classes))
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
    
    
        return loss_contrastive
