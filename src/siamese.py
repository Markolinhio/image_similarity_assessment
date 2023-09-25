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


def generate_train_test_set(product_path, test_ratio=0.2):
    good_path = os.path.join(product_path, 'train')
    defect_path = os.path.join(product_path, 'test')

    good_image_list = [os.path.abspath(os.path.join(good_path, image_path)) for image_path in os.listdir(good_path)]
    train_good_image_list, test_good_image_list, _, _ = train_test_split(good_image_list, [1]*len(good_image_list),
                                                                   test_size=test_ratio, random_state=205)

    all_train_defect_image_list = []
    all_test_defect_image_list = []
    for case in os.listdir(defect_path):
        case_path = os.path.abspath(os.path.join(defect_path, case))
        defect_image_list = [os.path.abspath(os.path.join(case_path, image_path)) for image_path in os.listdir(case_path)]
        train_defect_image_list, test_defect_image_list, _, _ = train_test_split(defect_image_list, [0]*len(defect_image_list),
                                                                   test_size=test_ratio, random_state=205)
        # print(len(train_defect_image_list), len(test_defect_image_list))
        all_train_defect_image_list += train_defect_image_list
        all_test_defect_image_list += test_defect_image_list
    # print(len(all_train_defect_image_list), len(all_test_defect_image_list))

    return train_good_image_list, test_good_image_list, all_train_defect_image_list, all_test_defect_image_list  


class ProductDataset(Dataset):
    def __init__(self, image_label_dict, transform=None):
        self.image_label_dict = image_label_dict
        self.transform = transform
        
    def __getitem__(self,index):
        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        image_path_1 = random.choice(list(image_label_dict.keys()))
        label_1 = self.image_label_dict[image_path_0]
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

        image_1 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image_2 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        return image_1, image_2, torch.from_numpy(np.array([int(label_1 != label_2)], dtype=np.float32))
    
    def __len__(self):
        return len(self.image_label_dict.keys())


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2