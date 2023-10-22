import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_similarity_measures import quality_metrics as qm
from tqdm import tqdm
from utils import *
import time

def table_of_score(model_image_dict, image_dict):
    model_image_path, model_label = list(model_image_dict.items())[0]
    model_image = cv2.imread(model_image_path, cv2.IMREAD_COLOR)
    score_list = []
    metrics = ['rmse', 'psnr', 'fsim', 'ssim', 'uiq', 'sam', 'sre']
    
    for image_path, label in tqdm(image_dict.items()):
        #start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        scores = []
        for metric in metrics:
            try:
                metric_function = qm.metric_functions[metric]
                score = metric_function(model_image, image)
                scores.append(score)
            except:
                scores.append(np.nan)
        
        if model_label != label:
            scores.append(0.)
        else:
            scores.append(1.)
            
        score_list.append(np.array(scores))
        #end = time.time()
        #print("Time elapsed:", end - start,"sec\n")
    score_list = np.array(score_list)
    return score_list