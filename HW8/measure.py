# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

from rare_traffic_sign_solution import *

# !Этих импортов достаточно для решения данного задания


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CustomNetwork()
model.load_state_dict(torch.load("simple_model_augms_low_prob_3epochs.pth"))

# here = os.path.dirname(os.path.realpath(__file__))
#     train_dataset = DatasetRTSD(
#         root_folders=[f"{here}/cropped-train"],
#         path_to_classes_json=f"{here}/classes.json",
#     )

print(
    test_classifier(
        model,
        'smalltest',
        'smalltest_annotations.csv'
    )
)