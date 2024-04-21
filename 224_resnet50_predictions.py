"""
Prediction-generating script for trained ResNet models, reports relevant metrics
"""


import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, ToTensor, Resize, ColorJitter, RandomAutocontrast, RandomInvert, RandomRotation, Compose
from torchvision.transforms.functional import rotate
import torch.nn as nn
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import time
from PIL import Image


device = torch.device("cuda")
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))


train_path = "/home/ubuntu/project/data/data_train/"
val_path = "/home/ubuntu/project/data/data_val/"
test_path = "/home/ubuntu/project/data/data_test/"

img_height = 224
img_width = 224

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.35205421, 0.37928827, 0.34679396], std=[0.19838193, 0.17464092, 0.17116936]),
])

def try_load(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
        return None
    
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)    
test_dataset = datasets.ImageFolder(root=test_path, transform=transform, loader=try_load)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

class_sample_count = np.unique(train_dataset.targets, return_counts=True)[1]
pos_weight = torch.tensor([class_sample_count[0] / class_sample_count[1]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
learn_rate = 1e-3

def predictions_stats(res_num, dataset_name, split, best_stat):
    # res_num: '18', '50'
    # dataset: 'tilewise', 'basinwise'
    # split: 'train', 'val', 'test'
    # best_stat: 'loss', 'prc'
    
    print(f'Predicting with best {best_stat} ResNet-{res_num} on {dataset_name} {split} set (Normalized)')
    
    if split == 'train':
        dataset = train_dataset
        loader = train_loader
    elif split == 'val':
        dataset = val_dataset
        loader = val_loader
    elif split == 'test':
        dataset = test_dataset
        loader = test_loader
    else:
        return

    # Load model
    cp_filepath = f"/home/ubuntu/project/224_pt_saves/best_{best_stat}_resnet{res_num}_model_224_norm.pth"

    checkpoint = torch.load(cp_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Switch to eval mode
    model.eval()

    predictions, actuals, filenames, probabilities = [], [], [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            if idx % 5000 == 0:
                print(f'predicting on {split} set, example ', idx)
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            
            predictions.extend([preds.item()])
            actuals.extend([labels.item()])
            filenames.extend([dataset.samples[idx][0]])
            probabilities.extend([torch.sigmoid(outputs).item()])

    # Save predictions
    df = pd.DataFrame({'filename': filenames, 'predictions': predictions, 'actuals': actuals, 'probability': probabilities})
    df.to_csv(f'224_preds/resnet_{res_num}_{best_stat}_{dataset_name}_{split}_predictions_norm.csv', index=False)
    print(f'Saved csv at: 224_preds/resnet_{res_num}_{best_stat}_{dataset_name}_{split}_predictions_norm.csv')

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    roc_auc = roc_auc_score(actuals, predictions)
    pr_auc = average_precision_score(actuals, predictions)

    print(f'{split.capitalize()} Accuracy: {accuracy:.4f}')
    print(f'{split.capitalize()} Precision: {precision:.4f}')
    print(f'{split.capitalize()} Recall: {recall:.4f}')
    print(f'{split.capitalize()} F1-Score: {f1:.4f}')
    print(f'{split.capitalize()} AUC-ROC: {roc_auc:.4f}')
    print(f'{split.capitalize()} AUC-PR: {pr_auc:.4f}')

# RESNET-50

model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
optimizer = Adam(model.parameters(), lr=learn_rate)

model = model.to(device)

predictions_stats(res_num=50, dataset_name='224', split='test', best_stat='prc')
predictions_stats(res_num=50, dataset_name='224', split='test', best_stat='loss')



predictions_stats(res_num=50, dataset_name='224', split='train', best_stat='prc')
predictions_stats(res_num=50, dataset_name='224', split='val', best_stat='prc')
predictions_stats(res_num=50, dataset_name='224', split='train', best_stat='loss')
predictions_stats(res_num=50, dataset_name='224', split='val', best_stat='loss')