"""
Training script for the ResNet models, using 224 x 224 image inputs
"""


import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
import time

# Hyperparameters:
num_epochs = 100
learn_rate = 1e-3

device = torch.device("cuda")
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

train_path = "/home/ubuntu/project/data/data_train/"
val_path = "/home/ubuntu/project/data/data_val/"

# batch_size = 128 * 4
batch_size = 64
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

print('Loading datasets')
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print('Loading pretrained ResNet')
model = models.resnet50(pretrained=True)
model = model.to(device)

class_sample_count = np.unique(train_dataset.targets, return_counts=True)[1]
pos_weight = torch.tensor([class_sample_count[0] / class_sample_count[1]]).to(device)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = Adam(model.parameters(), lr=learn_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# best_metric = 'loss'
best_metric = 'prc'
cp_filepath = f"/home/ubuntu/project/224_pt_saves/best_{best_metric}_resnet50_model_224_norm.pth"

loss_history = []
accuracy_history = []
precision_history = []
recall_history = []
val_loss_history = []
val_accuracy_history = []
val_precision_history = []
val_recall_history = []
val_prc_history = []

try:
    checkpoint = torch.load(cp_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    best_prc = checkpoint['best_prc']
    print('Loaded check point model')
except:
    start_epoch = 0
    best_loss = float('inf')
    best_prc = 0.0


for epoch in range(start_epoch, num_epochs):
    print("Epoch {} running".format(epoch))

    """ Training Phase """
    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        labels = labels.view(-1, 1).float()
        optimizer.zero_grad()
        model = model.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(outputs) > 0.5

        # Calculate metrics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        true_positives += torch.sum((preds == 1) & (labels.data == 1))
        false_positives += torch.sum((preds == 1) & (labels.data == 0))
        false_negatives += torch.sum((preds == 0) & (labels.data == 1))

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Save metrics
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_acc)
    precision_history.append(precision)
    recall_history.append(recall)

    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch} completed in: {epoch_duration:.2f} seconds")
    print(f'[Train #{epoch}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Precision: {precision:.4f} Recall: {recall:.4f}')
    
    """ Validation Phase """
    start_time = time.time()
    model.eval()  # Set model to evaluate mode
    val_running_loss = 0.0
    val_running_corrects = 0
    val_true_positives = 0
    val_false_positives = 0
    val_false_negatives = 0

    with torch.no_grad():
        y_true = []
        y_scores = []
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            labels = labels.view(-1, 1).float()

            outputs = model(inputs)
            y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)

            preds = torch.sigmoid(outputs) > 0.5

            # Calculate metrics
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            val_true_positives += torch.sum((preds == 1) & (labels.data == 1))
            val_false_positives += torch.sum((preds == 1) & (labels.data == 0))
            val_false_negatives += torch.sum((preds == 0) & (labels.data == 1))

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects / len(val_dataset) * 100.0
        val_precision = val_true_positives / (val_true_positives + val_false_positives)
        val_recall = val_true_positives / (val_true_positives + val_false_negatives)

        # Save metrics
        val_loss_history.append(val_epoch_loss)
        val_accuracy_history.append(val_epoch_acc)
        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        curv_precision, curve_recall, _ = precision_recall_curve(y_true, y_scores)
        val_prc = auc(curve_recall, curv_precision)
        
        end_time = time.time()
        validation_duration = end_time - start_time
        print(f"Validation for Epoch {epoch} completed in: {validation_duration:.2f} seconds")
        print(f'[Val #{epoch}] Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}% Precision: {val_precision:.4f} Recall: {val_recall:.4f} PRC: {val_prc:.4f}')
    
        scheduler.step(val_epoch_loss)

    if val_epoch_loss < best_loss:
        print(f'Saving model.  Validation loss: {val_epoch_loss:.4f} improved over previous {best_loss:.4f}')
        best_loss = val_epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_prc': best_prc,
            'loss_history': loss_history,
            'accuracy_history': accuracy_history,
            'precision_history': precision_history,
            'recall_history': recall_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'val_precision_history': val_precision_history,
            'val_recall_history': val_recall_history,
            'val_prc_history':val_prc_history
            }, "/home/ubuntu/project/224_pt_saves/best_loss_resnet50_model_224_norm.pth")

    if val_prc > best_prc:
        print(f'Saving model.  Validation prc: {val_prc:.4f} improved over previous {best_prc:.4f}')
        best_prc = val_prc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_prc': best_prc,
            'loss_history': loss_history,
            'accuracy_history': accuracy_history,
            'precision_history': precision_history,
            'recall_history': recall_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'val_precision_history': val_precision_history,
            'val_recall_history': val_recall_history,
            'val_prc_history':val_prc_history
            }, "/home/ubuntu/project/224_pt_saves/best_prc_resnet50_model_224_norm.pth")