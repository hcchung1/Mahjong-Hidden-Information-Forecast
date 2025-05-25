import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes = 1): # num_classes 是輸出層的類別數
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 28 * 28, num_classes)
        # raise NotImplementedError

    def forward(self, x):
        # (TODO) Forward the model
        # x: [batch_size, 3, 224, 224]
        x = self.pool1(self.relu1(self.conv1(x))) 
        # x: [batch_size, 16, 112, 112]
        x = self.pool2(self.relu2(self.conv2(x))) 
        # x: [batch_size, 64, 28, 28]
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # x: [batch_size, 32, 56, 56]
        x = x.view(x.size(0), -1)
        x = self.fc(x)  
        # raise NotImplementedError
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress

    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    # raise NotImplementedError
    model.eval()  # 設置模型為評估模式
    total_loss = 0.0
    total_samples = 0
    correct = 0
    
    with torch.no_grad():  # 禁用梯度計算
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            _, predicted = torch.max(outputs, 1)  # 獲取預測類別
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    # raise NotImplementedError
    model.eval()  # 設置模型為評估模式
    results = []
    
    with torch.no_grad():  # 禁用梯度計算
        for images, image_names in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 獲取預測類別
            
            for image_name, pred in zip(image_names, predicted.cpu().numpy()):
                results.append({'id': image_name, 'prediction': pred})
    
    # 保存結果到 CNN.csv
    df = pd.DataFrame(results)
    df.to_csv('CNN.csv', index=False)
    print(f"Predictions saved to 'CNN.csv'")
    return
