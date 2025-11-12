import os
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data_util import mat2numpy
from .model_util import get_datalabel, kfold_split, torch_dataloader
from .model_cnn import CNN

class classification:
    def __init__(self, data_path, model_path, prefix):
        self.data_path = data_path
        self.model_path = model_path
        self.seiz_path = os.path.join(data_path, f"{prefix}_seizure_data.mat")
        self.nseiz_path = os.path.join(data_path, f"{prefix}_non_seizure_data.mat")

    def get_data(self):
        seiz_data = mat2numpy(self.seiz_path, "data")
        nseiz_data = mat2numpy(self.nseiz_path, "data")
        
        full_data = np.concatenate((seiz_data, nseiz_data), axis=0)
        full_label = get_datalabel(seiz_data, nseiz_data)
        print(full_label.shape)
        
        full_data = torch.FloatTensor(full_data)
        full_label = torch.LongTensor(full_label).squeeze()
        
        num_classes = torch.unique(full_label).numel()
        
        print(full_data.shape)
        print(full_label.shape)
        
        torch.manual_seed(0)
        
        # KFold split
        X_train, X_test, y_train, y_test = kfold_split(full_data, full_label)
        
        # Add channel dim [B, 1, 25, 250]
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)
        
        torch.manual_seed(1)
        
        trainloader = torch_dataloader(X_train, y_train, batch_size=64, datatype="train")
        testloader = torch_dataloader(X_test, y_test, batch_size=64, datatype="test")
        return trainloader, testloader, num_classes

    def train(self):
        trainloader, testloader, num_classes = self.get_data()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(num_classes=num_classes).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
            
            avg_loss = running_loss / len(trainloader.dataset)
            print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in testloader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = 100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), self.model_path)
                print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

    def test(self):
        trainloader, testloader, num_classes = self.get_data()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Load model
        model = CNN(num_classes=num_classes).to(device)
        state_dict = torch.load(self.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

