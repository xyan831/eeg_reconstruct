import os
import numpy as np
from scipy.io import loadmat
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data_util import mat2numpy
from .model_util import get_datalabel, kfold_split, torch_dataloader
from .model_cnn import CNN
from .model_cnn_lstm import CNN_LSTM
from .model_transformer import EEGTransformerClassifier

class classification:
    def __init__(self, data_path, model_path, log_path, name, model, model_type="cnn", num_epochs=10):
        self.data_path = data_path
        self.model_path = model_path
        self.log_path = log_path
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.model_name = f"{model}_class_{model_type}.pth"
        self.log_name = f"{model}_class_{model_type}.txt"
        self.seiz_name = f"{name}_seiz.mat"
        self.nseiz_name = f"{name}_nseiz.mat"
        self.name = name
        self.label = "data"

    def file_config(self, name, namelist=[]):
        self.seizlist = [f"{name}_seiz.mat" for name in namelist]
        self.nseizlist = [f"{name}_nseiz.mat" for name in namelist]

    def get_logname(self):
        now = datetime.now()
        return f"{self.log_name}_{now.strftime('%Y%m%d_%H%M%S')}.txt"

    def get_model(self, num_classes, load_pretrain=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_type=="cnn":
            model = CNN(num_classes=num_classes).to(device)
        elif self.model_type=="lstm":
            model = CNN_LSTM(num_classes=num_classes).to(device)
        elif self.model_type=="transformer":
            if "our" in self.name:
                num_channels = 26
            elif "nicu" in self.name:
                num_channels = 21
            else:
                num_channels = 21
            model = EEGTransformerClassifier(
                num_classes=num_classes,
                num_channels=num_channels,      # Your EEG has 26 channels
                signal_len=496,       # Your EEG sequence length
                patch_size=16,        # You can adjust this
                embed_dim=128,        # Embedding dimension
                depth=6,              # Number of transformer blocks
                num_heads=4,          # Number of attention heads
                mlp_ratio=4.0,        # MLP expansion ratio
                drop_rate=0.1,        # Dropout rate
                attn_drop_rate=0.1,   # Attention dropout rate
                drop_path_rate=0.1    # Stochastic depth rate
            ).to(device)
        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if load_pretrain==True:
            print("load model: ", self.model_name)
            model_file = os.path.join(self.model_path, self.model_name)
            state_dict = torch.load(model_file, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        return device, model, criterion, optimizer

    def get_data(self):
        #seiz_data = mat2numpy(os.path.join(self.data_path, self.seiz_name), self.label)
        #nseiz_data = mat2numpy(os.path.join(self.data_path, self.nseiz_name), self.label)
        seiz_datalist = []
        nseiz_datalist = []
        for seizfile, nseizfile in zip(self.seizlist, self.nseizlist):
            data_seiz = mat2numpy(os.path.join(self.data_path, seizfile), self.label)
            data_nseiz = mat2numpy(os.path.join(self.data_path, nseizfile), self.label)
            seiz_datalist.append(data_seiz)
            nseiz_datalist.append(data_nseiz)
        seiz_data = np.concatenate(seiz_datalist)        
        nseiz_data = np.concatenate(nseiz_datalist)
        
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
        # create results file for log
        #log_name = self.get_logname()
        log_file = os.path.join(self.log_path, self.log_name)
        # Create results file with header
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,val_acc\n")
        
        # get data
        trainloader, testloader, num_classes = self.get_data()
        device, model, criterion, optimizer = self.get_model(num_classes, load_pretrain=False)
        best_accuracy = 0.0
        for epoch in range(self.num_epochs):
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
            
            # Log results to file
            with open(log_file, "a") as f:
                f.write("{},{:.4f},{:.4f}\n".format(epoch+1, avg_loss, accuracy))
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name))
                print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
        
        print("saved model to", self.model_name)
        print("results logged to", self.log_name)

    def test(self):
        print(f"testing {self.name}")
        trainloader, testloader, num_classes = self.get_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model
        device, model, criterion, optimizer = self.get_model(num_classes, load_pretrain=True)
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

