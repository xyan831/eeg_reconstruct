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
    def __init__(self, path_config, model_config, param_config):
        self.data_path = path_config.get("gen_path")
        self.model_path = path_config.get("model_path")
        self.log_path = path_config.get("log_path")
        
        self.model_type = model_config.get("model_type", "cnn")
        self.epoch_num = model_config.get("epoch_num", 10)
        self.learning_rate = model_config.get("learning_rate", 1e-3)
        
        self.name_prefix = param_config.get("name_prefix")
        self.model_prefix = param_config.get("model_prefix")
        self.namelist = param_config.get("namelist", [self.name_prefix])
        
        is_usemat = param_config.get("is_usemat", False)
        is_usetrain = param_config.get("is_usetrain", False)
        if is_usemat:
            self.data_path = path_config.get("mat_path")
        elif is_usetrain:
            self.data_path = path_config.get("data_path")
        
        #self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label = "data"
        self.file_config()

    def file_config(self):
        model_name = f"{self.model_prefix}_class_{self.model_type}.pth"
        log_name = f"{self.model_prefix}_class_{self.model_type}.txt"
        test_log_name = "test_log.txt"
        time_log_name = "time_log.txt"
        seiz_name = f"{self.name_prefix}_seiz.mat"
        nseiz_name = f"{self.name_prefix}_nseiz.mat"
        self.seizlist = [f"{name}_seiz.mat" for name in self.namelist]
        self.nseizlist = [f"{name}_nseiz.mat" for name in self.namelist]
        self.model_file = os.path.join(self.model_path, model_name)
        self.log_file = os.path.join(self.log_path, log_name)
        self.test_log_file = os.path.join(self.log_path, test_log_name)
        self.time_log_file = os.path.join(self.log_path, time_log_name)

    def get_model(self, num_classes, load_pretrain=False):
        if self.model_type=="cnn":
            model = CNN(num_classes=num_classes).to(self.device)
        elif self.model_type=="lstm":
            model = CNN_LSTM(num_classes=num_classes).to(self.device)
        elif self.model_type=="transformer":
            if "our" in self.name_prefix:
                num_channels = 26
            elif "nicu" in self.name_prefix:
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
            ).to(self.device)
        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if load_pretrain:
            print("load model: ", self.model_file)
            state_dict = torch.load(self.model_file, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
        return model, criterion, optimizer

    def get_data(self):
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
        
        trainloader = torch_dataloader(X_train, y_train, batch_size=64, is_train=True)
        testloader = torch_dataloader(X_test, y_test, batch_size=64, is_train=False)
        return trainloader, testloader, num_classes

    def train(self):
        # Create results log file header
        with open(self.log_file, "w") as f:
            f.write("epoch,train_loss,val_acc,val_sen,val_spe,val_f1\n")
        
        # get data
        trainloader, testloader, num_classes = self.get_data()
        model, criterion, optimizer = self.get_model(num_classes, load_pretrain=False)
        best_accuracy = 0.0
        # track time for process
        start_time = datetime.now()
        for epoch in range(self.epoch_num):
            model.train()
            running_loss = 0.0
            for X, y in trainloader:
                X, y = X.to(self.device), y.to(self.device)
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
            TP = TN = FP = FN = 0
            with torch.no_grad():
                for X, y in testloader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = model(X)
                    _, predicted = torch.max(outputs.data, 1)
                    TP += ((predicted == 1) & (y == 1)).sum().item()
                    TN += ((predicted == 0) & (y == 0)).sum().item()
                    FP += ((predicted == 1) & (y == 0)).sum().item()
                    FN += ((predicted == 0) & (y == 1)).sum().item()

            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
            print(f"acc: {accuracy * 100:.2f}%, sen: {sensitivity * 100:.2f}%, spe: {specificity * 100:.2f}%, f1: {f1 * 100:.2f}%")
            
            # Log results to file
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{accuracy:.4f},{sensitivity:.4f},{specificity:.4f},{f1:.4f}\n")
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), self.model_file)
                print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
        
        # track time for process
        end_time = datetime.now()
        time_taken = end_time - start_time
        # Log results to file
        with open(self.time_log_file, "a") as f:
            f.write(f"Class {self.name_prefix}_{self.model_type} Train Time:{time_taken}\n")
        print("saved model to", self.model_file)
        print("results logged to", self.log_file)
        print("time taken:", time_taken)

    def test(self):
        print(f"testing {self.name_prefix}")
        # get data
        trainloader, testloader, num_classes = self.get_data()
        # Load model
        model, criterion, optimizer = self.get_model(num_classes, load_pretrain=True)
        model.eval()
        correct = 0
        total = 0
        # track time for process
        start_time = datetime.now()
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        # track time for process
        end_time = datetime.now()
        time_taken = end_time - start_time
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("time Taken:", time_taken)
        # Log results to file
        with open(self.test_log_file, "a") as f:
            f.write(f"Class {self.name_prefix}_{self.model_type} Accuracy:{accuracy:.2f}%\n")
        with open(self.time_log_file, "a") as f:
            f.write(f"Class {self.name_prefix}_{self.model_type} Test Time:{time_taken}\n")

