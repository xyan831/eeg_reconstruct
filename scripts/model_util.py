import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from .data_util import mat2numpy, crop_timestep

# ----------------------
# Data Loading & Processing
# ----------------------

# normalize data
def normalize(data,scaler,type="normal"):
    samples,channels,timesteps = data.shape
    data_norm = data.reshape(-1, timesteps)                         # Flatten for scaling
    if type=="normal":
        data_norm = scaler.fit_transform(data_norm)                 # normalize data
    elif type=="reverse":
        data_norm = scaler.inverse_transform(data_norm)             # inverse normalized data
    data_norm = data_norm.reshape(samples, channels, timesteps)     # Reshape back
    return data_norm

# get label from seiz and nseiz data
def get_datalabel(seiz_data, nseiz_data):
    seiz_label = np.ones(len(seiz_data))
    nseiz_label = np.zeros(len(nseiz_data))
    return np.concatenate((seiz_label, nseiz_label), axis=0)

def kfold_split(data, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(data):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        break  # Use first fold
    return X_train, X_test, y_train, y_test

def torch_dataloader(X_data, y_data, batch_size=32, datatype="test"):
    # Convert to PyTorch tensors for torch unet
    data = TensorDataset(X_data, y_data)
    
    # Create dataloader
    if datatype == "train":
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    elif datatype == "test":
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        print("error: invalid datatype")
    return data_loader

# ----------------------
# Training & Evaluation
# ----------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=30)
    
    for step, data in enumerate(data_loader):
        patches, labels = data  # unpack batch
        
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        pred = model(patches)
        
        sample_num += labels.size(0)
        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()
        
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    all_preds = []
    all_labels = []

    seizureCount = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=30)

    for step, data in enumerate(data_loader):
        patches, labels = data
        patches = patches.to(device)
        labels = labels.to(device)

        pred = model(patches)
        loss = loss_function(pred, labels)

        accu_loss += loss.detach()
        sample_num += labels.size(0)

        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()
        seizureCount += (pred_classes == 1).sum()

        all_preds.append(pred_classes.cpu())
        all_labels.append(labels.cpu())
        
        

        data_loader.desc = "[testing epoch {}] loss: {:.3f}, acc: {:.3f}, pred_seizure: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            seizureCount.item() / sample_num
        )
    # calculate sen, spe, f1
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, sen, spe, f1

# Training Loop
def train_model(device, model, model_type, dataloader, criterion, optimizer, epochs=10, results_file="results.txt"):
    # Create results file with header
    with open(results_file, "w") as f:
        if model_type=="unet":
            f.write("epoch,loss\n")
        elif model_type=="vae":
            f.write("epoch,total_loss,recon_loss,kl_loss\n")
        else:
            raise ValueError(f"Invalid model type {model_type}")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            if model_type=="unet":
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                batch_recon = loss.item()
                batch_kl = 0.0
            elif model_type=="vae":
                recon_outputs, mu, logvar = model(X_batch)
                loss, batch_recon, batch_kl = criterion(recon_outputs, y_batch, mu, logvar)
                batch_recon = batch_recon.item()
                batch_kl = batch_kl.item()
            else:
                raise ValueError(f"Invalid model type {model_type}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            recon_loss += batch_recon
            kl_loss += batch_kl
        
        # calculate epoch average
        avg_loss = running_loss / len(dataloader)
        avg_recon_loss = recon_loss / len(dataloader)
        avg_kl_loss =  kl_loss / len(dataloader)
        
        # Log results to file
        with open(results_file, "a") as f:
            if model_type=="unet":
                f.write("{},{:.4f}\n".format(epoch+1, avg_loss))
            elif model_type=="vae":
                f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, avg_loss, avg_recon_loss, avg_kl_loss))
            else:
                raise ValueError(f"Invalid model type {model_type}")
        # print results
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Prediction/Inference
def predict(device, model, dataloader):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            # Handle different model types
            if hasattr(model, 'model_type') and model.model_type == "vae":
                # For VAE, extract only the reconstruction from the tuple
                recon_outputs, _, _ = model(X_batch)
                y_out = recon_outputs
            else:
                # For UNet and other models
                y_out = model(X_batch)
            y_pred.append(y_out.cpu().numpy())
    return np.concatenate(y_pred, axis=0)

