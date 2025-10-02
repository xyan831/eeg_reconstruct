import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np

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

# Training Loop
def train_model(device, model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Prediction/Inference
def predict(device, model, dataloader):
    model.eval()
    y_pred = []
    bn_pred = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            y_out, bn_out = model(X_batch)
            y_pred.append(y_out.cpu().numpy())
            bn_pred.append(bn_out.cpu().numpy())
    return np.concatenate(y_pred, axis=0), np.concatenate(bn_pred, axis=0)

