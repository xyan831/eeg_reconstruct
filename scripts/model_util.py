import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

def train_model2(device, model, train_loader, val_loader, criterion, optimizer,
                epochs=10, results_file="results.txt"):

    # Create results file with header
    with open(results_file, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_sen,val_spe,val_f1\n")

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        # Training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        val_loss, val_acc, val_sen, val_spe, val_f1 = evaluate(model, val_loader, device, criterion)

        # Save best model (optional)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")

        # Log results to file
        with open(results_file, "a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                epoch+1, train_loss, train_acc, val_loss, val_acc, val_sen, val_spe, val_f1
            ))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

# evaluate model
def evaluate(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Compute metrics
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="binary")
    val_sen = recall_score(all_labels, all_preds, pos_label=1)  # Sensitivity = Recall (for positive class)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    val_spe = tn / (tn + fp)

    return val_loss / len(dataloader), val_acc, val_sen, val_spe, val_f1

# Prediction/Inference
def predict(device, model, dataloader):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            y_out = model(X_batch)
            y_pred.append(y_out.cpu().numpy())
    return np.concatenate(y_pred, axis=0)

