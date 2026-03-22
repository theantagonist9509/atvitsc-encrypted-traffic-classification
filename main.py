# %%
from collections import Counter
import os
import glob
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import random_split, DataLoader
from torchinfo import summary
from tqdm import tqdm

from session_image_dataset import SessionImageDataset
from arch import ATVITSC

# %%
# Configuration
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
M_SIZE = 256
N_SIZE = 16

BENIGN_PCAP_GLOB = 'archive/Benign/**/*.pcap'
MALWARE_PCAP_GLOB = 'archive/Malware/**/*.pcap'

PCAP_SIZE_LIMIT = 30 * 1024 * 1024

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# %%
# Load dataset
def filter_pcaps_by_size(pcap_paths, size_limit):
    filtered_pcaps = []
    for pcap_path in pcap_paths:
        if os.path.getsize(pcap_path) < size_limit:
            filtered_pcaps.append(pcap_path)
        else:
            print(f"Removed {pcap_path} due to size limit {os.path.getsize(pcap_path):,} > {size_limit:,}")
    return filtered_pcaps

benign_pcap_paths = glob.glob(BENIGN_PCAP_GLOB, recursive=True)
malware_pcap_paths = glob.glob(MALWARE_PCAP_GLOB, recursive=True)

benign_pcap_paths = filter_pcaps_by_size(benign_pcap_paths, PCAP_SIZE_LIMIT)
malware_pcap_paths = filter_pcaps_by_size(malware_pcap_paths, PCAP_SIZE_LIMIT)

pcap_labels = ([0] * len(benign_pcap_paths)) + ([1] * len(malware_pcap_paths))

dataset = SessionImageDataset(benign_pcap_paths + malware_pcap_paths, pcap_labels, m=M_SIZE, n=N_SIZE)

counts = Counter(dataset.labels)
print(counts)

# %%
# Split dataset
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {train_size}")
print(f"Val   size: {val_size}")

# %%
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model, optimizer, and criterion
model = ATVITSC()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

summary(model, input_data=[torch.randn(BATCH_SIZE, 1, 64, 64), torch.randint(0, 1502, (BATCH_SIZE, 16))])

# %%
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_dir="checkpoints"):
    model.to(device)
    
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': []
    }
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_paths = glob.glob(os.path.join(save_dir, "model_epoch_*.pth"))
    if checkpoint_paths:
        latest_checkpoint = max(checkpoint_paths, key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, packet_lens, labels in loop:
            # images: (B, H, W) -> (B, 1, H, W)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            
            images = images.to(device).float()
            packet_lens = packet_lens.to(device).long()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images, packet_lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=train_loss/train_total, acc=100.*train_correct/train_total)
            
        epoch_train_loss = train_loss / train_total
        history['train_loss'].append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, packet_lens, labels in val_loop:
                if images.ndim == 3:
                    images = images.unsqueeze(1)
                
                images = images.to(device).float()
                packet_lens = packet_lens.to(device).long()
                labels = labels.to(device).long()
                
                outputs = model(images, packet_lens)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Use [:, 1] for binary classification roc_auc, assuming 1 is positive class
                all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_loop.set_postfix(loss=loss.item())
                
        epoch_val_loss = val_loss / len(val_set)
        history['val_loss'].append(epoch_val_loss)
        
        # Calculate metrics
        val_acc = accuracy_score(all_labels, all_preds)
        
        try:
            val_prec = precision_score(all_labels, all_preds, zero_division=0)
            val_rec = recall_score(all_labels, all_preds, zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            if len(np.unique(all_labels)) > 1:
                val_roc = roc_auc_score(all_labels, all_probs)
            else:
                val_roc = float('nan')
            val_cm = confusion_matrix(all_labels, all_preds)
        except ValueError:
            val_prec, val_rec, val_f1, val_roc, val_cm = 0, 0, 0, 0, None
            
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc)
        
        print(f"\nEpoch {epoch+1}/{epochs} Results:")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | ROC AUC: {val_roc:.4f}")
        if val_cm is not None:
            print(f"Confusion Matrix:\n{val_cm}\n")
            
        # Save model and training state after each epoch
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(checkpoint, save_path)
        
    return history

# %%
# Start training
history = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=EPOCHS, save_dir=SAVE_DIR)

# %%
# Plotting
def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot Losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy & F1
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.plot(epochs, history['val_f1'], label='Val F1 Score')
    plt.title('Accuracy & F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot Precision & Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_precision'], label='Val Precision')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot ROC AUC
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_roc_auc'], label='Val ROC AUC', color='purple')
    plt.title('ROC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_metrics(history)

# %%
