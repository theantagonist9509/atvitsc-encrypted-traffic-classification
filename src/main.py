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
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
M_SIZE = 256
N_SIZE = 16

BENIGN_PCAP_PATHS = [
    "archive/Benign/BitTorrent.pcap",
    "archive/Benign/Facetime.pcap",
    "archive/Benign/Gmail.pcap",
    "archive/Benign/Outlook.pcap",
    "archive/Benign/Skype.pcap",
    "archive/Benign/WorldOfWarcraft.pcap",
]

MALWARE_PCAP_PATHS = [
    "archive/Malware/Miuref.pcap",
    "archive/Malware/Tinba.pcap",
    "archive/Malware/Zeus.pcap",
]

PCAP_PATHS = BENIGN_PCAP_PATHS + MALWARE_PCAP_PATHS
PCAP_LABELS = [0] * len(BENIGN_PCAP_PATHS) + [1] * len(MALWARE_PCAP_PATHS)

SAVE_DIR = "checkpoints/benign-malware"
os.makedirs(SAVE_DIR, exist_ok=True)

# %%
# Load dataset
dataset = SessionImageDataset(PCAP_PATHS, PCAP_LABELS, m=M_SIZE, n=N_SIZE)

class_counts = Counter(dataset.labels)
print(class_counts)

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
model = ATVITSC(num_classes=len(class_counts))
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

            loop.set_postfix(loss=train_loss/train_total, acc=train_correct/train_total)

        epoch_train_loss = train_loss / train_total
        history['train_loss'].append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_total = 0

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
                val_total += images.size(0)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_loop.set_postfix(loss=val_loss/val_total)

        epoch_val_loss = val_loss / val_total
        history['val_loss'].append(epoch_val_loss)

        # Calculate metrics
        val_acc = accuracy_score(all_labels, all_preds)
        val_prec = precision_score(all_labels, all_preds, average='weighted')
        val_rec = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_cm = confusion_matrix(all_labels, all_preds)

        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        print(f"\nEpoch {epoch+1}/{epochs} Results:")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
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

    plt.tight_layout()
    plt.show()

plot_metrics(history)

# %%
