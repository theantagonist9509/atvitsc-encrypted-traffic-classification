# ATVITSC: Encrypted Traffic Classification

This repository contains the implementation of the Attention-based Vision Transformer and Spatiotemporal Feature Extractor (ATVITSC) model for encrypted traffic classification.

## Repository Structure

```text
.
├── report                          # Project reports and generated images
│   ├── report.pdf                  # Rendered project report in PDF format
│   ├── report.md                   # Markdown source of the project report
│   ├── images                      # Plots and visualizations generated during evaluation
│   │   ├── benign-malware-run.png
│   │   ├── benign-multiclass-run.png
│   │   └── example-random-session.png
│   └── paper.pdf                   # Original paper
├── pyproject.toml                  # Python project metadata and dependencies
├── uv.lock                         # Lockfile for consistent package resolution via `uv`
├── src                             # Core source code of the project
│   ├── arch.py                     # Contains the PyTorch neural network architectures
│   ├── main.py                     # Central script containing data configuration, execution, and training loops
│   └── session_image_dataset.py    # Custom PyTorch Dataset for parsing PCAP files into image tensors
├── runs                            # Jupyter notebooks for running specific classification tasks
│   ├── benign_malware.ipynb        # Notebook for binary classification (Benign vs. Malware)
│   └── benign_multiclass.ipynb     # Notebook for multi-class classification among different benign classes
└── archive                         # Directory containing raw PCAP files for different traffic classes
    ├── Benign                      # PCAP files for benign applications
    │   ├── BitTorrent.pcap
    │   ├── Facetime.pcap
    │   ├── Gmail.pcap
    │   ├── Outlook.pcap
    │   ├── Skype.pcap
    │   └── WorldOfWarcraft.pcap
    └── Malware                     # PCAP files for malware applications
        ├── Miuref.pcap
        ├── Tinba.pcap
        └── Zeus.pcap
```

## Source Code Details (`src/`)

The `src/` directory houses the core components of our pipeline. 

### `src/arch.py`
This file implements the core algorithmic components to build the **ATVITSC** network:
- **`PVT`** *(Packet-based Vision Transformer)*: Extracts global spatial features from session packet images, incorporating custom packet length embeddings into patch embeddings.
- **`ResAtConv`** *(Residual Attention Convolutional Block)*: Learns detailed local and compressed representations using 1D/2D attention mechanisms.
- **`STFE`** *(Spatial-Temporal Feature Extractor)*: Combines `ResAtConv` with a Bidirectional LSTM to capture the temporal progression of sequential packet representation mappings.
- **`ATVITSC`** *(Overall Architecture)*: The main model unifying all the modules above. It merges features from `PVT` and `STFE` through a dynamic weighting network, passing the result to a linear classification head.

### `src/session_image_dataset.py`
This module handles translating raw traffic data into model inputs:
- **`SessionImageDataset`**: A PyTorch `Dataset` that parses raw `.pcap` files. It groups network packets chronologically by session (source/destination IP, ports, and protocol). It then extracts the payload, converting the bytes into 2D image matrices padded to a fixed size, along with packet length arrays.

### `src/main.py`
The primary entry point:
- Configures epochs, batch size, learning rates, and dataset paths.
- Instantiates the model, optimizer, and `CrossEntropyLoss` criterion.
- Provides a `train_model()` function that handles the training loop, checkpoints, and evaluation (Precision, Recall, Accuracy, and F1).
- Contains plotting functions for the evaluation metrics.

> **Note on `runs/` Notebooks**: The `.ipynb` notebooks in the `runs/` directory are variations of `src/main.py` adapted for specific classification tasks (e.g., Benign vs. Malware in `benign_malware.ipynb`, or Benign multi-class in `benign_multiclass.ipynb`).

## Replicating Results

This project uses [`uv`](https://github.com/astral-sh/uv) to manage Python dependencies. Follow these steps to reproduce the results:

1. **Install `uv`**:
   If you have not installed `uv`, follow the [official instructions](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Dependencies**:
   Run the following in the project root to install packages from `pyproject.toml` and `uv.lock`:
   ```bash
   uv sync
   ```

3. **Launch Notebook Server**:
   Start Jupyter using the `uv` environment:
   ```bash
   uv run jupyter notebook
   ```

4. **Execute Notebooks:**
   In your browser, navigate to the `runs/` directory and open:
   - `benign_malware.ipynb`
   - `benign_multiclass.ipynb`
   
   Run all cells to process the data in `archive/`, train the models, save checkpoints to `checkpoints/`, and generate the final evaluation plots.
