## MNIST CNN — Lightweight Model (<20K params) with Training Metrics

### Overview
This project trains a compact convolutional neural network on MNIST. The network is designed to stay under a small parameter budget while remaining easy to train and evaluate.

### Architecture
- Input: 1×28×28 grayscale image
- Feature extractor:
  - Block 1: Conv(1→6, 3×3, padding=1) → BatchNorm(6) → ReLU → MaxPool(2×2)
  - Block 2: Conv(6→12, 3×3, padding=1) → BatchNorm(12) → ReLU → MaxPool(2×2)
  - Block 3: Conv(12→24, 3×3, padding=1) → BatchNorm(24) → ReLU → MaxPool(2×2)
  - Block 4: Conv(24→32, 3×3, padding=1) → BatchNorm(32) → ReLU → MaxPool(2×2)
- Classifier:
  - Flatten (32×1×1)
  - Linear(32 → 10)
- Output: log-probabilities over 10 classes

Flow of tensor shapes:
```
28×28×1 → 14×14×6 → 7×7×12 → 3×3×24 → 1×1×32 → 10
```

### Key Training Settings
- Optimizer: Adam (lr = 1e-3)
- Loss: Negative Log-Likelihood (with model outputting log_softmax)
- Batch size: 64
- Epochs: 10 (configurable in `model.py` main)
- Normalization: mean = 0.1307, std = 0.3081

### How to Run
```bash
cd /Users/sandipan/Documents/MyProjects/MNIST-Model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python model.py
```

### Parameter Count
`model.py` prints the total and trainable parameter counts at startup. Example:
```
Final model - Total parameters: 12,xxx | Trainable parameters: 12,xxx
```

### Training Metrics (per epoch)
The script prints epoch-wise metrics to the console. Record them here after a run.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 1 |        |        |        |        |
| 2 |        |        |        |        |
| 3 |        |        |        |        |
| 4 |        |        |        |        |
| 5 |        |        |        |        |
| 6 |        |        |        |        |
| 7 |        |        |        |        |
| 8 |        |        |        |        |
| 9 |        |        |        |        |
| 10 |       |        |        |        |

Tip: You can copy the printed epoch lines into this table. They look like:
```
Epoch 05 | Train Loss: 0.0xxx | Train Acc: 0.99xx | Val Loss: 0.0xxx | Val Acc: 0.99xx
```

*Metrics updated automatically after training.*

### Optional: Save Metrics Automatically
If you want to save metrics during training, you can serialize the `history` dict returned by `train(...)`:
```python
import json
with open('history.json', 'w') as f:
    json.dump(history, f)
```
Then convert it to CSV/Markdown as you prefer.

### Troubleshooting
- SSL certificate errors when downloading MNIST: install `certifi` and set `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` to `certifi.where()`.
- BatchNorm shape errors: ensure `BatchNorm2d(C)` matches the preceding `Conv2d` out_channels and restart the Python session after architecture edits.

### License
MIT


