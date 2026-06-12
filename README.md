# Two Digit Classifier

> *Costella Matteo — Deep Learning and Generative Models — A.A. 2025/26*

---

## Overview

During the training of the LeNet99 classifier on MNIST99, all runs are logged via **TensorBoard** to monitor loss and accuracy curves in real time, both on the training and validation sets. The same metrics are used to compare results across the ablation study runs.

---

## How logs are generated

Both in `train_lenet99.py` and `train_lenet99_ablation.py`, a `SummaryWriter` is instantiated at the start of training and writes logs to a dedicated subfolder for the current run:

```python
from torch.utils.tensorboard import SummaryWriter

logdir = os.path.join(RUNS_DIR, RUN_NAME)
writer = SummaryWriter(log_dir=logdir)
```

The default path for the logs is:

```
/kaggle/working/runs/<RUN_NAME>/
```

where `RUN_NAME` is the name defined in the run's YAML configuration file (e.g. `A0_base_adam`).

---

## Logged metrics

At the end of each epoch, the following values are written:

```python
writer.add_scalar("loss/train", train_loss, epoch)
writer.add_scalar("acc/train",  train_acc,  epoch)
writer.add_scalar("loss/val",   val_loss,   epoch)
writer.add_scalar("acc/val",    val_acc,    epoch)
writer.add_scalar("lr",         lr_now,     epoch)
```

---

## Log file structure

After training, the `runs/` folder will contain one directory per executed run:

```
runs/
├── lenet99_adam_lr1e-3/      ← base run (train_lenet99.py)
├── A0_base_adam/             ← ablation baseline
├── A1_sgd_momentum/          ← SGD ablation
├── A2_dropout0/              ← no-dropout ablation
└── A3_poolmax/               ← MaxPool ablation
```

Each folder contains TensorFlow event files (format `events.out.tfevents.*`) that can be loaded and compared in a single TensorBoard session.

---

## Kaggle limitation — Exporting logs

TensorBoard is not directly supported on Kaggle because the environment does not expose external ports. The solution adopted is to **export the logs** and import them into the Google Colab notebook.

### 1. Compress the logs

```bash
zip -r runs.zip runs
```

### 2. Download `runs.zip` from Kaggle

From the Kaggle output file panel, download the generated `runs.zip` file.

### 3. Open the dedicated Colab notebook

Import the file into Google Colab and launch TensorBoard from there:

> [TwoDigitClassifier (TensorBoard).ipynb](https://colab.research.google.com/drive/1zC6tYKHBNXDp3eWyH95o9FL5Qb9nZaA5?usp=sharing)

### 4. Cleanup (optional)

```bash
rm runs.zip
```
