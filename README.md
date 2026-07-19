# Drug Response Gene Expression Prediction with MLOps

Predicting drug-induced gene expression changes using deep learning models and an end-to-end MLOps pipeline.

---

# 📌 Overview

This project predicts differential gene expression profiles in human immune cells after drug perturbation. Given drug information, molecular structure (SMILES), and cell metadata, the models predict the expression levels of thousands of genes.

The project combines deep learning architectures with modern MLOps practices such as:

- Experiment tracking with MLflow
- Data versioning with DVC
- Containerization using Docker
- CI/CD using GitHub Actions
- Cloud deployment with AWS

---

# 🚀 Features

✅ Predict expression levels for 3000 highly variable genes.

✅ Multiple deep learning architectures:

- GRU
- GRU + Attention
- Graph Neural Network (GCN)
- Graph Attention Network (GAT)

✅ Molecular graph construction using RDKit.

✅ Experiment tracking with MLflow.

✅ Data and model versioning with DVC.

✅ Dockerized training pipeline.

✅ CI/CD ready.

---

# 🧬 Dataset

The dataset consists of drug perturbation experiments performed on human PBMC cells.

## Input Features

### Structured Features

- Cell Type
- Drug Name
- LINCS ID

### Molecular Features

- SMILES representation

### Target

- Differential expression values of the top 3000 most variable genes.

---

# 🏗️ Project Structure

```text
drug-response-mlops/

├── artifacts/
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── components/
│   │   ├── config_loader.py
│   │   ├── data_ingestion.py
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   ├── data_loader.py
│   │   ├── train_sequence.py
│   │   ├── train_graph.py
│   │   ├── mlflow_logger.py
│   │   └── visualize.py
│   │
│   ├── models/
│   │   ├── gru.py
│   │   ├── gru_attention.py
│   │   ├── gnn.py
│   │   └── gat.py
│   │
│   └── pipeline/
│       └── training_pipeline.py
│
├── .github/
│
├── Dockerfile
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
└── dvc.yaml
```

---

# 🧠 Models

## 1. GRU Model

- Embedding layer for SMILES sequences.
- Multi-layer GRU encoder.
- Structured feature encoder.
- Fully connected regression head.

---

## 2. GRU + Attention

- GRU backbone.
- Attention mechanism over hidden states.
- Improved sequence representation.

---

## 3. Graph Neural Network (GCN)

Drug molecules are converted into graphs using RDKit.

### Node Features

- Atomic number
- Degree
- Formal charge
- Hybridization
- Aromaticity
- Number of hydrogens
- Valence

---

## 4. Graph Attention Network (GAT)

- Multi-head graph attention.
- Batch normalization.
- Structured feature fusion.
- Deep regression head.

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/Vatsal0025/drug-response-mlops
cd drug-response-mlops
```

---

## Create virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Running the Training Pipeline

Execute the complete pipeline:

```bash
python src/pipeline/training_pipeline.py
```

The pipeline performs:

1. Data loading
2. Gene selection
3. Feature encoding
4. Train-validation split
5. Dataset creation
6. Model training
7. Evaluation
8. MLflow logging
9. Artifact saving

---

# 📊 Experiment Tracking with MLflow

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

MLflow tracks:

- Hyperparameters
- Training loss
- Validation loss
- Metrics
- Model checkpoints
- Plots

---

# 📁 Data Versioning with DVC

Initialize DVC:

```bash
dvc init
```

Track datasets:

```bash
dvc add data/
```

Push to remote storage:

```bash
dvc push
```

Pull data:

```bash
dvc pull
```

---

# 🐳 Docker

## Build Docker image

```bash
docker build -t drug-response-mlops .
```

---

## Run Docker container

```bash
docker run drug-response-mlops
```

---

# ☁️ MLOps Pipeline

The project follows the following MLOps workflow:

```text
GitHub
   ↓

GitHub Actions
   ↓

Docker Build
   ↓

Amazon ECR
   ↓

Amazon EC2 / EKS
   ↓

Model Training & Deployment
```

---

# 🔧 Tech Stack

## Deep Learning

- PyTorch
- Torch Geometric
- RDKit

## Data Processing

- NumPy
- Pandas
- Scikit-learn

## Visualization

- Matplotlib

## MLOps

- Docker
- DVC
- MLflow
- GitHub Actions

## Cloud

- AWS ECR
- AWS EC2
- AWS EKS

---

# 📈 Results

| Model | Validation Loss | Sign Accuracy |
|--------|--------|--------|
| GRU | ~0.27 | ~82% |
| GAT | ~1.05 | ~63% |

---

# 🔮 Future Improvements

- Hyperparameter optimization.
- Distributed training.
- Model serving with FastAPI.
- Kubernetes deployment.
- Monitoring and alerting.
- Automated retraining.

---
