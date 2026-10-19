# Polymer Representation Comparison for Tg Prediction

This repository contains research code for predicting the glass transition temperature (Tg) of polymers by comparing various chemical language representations (e.g., BigSMILES, PSMILES, HAPPY) and Graph Neural Network (GNN) architectures.

## 📌 Key Features
- **Polymer Representations:** Comparison of tokenization and embedding strategies across representations including BigSMILES, PSMILES, HAPPY, and GT4SD.
- **Models:** Implementation and application of various deep learning models such as GAT, GCN, MPNN, and HPG.
- **Task:** Regression prediction of the glass transition temperature (Tg) using polymer datasets.

## 📦 Pre-trained Models & Checkpoints
Due to size limits, large model checkpoint files (`.pth`, `.bin`, etc.) are not included in this GitHub repository. They can be downloaded directly from our Hugging Face model hub.

- **Hugging Face Checkpoints:** [https://huggingface.co/h0518a/polymer_tg_prediction_checkpoints](https://huggingface.co/h0518a/polymer_tg_prediction_checkpoints)
- Please place the downloaded folders (e.g., `ckpt/`, `happy_tg/checkpoint/`) in the root directory of this project.

## 🚀 Installation
The main Python libraries required to run this project are as follows:
```bash
pip install torch dgl rdkit pandas scikit-learn