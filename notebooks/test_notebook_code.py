import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath('..'))

from src.models.cnn_model import arsitekturCNN
from src.data.processing import (
    HLA_Dictionary, dictionary, construct_aaindex,
    peptide_iterate, hla_iterate
)
from pathlib import Path

# Fix TF warning
os.environ['TF_USE_LEGACY_KERAS'] = '1'

BASE_DIR = Path('..').resolve()
after_pca_path = BASE_DIR / 'data' / 'raw' / 'after_pca.txt'
hla_path = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'
model_weights_path = BASE_DIR / 'models' / 'CNN_WEIGHT_OPSI'

# Load required data for preprocessing
after_pca = np.loadtxt(after_pca_path)
hla = pd.read_csv(hla_path, sep='\t')
hla_dic = HLA_Dictionary(hla)
inventory = list(hla_dic.keys())
dic_inventory = dictionary(inventory)

# Load model
cnn_model = arsitekturCNN()
cnn_model.load_weights(str(model_weights_path) + '/')
print("Model loaded successfully!")

# Load your dataset here for evaluation
# For example, using the IEDB Dataset
dataset_path = BASE_DIR / 'data' / 'raw' / 'IEDB Dataset.csv'
df = pd.read_csv(dataset_path)

print("Original Data:")
print(df.head())

# Preprocess the ground truth labels
if 'immunogenicity' in df.columns:
    if df['immunogenicity'].dtype == object:
        # Convert 'Positive' to 1, others to 0
        df['label'] = df['immunogenicity'].apply(lambda x: 1 if str(x).strip().lower() == 'positive' else 0)
    else:
        # Assuming they are already numerical probabilities, threshold at 0.5
        df['label'] = df['immunogenicity'].apply(lambda x: 1 if float(x) >= 0.5 else 0)
else:
    print("Warning: 'immunogenicity' column not found. You need labels for evaluation.")

# Drop rows with missing peptide or HLA
df = df.dropna(subset=['peptide', 'HLA'])
print(f"\nTotal samples for evaluation: {len(df)}")

# The construct_aaindex function expects a DataFrame with 'peptide', 'HLA', and 'immunogenicity'
eval_df = df[['peptide', 'HLA']].copy()
eval_df['immunogenicity'] = ['0'] * len(eval_df)  # Dummy target for preprocessing

# Construct inputs for the model
dataset_score = construct_aaindex(eval_df, hla_dic, after_pca, dic_inventory)

input1_score = peptide_iterate(dataset_score)
input2_score = hla_iterate(dataset_score)

print("Running predictions...")
predictions_prob = cnn_model.predict(x=[input1_score, input2_score], verbose=1)
df['pred_prob'] = predictions_prob

# Convert probability to binary prediction (threshold = 0.5)
threshold = 0.5
df['pred_label'] = (df['pred_prob'] > threshold).astype(int)

y_true = df['label'].values
y_pred = df['pred_label'].values
y_prob = df['pred_prob'].values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = np.nan # If only one class is present in true labels

print("=== Evaluation Metrics ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
