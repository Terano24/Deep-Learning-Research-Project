# Prediction of Immunogenic Peptides for T-Cell Immunity as Potential Epitopes for Targeted Vaccines and Targeted Immunotherapy in Lung Cancer Based on Deep Learning Using Convolutional Neural Network (CNN) Algorithms.
Prediksi Peptida  untuk Imunitas Sel-T sebagai Epitop Potensial Targeted-Vaccine dan Targeted-Immunotherapy pada Kanker Paru Berbasis Deep Learning Menggunakan Algoritma Convolutional Neural Network (CNN)


-----------------------------------------
OLIMPIADE PENELIIAN SISWA INDONESIA 2024
-----------------------------------------
Original work by:
I Nyoman Agung Ananda Terano & Mirrabell Frederica Hadiwijono 
-----------------------------------------

## 🚀 Quick Setup Guide

Because this repository separates code from heavy datasets and model weights, you need to follow these steps to get it running on your local machine.

### 1. Python Environment Requirements
- **Python Version**: `3.11` or `3.12` is highly recommended. *(Python 3.14+ is currently too new for TensorFlow packages!)*
- If you just installed Python on Windows, ensure **"Add Python to PATH"** was checked during installation.

### 2. Download Dependencies
If you are using Visual Studio Code, you can instantly install all required libraries by using the built-in task:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`).
2. Type **Tasks: Run Task** and select it.
3. Choose **setup**.

*(Alternatively, run `pip install -r requirements.txt` directly in your terminal).*

### 3. Running the Model
Once everything is placed, you can run the model directly from the command line:

**Single Mode (Test a specific Peptide + HLA pair):**
```bash
python main.py --mode single --epitope "ALAKAAA" --hla "HLA-A*0201"
```

**Multiple Mode (Batch Processing):**
```bash
python main.py --mode multiple --intdir "data/raw/input.csv" --outdir "data/results/"
```

> **Note on TensorFlow Warnings**: If you see warnings like `Value in checkpoint could not be found... (root).optimizer.iter`, this is completely normal! TensorFlow is just informing you that the optimizer state wasn't restored, which is expected since we are only running *inference* (predictions) and not training.
