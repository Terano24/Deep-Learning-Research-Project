"""
DeepImmuno's published model, scored through its own released weights.

There are two checkpoints in this repository and they are not interchangeable:

  models/CNN_WEIGHT_OPSI   DeepImmuno's released cnn_model_331_3_7, byte for byte
                           (sha256 of .data-00000-of-00001: 6a8c81c535c2647e3d139175...).
                           This is the published predictor. Use it whenever the paper
                           reports a number *about DeepImmuno*.

  models/CNN_RETRAINED     the same architecture, retrained by us on our own training
                           split (src/models/train.py). This is OUR model. It is not
                           DeepImmuno and must never be labelled as such.

Scoring the published checkpoint is the only honest way to report DeepImmuno, so every
analysis that needs it imports `score` from here rather than reading a column out of a
CSV and trusting its name.

Run:  python -m src.models.deepimmuno        (prints the AUC on the held-out test split)
"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PUBLISHED_WEIGHTS = BASE_DIR / 'models' / 'CNN_WEIGHT_OPSI'
AFTER_PCA = BASE_DIR / 'data' / 'raw' / 'after_pca.txt'
PARATOPE = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'


def score(df):
    """Immunogenicity scores from DeepImmuno's released checkpoint.

    `df` needs 'peptide' and 'HLA' columns; the returned array is in row order.
    """
    from src.models.cnn_model import arsitekturCNN
    from src.data.processing import (HLA_Dictionary, dictionary, construct_aaindex,
                                     peptide_iterate, hla_iterate)
    after_pca = np.loadtxt(AFTER_PCA)
    hla_dic = HLA_Dictionary(pd.read_csv(PARATOPE, sep='\t'))
    dic_inv = dictionary(list(hla_dic.keys()))
    ori = df[['peptide', 'HLA']].copy()
    ori['immunogenicity'] = '0'
    ds = construct_aaindex(ori, hla_dic, after_pca, dic_inv)
    model = arsitekturCNN()
    model.load_weights(str(PUBLISHED_WEIGHTS) + '/')
    return model.predict([peptide_iterate(ds), hla_iterate(ds)],
                         batch_size=512, verbose=0).ravel()


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    from src.data.dataset import OUT_PATH
    test = pd.read_csv(OUT_PATH).query("split == 'test'").reset_index(drop=True)
    s = score(test)
    print(f'DeepImmuno (published checkpoint) on {len(test)} held-out pairs: '
          f'AUC {roc_auc_score(test["label"], s):.3f}')
