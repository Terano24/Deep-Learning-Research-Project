"""
Train and evaluate the peptide-HLA CNN with a real pipeline:
  * peptide-grouped 10-fold CV: a fresh model is trained for every fold
  * a final model trained on all training folds and scored once on the held-out test set
  * baselines (allele prior, gradient boosting) on the same split
  * metrics for the full test set and for the lung-cancer subset (bootstrap CIs)

Run from the project root:  python -m src.models.train
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from src.data.dataset import build_dataset, OUT_PATH
from src.data.processing import HLA_Dictionary, peptide_data_aaindex, hla_data_aaindex, dictionary
from src.models.cnn_model import arsitekturCNN

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEED = 42


def encode(df):
    after_pca = np.loadtxt(BASE_DIR / 'data' / 'raw' / 'after_pca.txt')
    hla_dic = HLA_Dictionary(pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt', sep='\t'))
    dic_inventory = dictionary(list(hla_dic.keys()))
    pep_enc = {p: peptide_data_aaindex(p, after_pca) for p in df['peptide'].unique()}
    hla_enc = {h: hla_data_aaindex(hla_dic, h, after_pca, dic_inventory) for h in df['HLA'].unique()}
    x_pep = np.stack([pep_enc[p] for p in df['peptide']]).astype('float32')
    x_hla = np.stack([hla_enc[h] for h in df['HLA']]).astype('float32')
    return x_pep, x_hla


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        'n': int(len(y)), 'n_pos': int(y.sum()),
        'auc': float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        'pr_auc': float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
        'mcc': float(matthews_corrcoef(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'specificity': tn / (tn + fp) if (tn + fp) else None,
        'accuracy': float((pred == y).mean()),
    }


def bootstrap_auc(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def fit_cnn(x_pep, x_hla, y, val=None, epochs=30, patience=4, batch=128):
    tf.keras.utils.set_random_seed(SEED)
    model = arsitekturCNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    cw = {0: 1.0, 1: float((y == 0).sum() / max((y == 1).sum(), 1))}
    callbacks = []
    if val is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True))
    hist = model.fit([x_pep, x_hla], y, epochs=epochs, batch_size=batch, class_weight=cw,
                     validation_data=([val[0], val[1]], val[2]) if val is not None else None,
                     callbacks=callbacks, verbose=0)
    best = int(np.argmin(hist.history['val_loss'])) + 1 if val is not None else epochs
    return model, best


def predict(model, x_pep, x_hla):
    return model.predict([x_pep, x_hla], batch_size=512, verbose=0).ravel()


def gbm_features(df, x_pep):
    alleles = pd.Categorical(df['HLA'], categories=sorted(df['HLA'].unique())).codes
    return np.column_stack([x_pep.reshape(len(df), -1), alleles])


def main(args):
    df = pd.read_csv(OUT_PATH) if OUT_PATH.exists() and not args.rebuild else build_dataset()
    x_pep, x_hla = encode(df)
    y = df['label'].values.astype('float32')
    train = (df['split'] == 'train').values
    test = (df['split'] == 'test').values
    results = {'config': vars(args)}

    # ---- 10-fold CV, fresh model per fold -------------------------------------------------
    folds = range(args.folds)
    oof = np.full(len(df), np.nan)
    fold_metrics, best_epochs = [], []
    for k in folds:
        va = train & (df['fold'].values == k)
        tr = train & (df['fold'].values != k)
        model, best = fit_cnn(x_pep[tr], x_hla[tr], y[tr], (x_pep[va], x_hla[va], y[va]),
                              epochs=args.epochs)
        oof[va] = predict(model, x_pep[va], x_hla[va])
        m = metrics(y[va].astype(int), oof[va])
        m.update(fold=k, best_epoch=best)
        fold_metrics.append(m)
        best_epochs.append(best)
        print(f'fold {k}: AUC {m["auc"]:.3f}  PR-AUC {m["pr_auc"]:.3f}  MCC {m["mcc"]:.3f}  best_epoch {best}')
    results['cv_folds'] = fold_metrics
    results['cv_mean'] = {k: float(np.mean([f[k] for f in fold_metrics]))
                          for k in ['auc', 'pr_auc', 'mcc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']}
    results['cv_std'] = {k: float(np.std([f[k] for f in fold_metrics])) for k in ['auc', 'pr_auc', 'mcc']}

    # ---- final model on all training folds -> held-out test ------------------------------
    final_epochs = int(np.median(best_epochs))
    model, _ = fit_cnn(x_pep[train], x_hla[train], y[train], epochs=final_epochs)
    out_dir = BASE_DIR / 'models' / 'CNN_RETRAINED'
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out_dir / 'weights'))
    p_cnn = predict(model, x_pep[test], x_hla[test])
    yt = y[test].astype(int)
    lung = df.loc[test, 'lung_cancer'].values

    # ---- baselines -----------------------------------------------------------------------
    prior = df[train].groupby('HLA')['label'].mean()
    p_prior = df.loc[test, 'HLA'].map(prior).fillna(df.loc[train, 'label'].mean()).values
    gbm = HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')
    feats = gbm_features(df, x_pep)
    gbm.fit(feats[train], y[train])
    p_gbm = gbm.predict_proba(feats[test])[:, 1]

    models = {'cnn': p_cnn, 'gbm_baseline': p_gbm, 'allele_prior_baseline': p_prior}
    results['test'] = {'final_epochs': final_epochs}
    for name, p in models.items():
        results['test'][name] = {'all': metrics(yt, p), 'lung_cancer': metrics(yt[lung], p[lung])}
        if name == 'cnn':
            results['test'][name]['all']['auc_ci95'] = bootstrap_auc(yt, p)
            results['test'][name]['lung_cancer']['auc_ci95'] = bootstrap_auc(yt[lung], p[lung])
        a, l = results['test'][name]['all'], results['test'][name]['lung_cancer']
        print(f'{name:>22}  test AUC {a["auc"]:.3f} MCC {a["mcc"]:.3f} | lung AUC {l["auc"]:.3f} MCC {l["mcc"]:.3f}')

    out = BASE_DIR / 'results'
    out.mkdir(exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    pred_df = df[test][['peptide', 'HLA', 'label', 'lung_cancer']].copy()
    for name, p in models.items():
        pred_df[name] = p
    pred_df.to_csv(out / 'test_predictions.csv', index=False)
    if args.folds == 10:
        cv_df = df[train][['peptide', 'HLA', 'label', 'lung_cancer', 'fold']].copy()
        cv_df['cnn_oof'] = oof[train]
        cv_df.to_csv(out / 'cv_predictions.csv', index=False)
    print('saved results to', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=10, help='number of CV folds to run (max 10)')
    ap.add_argument('--epochs', type=int, default=30, help='max epochs (early stopping on val loss)')
    ap.add_argument('--rebuild', action='store_true', help='rebuild data/processed/dataset.csv')
    main(ap.parse_args())
