"""
Does the architecture actually matter? The CNN, evaluated the way the paper evaluates.

The decision to replace the Conv2D architecture with cross-attention rested on a
+0.043 AUC margin measured on the peptide-grouped split -- the same split this paper shows
is indistinguishable from random splitting. That justification was never re-tested under
study-grouped evaluation, so it is tested here.

The CNN is the original two-branch Conv2D network (identical to DeepImmuno's) trained from
its own initialisation, never from the published checkpoint, so nothing is inherited. To
keep the comparison fair it receives the same advantages as the attention model: binding
pretraining, class-balanced weights, study-held-out early stopping, identical folds and
seed. Binding pretraining is subsampled to keep the dense AAindex tensors in memory; the
subsample size is recorded in the output.

Run:  python -m src.analysis.cnn_study_eval
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.dataset import OUT_PATH, HLA_PATH
from src.data.processing import (HLA_Dictionary, dictionary, aaindex,
                                 peptide_data_aaindex, hla_data_aaindex)
from src.models.cnn_model import arsitekturCNN
from src.analysis.robustness import study_val_split
from src.analysis.confounds import within_allele_auc
from src.analysis.within_study import per_study_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
BINDING = BASE_DIR / 'data' / 'raw' / 'iedb_mhc_binding.csv'
SEED = 42
MIN_GROUP = 10


def encoders():
    after_pca = np.loadtxt(BASE_DIR / 'data' / 'raw' / 'after_pca.txt')
    hla_tab = pd.read_csv(HLA_PATH, sep='\t')
    hla_dic = HLA_Dictionary(hla_tab)
    dic_inv = dictionary(list(hla_dic.keys()))
    return after_pca, hla_dic, dic_inv


def encode(df, after_pca, hla_dic, dic_inv):
    """Dense AAindex tensors: peptide (n,10,12,1), HLA (n,46,12,1)."""
    pep_cache, hla_cache = {}, {}
    for p in df['peptide'].unique():
        pep_cache[p] = peptide_data_aaindex(p, after_pca)
    for h in df['HLA'].unique():
        hla_cache[h] = hla_data_aaindex(hla_dic, h, after_pca, dic_inv)
    x1 = np.stack([pep_cache[p] for p in df['peptide']]).astype('float32')
    x2 = np.stack([hla_cache[h] for h in df['HLA']]).astype('float32')
    return x1, x2


def balanced(y):
    pos = max(float(y.sum()), 1.0)
    return np.where(y == 1, (len(y) - pos) / pos, 1.0).astype('float32')


def pretrain_cnn(n_max, after_pca, hla_dic, dic_inv, exclude, epochs=6):
    """Pretrain the CNN on binding/presentation data, as the attention model was."""
    if not BINDING.exists():
        print('  no binding file; CNN will train from scratch', flush=True)
        return None
    b = pd.read_csv(BINDING).dropna(subset=['linear_sequence', 'mhc_restriction',
                                            'qualitative_measure'])
    b = b[b['mhc_restriction'].str.match(r'^HLA-[ABC]\*\d\d:\d\d$')]
    b['HLA'] = b['mhc_restriction'].str.replace(':', '', regex=False)
    b = b[b['HLA'].isin(hla_dic)]
    b = b[b['linear_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]{9,10}$')]
    b = b[~b['linear_sequence'].isin(exclude)]
    b['label'] = b['qualitative_measure'].str.startswith('Positive').astype(int)
    g = b.groupby(['linear_sequence', 'HLA'])['label'].mean().reset_index()
    g = g[g['label'] != 0.5]
    g['label'] = (g['label'] > 0.5).astype(int)
    g = g.rename(columns={'linear_sequence': 'peptide'})
    if len(g) > n_max:
        g = g.sample(n_max, random_state=SEED)
    print(f'  binding pretrain pairs: {len(g)} ({g["label"].mean():.1%} positive)', flush=True)
    x1, x2 = encode(g, after_pca, hla_dic, dic_inv)
    y = g['label'].values.astype('float32')
    tf.keras.utils.set_random_seed(SEED)
    m = arsitekturCNN()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    m.fit([x1, x2], y, epochs=epochs, batch_size=256, verbose=2,
          sample_weight=balanced(y), validation_split=0.05,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                                      restore_best_weights=True)])
    w = m.get_weights()
    del x1, x2
    return w


def main(args):
    df = pd.read_csv(OUT_PATH)
    after_pca, hla_dic, dic_inv = encoders()
    print(f'{len(df)} pairs; encoding AAindex tensors...', flush=True)
    x1, x2 = encode(df, after_pca, hla_dic, dic_inv)
    y = df['label'].values.astype('float32')

    init_w = None
    if not args.no_pretrain:
        print('pretraining CNN on binding data...', flush=True)
        init_w = pretrain_cnn(args.pretrain_n, after_pca, hla_dic, dic_inv,
                              set(df['peptide']))

    folds = list(GroupKFold(args.folds).split(df, groups=df['reference_id'].values))
    oof = np.full(len(df), np.nan)
    rng = np.random.default_rng(SEED)
    for k, (tr_idx, te_idx) in enumerate(folds):
        train_mask = np.zeros(len(df), bool); train_mask[tr_idx] = True
        test_mask = np.zeros(len(df), bool); test_mask[te_idx] = True
        fit, val = study_val_split(df, train_mask, rng)
        if val.sum() == 0 or len(set(y[val])) < 2:
            fit, val = train_mask, train_mask
        tf.keras.utils.set_random_seed(SEED)
        m = arsitekturCNN()
        if init_w is not None:
            m.set_weights(init_w)
        m.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy')
        m.fit([x1[fit], x2[fit]], y[fit], epochs=args.epochs, batch_size=128, verbose=0,
              sample_weight=balanced(y[fit]),
              validation_data=([x1[val], x2[val]], y[val]),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                          restore_best_weights=True)])
        oof[test_mask] = m.predict([x1[test_mask], x2[test_mask]],
                                   batch_size=512, verbose=0).ravel()
        print(f'  cnn fold {k} done', flush=True)

    np.save(RESULTS / 'oof_study_cnn.npy', oof)
    d = df.copy(); d['score'] = oof
    d = d[~d['score'].isna()]
    _, ws = per_study_auc(d, 'score', min_n=MIN_GROUP)
    _, wsa = per_study_auc(d, 'score', min_n=MIN_GROUP, by_allele=True)
    _, wa = within_allele_auc(d, 'score', min_n=MIN_GROUP)
    out = {'pretrained': init_w is not None, 'pretrain_n': args.pretrain_n,
           'pooled_auc': float(roc_auc_score(d['label'], d['score'])),
           'within_allele': wa, 'within_study': ws, 'within_study_and_allele': wsa}
    print(f'\nCNN (study-grouped OOF): pooled {out["pooled_auc"]:.3f}  '
          f'within-allele {wa.get("weighted_auc", float("nan")):.3f}  '
          f'within-study {ws.get("macro_auc", float("nan")):.3f}  '
          f'within-study+allele {wsa.get("macro_auc", float("nan")):.3f} '
          f'({wsa.get("n_groups", 0)} groups)')

    # paired comparison against the attention model on identical groups
    att = RESULTS / 'oof_study_pointwise.npy'
    if att.exists():
        scored = ~np.isnan(oof)                      # same rows that survived into `d`
        d['att'] = np.load(att)[scored]
        d = d[d['att'].notna()]
        groups = [g for _, g in d.groupby(['reference_id', 'HLA'])
                  if len(g) >= MIN_GROUP and g['label'].nunique() > 1]
        if groups:
            A = np.array([roc_auc_score(g['label'], g['score']) for g in groups])
            B = np.array([roc_auc_score(g['label'], g['att']) for g in groups])
            rng_b = np.random.default_rng(SEED)
            diffs = []
            for _ in range(4000):
                i = rng_b.integers(0, len(A), len(A))
                diffs.append(A[i].mean() - B[i].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out['cnn_minus_attention'] = {'n_groups': len(groups),
                                          'cnn': float(A.mean()), 'attention': float(B.mean()),
                                          'diff': float(np.mean(diffs)),
                                          'ci95': [float(lo), float(hi)]}
            print(f'  CNN {A.mean():.3f} vs attention {B.mean():.3f} -> '
                  f'{np.mean(diffs):+.3f} CI [{lo:+.3f}, {hi:+.3f}] '
                  f'({"significant" if lo > 0 or hi < 0 else "not significant"})')

    (RESULTS / 'cnn_study_eval.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'cnn_study_eval.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--pretrain-n', type=int, default=150000)
    ap.add_argument('--no-pretrain', action='store_true')
    main(ap.parse_args())
