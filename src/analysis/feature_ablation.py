"""
Does adding self-similarity / biochemical features improve honest ranking?

Uses gradient boosting (fast, and the strongest baseline from the confound analysis)
on the training folds only, scoring out-of-fold predictions with *within-allele* AUC
so that allele-prior signal cannot inflate the result. The test split is never touched.

Run:  python -m src.analysis.feature_ablation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data.dataset import OUT_PATH
from src.data.features import get_features, FEATURE_COLS
from src.analysis.confounds import within_allele_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
SEED = 42


def encoding_matrix(df):
    """The AAindex/PCA encoding the CNN uses, flattened, plus allele identity."""
    from src.data.processing import aaindex
    after_pca = np.loadtxt(BASE_DIR / 'data' / 'raw' / 'after_pca.txt')
    rows = []
    for p in df['peptide']:
        q = p[:5] + '-' + p[5:] if len(p) == 9 else p
        rows.append(aaindex(q, after_pca).ravel())
    enc = np.array(rows)
    allele = pd.Categorical(df['HLA'], categories=sorted(df['HLA'].unique())).codes[:, None]
    return np.hstack([enc, allele])


def run(df, X, y, folds):
    oof = np.full(len(df), np.nan)
    for k in sorted(set(folds)):
        if k < 0:
            continue
        va, tr = folds == k, (folds != k) & (folds >= 0)
        m = HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def main():
    df = pd.read_csv(OUT_PATH)
    df = df[df['split'] == 'train'].reset_index(drop=True)
    # get_features is indexed by *unique* peptide, so join on the key -- never by position
    feats = get_features(df['peptide'])
    df = df.join(feats, on='peptide')
    assert df[FEATURE_COLS].notna().all().all(), 'feature join left NaNs'
    y = df['label'].values
    folds = df['fold'].values
    enc = encoding_matrix(df)
    extra = df[FEATURE_COLS].astype(float).values

    sets = {
        'encoding_only': enc,
        'features_only': extra,
        'encoding_plus_features': np.hstack([enc, extra]),
    }
    out = {}
    print(f'{len(df)} training pairs, {len(set(folds[folds >= 0]))} folds\n')
    for name, X in sets.items():
        oof = run(df, X, y, folds)
        done = ~np.isnan(oof)
        d = df[done].copy()
        d['score'] = oof[done]
        overall = roc_auc_score(d['label'], d['score'])
        per, summary = within_allele_auc(d, 'score')
        lung = d[d['lung_cancer']]
        lung_auc = roc_auc_score(lung['label'], lung['score']) if lung['label'].nunique() > 1 else None
        out[name] = {'overall_auc': float(overall), 'within_allele': summary,
                     'lung_auc_pooled': float(lung_auc) if lung_auc else None,
                     'n_features': X.shape[1]}
        print(f'{name:>24} ({X.shape[1]:>3} feats): overall {overall:.3f}  '
              f'within-allele {summary["weighted_auc"]:.3f}  lung(pooled) '
              f'{lung_auc:.3f}' if lung_auc else '')

    base = out['encoding_only']['within_allele']['weighted_auc']
    plus = out['encoding_plus_features']['within_allele']['weighted_auc']
    print(f'\nwithin-allele gain from new features: {plus - base:+.4f}')
    out['within_allele_gain'] = float(plus - base)
    (RESULTS / 'feature_ablation.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'feature_ablation.json')


if __name__ == '__main__':
    main()
