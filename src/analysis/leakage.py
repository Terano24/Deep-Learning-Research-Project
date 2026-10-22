"""
How much of the measured performance survives a harder split?

IEDB records cluster by publication: one paper contributes many peptides tested with a
shared cohort, cell line and protocol. Splitting by peptide keeps the *same peptide* out
of training but still lets a model see its study-mates, so some of the apparent skill may
be study fingerprinting rather than transferable biology.

This compares three evaluation regimes with the same model class (gradient boosting, for
speed) on the same data:

  random        rows split at random            -- what the original paper effectively did
  peptide       grouped by peptide              -- our headline split
  study         grouped by source publication   -- the strict split

Each is scored overall and within-allele. Run:  python -m src.analysis.leakage
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from src.data.dataset import OUT_PATH
from src.data.features import get_features, FEATURE_COLS
from src.analysis.confounds import within_allele_auc
from src.analysis.feature_ablation import encoding_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
SEED = 42
N_SPLITS = 5


def oof_predictions(X, y, splitter, groups=None):
    oof = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, va in it:
        m = HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def main():
    df = pd.read_csv(OUT_PATH)
    df = df.join(get_features(df['peptide']), on='peptide')
    y = df['label'].values
    X = np.hstack([encoding_matrix(df), df[FEATURE_COLS].astype(float).values])

    regimes = {
        'random': (KFold(N_SPLITS, shuffle=True, random_state=SEED), None),
        'peptide': (GroupKFold(N_SPLITS), df['peptide'].values),
        'study': (GroupKFold(N_SPLITS), df['reference_id'].values),
    }

    out = {'n_pairs': len(df), 'n_studies': int(df['reference_id'].nunique())}
    print(f"{len(df)} pairs from {out['n_studies']} publications\n")
    print(f'{"regime":>10}  {"overall AUC":>12}  {"within-allele":>14}  {"lung AUC":>9}')
    for name, (splitter, groups) in regimes.items():
        oof = oof_predictions(X, y, splitter, groups)
        d = df.copy()
        d['score'] = oof
        d = d[~d['score'].isna()]
        overall = roc_auc_score(d['label'], d['score'])
        _, wa = within_allele_auc(d, 'score')
        lung = d[d['lung_cancer']]
        lung_auc = roc_auc_score(lung['label'], lung['score']) if lung['label'].nunique() > 1 else float('nan')
        out[name] = {'overall_auc': float(overall), 'within_allele': wa, 'lung_auc': float(lung_auc)}
        print(f'{name:>10}  {overall:>12.3f}  {wa["weighted_auc"]:>14.3f}  {lung_auc:>9.3f}')

    drop_o = out['random']['overall_auc'] - out['study']['overall_auc']
    drop_w = out['random']['within_allele']['weighted_auc'] - out['study']['within_allele']['weighted_auc']
    out['random_minus_study'] = {'overall': float(drop_o), 'within_allele': float(drop_w)}
    print(f'\nrandom-split optimism vs study-split: overall {drop_o:+.3f}, within-allele {drop_w:+.3f}')

    # how concentrated is the lung evidence?
    lung_raw = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'iedb_tcell_mhc1.csv')
    lung_raw = lung_raw[lung_raw['lung_cancer']]
    counts = lung_raw['reference_id'].value_counts()
    out['lung_study_concentration'] = {
        'n_records': int(len(lung_raw)), 'n_studies': int(len(counts)),
        'top3_share': float(counts.head(3).sum() / len(lung_raw)),
    }
    print(f"\nlung records: {len(lung_raw)} from {len(counts)} studies; "
          f"top 3 studies = {counts.head(3).sum() / len(lung_raw):.0%} of records")

    (RESULTS / 'leakage_analysis.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'leakage_analysis.json')


if __name__ == '__main__':
    main()
