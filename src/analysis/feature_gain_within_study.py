"""
H4: does the feature gain survive study-grouped evaluation, or was it a study artifact?

src/analysis/feature_ablation.py reported +0.034 within-allele from the biochemical and
self-similarity features. That was measured on peptide-grouped folds. Since then,
`self_exact` turned out to separate studies rather than peptides (within-study AUC 0.483,
0% of studies above chance), so the whole feature gain is suspect.

This compares two models that differ only in whether the feature vector is supplied,
both trained with study-grouped folds and scored out-of-fold, and both evaluated
within study and within study+allele -- bootstrapping over studies, which is the unit
of correlation.

Run (after both OOF files exist):
  python -m src.analysis.feature_gain_within_study
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.dataset import OUT_PATH

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
MIN_GROUP = 10
N_BOOT = 4000
SEED = 42
VARIANTS = {'with_features': 'oof_study_pointwise.npy',
            'without_features': 'oof_study_pointwise_nofeat.npy'}


def group_aucs(df, col, by_allele):
    keys = ['reference_id', 'HLA'] if by_allele else ['reference_id']
    out = {}
    for key, g in df.groupby(keys):
        if len(g) < MIN_GROUP or g['label'].nunique() < 2 or g[col].isna().any():
            continue
        out[key if isinstance(key, tuple) else (key,)] = roc_auc_score(g['label'], g[col])
    return out


def main():
    df = pd.read_csv(OUT_PATH)
    missing = []
    for name, fn in VARIANTS.items():
        p = RESULTS / fn
        if not p.exists():
            missing.append(fn)
        else:
            df[name] = np.load(p)
    if missing:
        raise SystemExit(f'missing OOF files: {missing} -- run src.analysis.robustness first')

    out = {}
    rng = np.random.default_rng(SEED)
    for tag, by_allele in [('within_study', False), ('within_study_and_allele', True)]:
        a = group_aucs(df, 'with_features', by_allele)
        b = group_aucs(df, 'without_features', by_allele)
        shared = sorted(set(a) & set(b))
        if len(shared) < 5:
            print(f'{tag}: too few shared groups'); continue
        A = np.array([a[k] for k in shared])
        B = np.array([b[k] for k in shared])
        diffs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(A), len(A))
            diffs.append(A[i].mean() - B[i].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        res = {'n_groups': len(shared),
               'with_features': float(A.mean()), 'without_features': float(B.mean()),
               'gain': float(np.mean(diffs)), 'ci95': [float(lo), float(hi)],
               'significant': bool(lo > 0)}
        out[tag] = res
        print(f'=== {tag} ({len(shared)} groups) ===')
        print(f'  with features    {A.mean():.3f}')
        print(f'  without features {B.mean():.3f}')
        verdict = 'SIGNIFICANT' if lo > 0 else ('SIGNIFICANTLY WORSE' if hi < 0 else 'not significant')
        print(f'  gain {np.mean(diffs):+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}\n')

    # for reference: the peptide-grouped number that originally motivated the features
    out['note'] = ('feature_ablation.py reported +0.034 within-allele on peptide-grouped '
                   'folds; this script re-tests that under study-grouped evaluation')
    (RESULTS / 'feature_gain_within_study.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'feature_gain_within_study.json')


if __name__ == '__main__':
    main()
