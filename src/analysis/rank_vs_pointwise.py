"""
Apples-to-apples: does the study-stratified ranking objective beat the pointwise one
at the task that matters -- ordering peptides inside a study the model never saw?

The two models are scored on *exactly the same* evaluation groups (the study-grouped
held-out split), because within-study AUC averaged over a different number of groups is
not comparable. Both models are trained without seeing any record from those studies.

Run:  python -m src.analysis.rank_vs_pointwise
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.dataset import OUT_PATH
from src.data.features import get_features, FEATURE_COLS
from src.analysis.feature_ablation import encoding_matrix
from src.analysis.leakage import oof_predictions
from src.analysis.within_study import per_study_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
MIN_GROUP = 20
SEED = 42


def group_aucs(df, col, by_allele, min_n=MIN_GROUP):
    """Per-group AUCs keyed so two models can be compared on identical groups."""
    keys = ['reference_id', 'HLA'] if by_allele else ['reference_id']
    rows = {}
    for key, g in df.groupby(keys):
        if len(g) < min_n or g['label'].nunique() < 2:
            continue
        rows[key if isinstance(key, tuple) else (key,)] = (
            roc_auc_score(g['label'], g[col]), len(g))
    return rows


def paired_group_bootstrap(df, col_a, col_b, by_allele, n=2000, seed=SEED):
    """Resample within each shared group, recompute both models' weighted mean AUC."""
    keys = ['reference_id', 'HLA'] if by_allele else ['reference_id']
    groups = [g for _, g in df.groupby(keys)
              if len(g) >= MIN_GROUP and g['label'].nunique() > 1]
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        a_s, b_s, w = [], [], []
        for g in groups:
            idx = rng.integers(0, len(g), len(g))
            y = g['label'].values[idx]
            if len(set(y)) < 2:
                continue
            a_s.append(roc_auc_score(y, g[col_a].values[idx]))
            b_s.append(roc_auc_score(y, g[col_b].values[idx]))
            w.append(len(g))
        if a_s:
            diffs.append(np.average(a_s, weights=w) - np.average(b_s, weights=w))
    if not diffs:
        return None
    return {'mean_diff': float(np.mean(diffs)),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            'p_gt_0': float(np.mean(np.array(diffs) > 0))}


def main():
    rank = pd.read_csv(RESULTS / 'rank_scores.csv')
    df = pd.read_csv(OUT_PATH)
    df = df.join(get_features(df['peptide']), on='peptide')

    # pointwise model, trained with study-grouped folds so each row is predicted by a
    # model that never saw its publication
    y = df['label'].values
    X = np.hstack([encoding_matrix(df), df[FEATURE_COLS].astype(float).values])
    df['pointwise'] = oof_predictions(X, y, GroupKFold(5), df['reference_id'].values)
    df = df.merge(rank[['peptide', 'HLA', 'score']].rename(columns={'score': 'ranker'}),
                  on=['peptide', 'HLA'], how='inner')

    test = df[df['study_split'] == 'test'].copy()
    out = {'n_held_out_pairs': int(len(test))}
    print(f'held-out studies: {len(test)} pairs, '
          f'{test["reference_id"].nunique()} studies, positive rate {test["label"].mean():.3f}\n')

    for by_allele in (False, True):
        tag = 'within_study_and_allele' if by_allele else 'within_study'
        a = group_aucs(test, 'ranker', by_allele)
        b = group_aucs(test, 'pointwise', by_allele)
        shared = sorted(set(a) & set(b))
        if not shared:
            print(f'{tag}: no shared evaluation groups'); continue
        wts = np.array([a[k][1] for k in shared], float)
        ra = np.array([a[k][0] for k in shared])
        rb = np.array([b[k][0] for k in shared])
        res = {
            'n_groups': len(shared), 'n_pairs': int(wts.sum()),
            'ranker_macro': float(ra.mean()), 'pointwise_macro': float(rb.mean()),
            'ranker_weighted': float(np.average(ra, weights=wts)),
            'pointwise_weighted': float(np.average(rb, weights=wts)),
            'ranker_groups_above_chance': float((ra > 0.5).mean()),
            'pointwise_groups_above_chance': float((rb > 0.5).mean()),
        }
        res['bootstrap'] = paired_group_bootstrap(test, 'ranker', 'pointwise', by_allele)
        out[tag] = res
        print(f'--- {tag}: {len(shared)} shared groups, {int(wts.sum())} pairs ---')
        print(f'  ranker     macro {res["ranker_macro"]:.3f}  weighted {res["ranker_weighted"]:.3f}'
              f'  above chance {res["ranker_groups_above_chance"]:.0%}')
        print(f'  pointwise  macro {res["pointwise_macro"]:.3f}  weighted {res["pointwise_weighted"]:.3f}'
              f'  above chance {res["pointwise_groups_above_chance"]:.0%}')
        if res['bootstrap']:
            bs = res['bootstrap']
            verdict = 'significant' if bs['ci95'][0] > 0 else 'NOT significant'
            print(f'  ranker - pointwise: {bs["mean_diff"]:+.3f} '
                  f'CI [{bs["ci95"][0]:+.3f}, {bs["ci95"][1]:+.3f}]  {verdict}')
        print()

    for col in ('ranker', 'pointwise'):
        out[f'{col}_pooled_auc'] = float(roc_auc_score(test['label'], test[col]))
    print(f'pooled AUC on held-out studies: ranker {out["ranker_pooled_auc"]:.3f}  '
          f'pointwise {out["pointwise_pooled_auc"]:.3f}')

    (RESULTS / 'rank_vs_pointwise.json').write_text(json.dumps(out, indent=2, default=str))
    print('saved', RESULTS / 'rank_vs_pointwise.json')


if __name__ == '__main__':
    main()
