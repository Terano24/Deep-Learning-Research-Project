"""
Confound analysis for the immunogenicity models.

The allele-prior baseline reaches ~0.75 AUC using *no peptide information at all*,
because IEDB study design correlates allele identity with assay outcome. Any headline
AUC on this data is therefore inflated. This script asks the harder question:

  does the model rank peptides correctly *within a single HLA allele*?

Reported:
  * within-allele AUC, computed per allele then averaged (weighted by n and unweighted)
  * per-allele breakdown for the best-represented alleles
  * AUC on peptide lengths 9 vs 10
  * DeLong-free bootstrap CI on the paired difference between two models

Run:  python -m src.analysis.confounds
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
MIN_PER_ALLELE = 30


def within_allele_auc(df, score_col, min_n=MIN_PER_ALLELE):
    """AUC computed inside each allele, so allele-level prior signal cannot contribute."""
    rows = []
    for allele, g in df.groupby('HLA'):
        if len(g) < min_n or g['label'].nunique() < 2:
            continue
        rows.append({'HLA': allele, 'n': len(g), 'n_pos': int(g['label'].sum()),
                     'auc': roc_auc_score(g['label'], g[score_col])})
    per = pd.DataFrame(rows).sort_values('n', ascending=False)
    if per.empty:
        return per, {}
    summary = {
        'n_alleles': len(per),
        'n_pairs_covered': int(per['n'].sum()),
        'macro_auc': float(per['auc'].mean()),
        'weighted_auc': float(np.average(per['auc'], weights=per['n'])),
    }
    return per, summary


def paired_bootstrap(df, col_a, col_b, stratified, n=2000, seed=0):
    """Bootstrap the difference (a - b). Resamples within allele when stratified."""
    rng = np.random.default_rng(seed)
    diffs = []
    groups = [g for _, g in df.groupby('HLA')] if stratified else [df]
    groups = [g for g in groups if len(g) >= MIN_PER_ALLELE and g['label'].nunique() > 1]
    if not groups:
        return None
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


def load_predictions():
    """Join the old-CNN/baseline predictions with the new model's on the same test split."""
    base = pd.read_csv(RESULTS / 'test_predictions.csv')
    new_path = RESULTS / 'improve_test_predictions_pretrained_lung.csv'
    if new_path.exists():
        new = pd.read_csv(new_path).rename(columns={'pred': 'pocket_attention'})
        base = base.merge(new[['peptide', 'HLA', 'pocket_attention']], on=['peptide', 'HLA'], how='inner')
    base['length'] = base['peptide'].str.len()
    return base


def main():
    df = load_predictions()
    models = [c for c in ['pocket_attention', 'cnn', 'gbm_baseline', 'allele_prior_baseline'] if c in df]
    out = {'n_test_pairs': len(df), 'models': models}
    print(f'test pairs: {len(df)}   models: {models}\n')

    print('=== overall AUC (allele prior CAN contribute) ===')
    out['overall_auc'] = {}
    for m in models:
        a = roc_auc_score(df['label'], df[m])
        out['overall_auc'][m] = float(a)
        print(f'  {m:>22}: {a:.3f}')

    print('\n=== within-allele AUC (allele prior CANNOT contribute) ===')
    out['within_allele'] = {}
    per_allele = {}
    for m in models:
        per, summary = within_allele_auc(df, m)
        out['within_allele'][m] = summary
        per_allele[m] = per
        if summary:
            print(f'  {m:>22}: macro {summary["macro_auc"]:.3f}  weighted {summary["weighted_auc"]:.3f}'
                  f'  ({summary["n_alleles"]} alleles, {summary["n_pairs_covered"]} pairs)')

    if 'pocket_attention' in per_allele and not per_allele['pocket_attention'].empty:
        print('\n=== per-allele detail (new model) ===')
        detail = per_allele['pocket_attention'].copy()
        for m in models:
            if m != 'pocket_attention' and not per_allele[m].empty:
                detail = detail.merge(per_allele[m][['HLA', 'auc']].rename(columns={'auc': m}), on='HLA', how='left')
        detail = detail.rename(columns={'auc': 'pocket_attention'})
        print(detail.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        detail.to_csv(RESULTS / 'per_allele_auc.csv', index=False)
        out['per_allele_file'] = 'results/per_allele_auc.csv'

    print('\n=== paired bootstrap: new model vs others ===')
    out['comparisons'] = {}
    for m in models:
        if m == 'pocket_attention' or 'pocket_attention' not in df:
            continue
        for label, strat in [('overall', False), ('within_allele', True)]:
            r = paired_bootstrap(df, 'pocket_attention', m, stratified=strat)
            if r:
                out['comparisons'][f'pocket_attention_vs_{m}_{label}'] = r
                sig = 'significant' if r['ci95'][0] > 0 else 'not significant'
                print(f'  vs {m:>22} [{label:>13}]: {r["mean_diff"]:+.3f} '
                      f'CI [{r["ci95"][0]:+.3f}, {r["ci95"][1]:+.3f}]  {sig}')

    print('\n=== by peptide length ===')
    out['by_length'] = {}
    for n in sorted(df['length'].unique()):
        g = df[df['length'] == n]
        row = {m: float(roc_auc_score(g['label'], g[m])) for m in models if g['label'].nunique() > 1}
        row['n'] = len(g)
        out['by_length'][int(n)] = row
        print(f'  length {n} (n={len(g)}): ' + '  '.join(f'{m} {row[m]:.3f}' for m in models if m in row))

    print('\n=== lung-cancer subset, within-allele ===')
    lung = df[df['lung_cancer']]
    out['lung_within_allele'] = {}
    for m in models:
        per, summary = within_allele_auc(lung, m, min_n=20)
        out['lung_within_allele'][m] = summary
        if summary:
            print(f'  {m:>22}: macro {summary["macro_auc"]:.3f} ({summary["n_alleles"]} alleles, '
                  f'{summary["n_pairs_covered"]} pairs)')
        else:
            print(f'  {m:>22}: too few per-allele samples to measure')

    (RESULTS / 'confound_analysis.json').write_text(json.dumps(out, indent=2))
    print('\nsaved', RESULTS / 'confound_analysis.json')


if __name__ == '__main__':
    main()
