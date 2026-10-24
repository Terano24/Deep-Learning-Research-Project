"""
Is the measured reproducibility ceiling (AUC 0.637) an artifact of how we measured it?

Three threats, each tested here.

H2  Selection bias. Pairs get retested for a reason. If laboratories preferentially
    repeat borderline or disputed peptides, the repeat-tested subset would disagree more
    than a typical pair and the ceiling would be too low. Tested by stratifying on how
    many studies tested a pair, and by comparing repeat-tested pairs with singly-tested
    ones on observable properties.

H3  Aggregation rule. The headline used each study's majority outcome and dropped
    within-study ties. Alternative rules are evaluated here: any-positive, ties-to-
    positive, ties-to-negative, and raw record pairs with no aggregation at all.

H5  Single-slice dependence. The headline pooled every assay type. Here the ceiling is
    recomputed per assay family and per allele group, as internal replication.

Run:  python -m src.analysis.ceiling_robustness
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
RAW = BASE_DIR / 'data' / 'raw' / 'iedb_tcell_mhc1.csv'
HLA = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'
SEED = 42
N_DRAWS = 200


def load():
    raw = pd.read_csv(RAW)
    known = set(pd.read_csv(HLA, sep='\t')['HLA'])
    raw = raw.dropna(subset=['linear_sequence', 'mhc_restriction', 'qualitative_measure',
                             'reference_id'])
    raw = raw[raw['mhc_restriction'].str.match(r'^HLA-[ABC]\*\d\d:\d\d$')]
    raw['HLA'] = raw['mhc_restriction'].str.replace(':', '', regex=False)
    raw = raw[raw['HLA'].isin(known)]
    raw = raw[raw['linear_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]{9,10}$')]
    raw['label'] = raw['qualitative_measure'].str.startswith('Positive').astype(int)
    raw['pair'] = raw['linear_sequence'] + '|' + raw['HLA']
    return raw


def study_labels(raw, rule='majority'):
    g = raw.groupby(['pair', 'reference_id'])['label'].agg(['mean', 'size']).reset_index()
    g = g.rename(columns={'mean': 'frac', 'size': 'n'})
    if rule == 'majority':
        g = g[g['frac'] != 0.5]
        g['label'] = (g['frac'] > 0.5).astype(int)
    elif rule == 'any_positive':
        g['label'] = (g['frac'] > 0).astype(int)
    elif rule == 'ties_positive':
        g['label'] = (g['frac'] >= 0.5).astype(int)
    elif rule == 'ties_negative':
        g['label'] = (g['frac'] > 0.5).astype(int)
    else:
        raise ValueError(rule)
    return g[['pair', 'reference_id', 'label']]


def _as_ragged(sub, key='pair', val='label'):
    """One numpy array of labels per pair, built once so the draw loop stays in numpy."""
    sub = sub.sort_values(key)
    labels = sub[val].to_numpy()
    _, starts = np.unique(sub[key].to_numpy(), return_index=True)
    starts = np.sort(starts)
    bounds = np.append(starts, len(labels))
    return [labels[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def _draw_stats(groups, n_draws, rng):
    """Repeatedly pick two observations per pair; return agreement/kappa/AUC per draw."""
    sizes = np.array([len(g) for g in groups])
    flat = np.concatenate(groups)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ag, kp, au = [], [], []
    for _ in range(n_draws):
        i = (rng.random(len(groups)) * sizes).astype(int)
        j = (rng.random(len(groups)) * (sizes - 1)).astype(int)
        j += (j >= i)                      # pick a different index without a loop
        a = flat[offs + i]
        b = flat[offs + j]
        ag.append(float((a == b).mean()))
        if a.min() != a.max() and b.min() != b.max():
            kp.append(cohen_kappa_score(a, b))
            au.append(roc_auc_score(b, a))
    return ag, kp, au


def reproducibility(sl, n_draws=N_DRAWS, seed=SEED, min_studies=2):
    counts = sl.groupby('pair')['reference_id'].nunique()
    keep = counts[counts >= min_studies].index
    sub = sl[sl['pair'].isin(keep)]
    groups = [g for g in _as_ragged(sub) if len(g) >= 2]
    if len(groups) < 50:
        return None
    ag, kp, au = _draw_stats(groups, n_draws, np.random.default_rng(seed))
    if not au:
        return None
    return {'n_pairs': len(groups), 'agreement': float(np.mean(ag)),
            'kappa': float(np.mean(kp)),
            'auc': float(np.mean(au)),
            'auc_ci95': [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
            'positive_rate': float(np.mean([g.mean() for g in groups]))}


def show(tag, r):
    import sys
    if r is None:
        print(f'  {tag:>30}: too few pairs', flush=True)
        return
    print(f'  {tag:>30}: n={r["n_pairs"]:>5}  AUC {r["auc"]:.3f} '
          f'[{r["auc_ci95"][0]:.3f}, {r["auc_ci95"][1]:.3f}]  kappa {r["kappa"]:.3f}  '
          f'agree {r["agreement"]:.3f}', flush=True)


def main():
    raw = load()
    out = {}
    base = study_labels(raw, 'majority')
    print(f'records {len(raw):,} | pairs {raw["pair"].nunique():,}\n')

    # ---------- H3: aggregation rule ----------
    print('=== H3: sensitivity to the aggregation rule ===', flush=True)
    out['H3_rules'] = {}
    for rule in ['majority', 'any_positive', 'ties_positive', 'ties_negative']:
        r = reproducibility(study_labels(raw, rule))
        out['H3_rules'][rule] = r
        show(rule, r)
    # no aggregation at all: two raw records drawn from two different studies.
    # Per-pair, per-study label arrays are built once; the draw loop then touches
    # only numpy, instead of doing ~1M pandas selections.
    rng = np.random.default_rng(SEED)
    recs = raw[['pair', 'reference_id', 'label']]
    multi = recs.groupby('pair')['reference_id'].nunique()
    recs = recs[recs['pair'].isin(multi[multi >= 2].index)]
    by_pair = [[g['label'].to_numpy() for _, g in p.groupby('reference_id')]
               for _, p in recs.groupby('pair')]
    by_pair = [p for p in by_pair if len(p) >= 2]
    ag, au = [], []
    for _ in range(N_DRAWS):
        a = np.empty(len(by_pair), int)
        b = np.empty(len(by_pair), int)
        for k, studies in enumerate(by_pair):
            i = int(rng.random() * len(studies))
            j = int(rng.random() * (len(studies) - 1))
            j += (j >= i)
            si, sj = studies[i], studies[j]
            a[k] = si[int(rng.random() * len(si))]
            b[k] = sj[int(rng.random() * len(sj))]
        ag.append((a == b).mean())
        if a.min() != a.max() and b.min() != b.max():
            au.append(roc_auc_score(b, a))
    out['H3_rules']['raw_records_no_aggregation'] = {
        'n_pairs': len(by_pair), 'agreement': float(np.mean(ag)), 'auc': float(np.mean(au)),
        'auc_ci95': [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))]}
    print(f'  {"raw records (no aggregation)":>30}: n={len(by_pair):>5}  '
          f'AUC {np.mean(au):.3f}  agree {np.mean(ag):.3f}')

    # ---------- H2: selection bias ----------
    print('\n=== H2: does reproducibility depend on how often a pair was tested? ===')
    counts = base.groupby('pair')['reference_id'].nunique()
    out['H2_by_n_studies'] = {}
    for lo, hi, tag in [(2, 2, 'exactly 2 studies'), (3, 3, 'exactly 3 studies'),
                        (4, 100, '4+ studies')]:
        sel = counts[(counts >= lo) & (counts <= hi)].index
        r = reproducibility(base[base['pair'].isin(sel)])
        out['H2_by_n_studies'][tag] = r
        show(tag, r)

    print('\n  are repeat-tested pairs unusual? (observable properties)')
    pair_stats = raw.groupby('pair').agg(n_rec=('label', 'size'),
                                         n_study=('reference_id', 'nunique'),
                                         pos=('label', 'mean')).reset_index()
    single = pair_stats[pair_stats['n_study'] == 1]
    repeat = pair_stats[pair_stats['n_study'] >= 2]
    out['H2_selection'] = {
        'n_single_study_pairs': int(len(single)), 'n_repeat_pairs': int(len(repeat)),
        'positive_rate_single': float(single['pos'].mean()),
        'positive_rate_repeat': float(repeat['pos'].mean())}
    print(f'    singly-tested pairs : n={len(single):>6}  mean positive fraction {single["pos"].mean():.3f}')
    print(f'    repeat-tested pairs : n={len(repeat):>6}  mean positive fraction {repeat["pos"].mean():.3f}')

    # ---------- H5: internal replication ----------
    print('\n=== H5: ceiling recomputed within slices ===')
    out['H5_slices'] = {}
    raw['assay_family'] = raw['assay_names'].fillna('unknown').astype(str).str.split('|').str[-1]
    top_assays = raw['assay_family'].value_counts().head(3).index
    for a in top_assays:
        sub = raw[raw['assay_family'] == a]
        r = reproducibility(study_labels(sub, 'majority'))
        out['H5_slices'][f'assay::{a[:28]}'] = r
        show(f'assay: {a[:22]}', r)
    for grp in ['A', 'B', 'C']:
        sub = raw[raw['HLA'].str.startswith(f'HLA-{grp}')]
        r = reproducibility(study_labels(sub, 'majority'))
        out['H5_slices'][f'HLA-{grp}'] = r
        show(f'HLA-{grp}*', r)

    (RESULTS / 'ceiling_robustness.json').write_text(json.dumps(out, indent=2))
    print('\nsaved', RESULTS / 'ceiling_robustness.json')


if __name__ == '__main__':
    main()
