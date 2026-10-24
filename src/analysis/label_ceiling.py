"""
How well can ANY model do? An empirical ceiling from assay reproducibility.

Throughout this project we have argued that peptide+HLA alone cannot determine
immunogenicity, so there must be a ceiling below 1.0. That was an assertion. Here it is
measured.

IEDB contains peptide-HLA pairs assayed more than once, often by different laboratories.
If one laboratory's result cannot predict another's on the same pair, then no model
reading only the peptide and the allele can do better: a repeat experiment is the most
informative possible "predictor", since it observes the actual biology.

Procedure: restrict to pairs tested in at least two distinct publications, take each
study's majority outcome for that pair, draw two studies at random, and ask how well
study A's label predicts study B's. Reported as agreement, Cohen's kappa and AUC, with
a bootstrap interval over pairs. The AUC is the ceiling estimate.

Run:  python -m src.analysis.label_ceiling
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
N_DRAWS = 200          # random study pairings, to avoid depending on one arbitrary draw


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
    return raw


def study_level(raw):
    """One majority outcome per (pair, study); drop within-study ties."""
    g = (raw.groupby(['linear_sequence', 'HLA', 'reference_id'])['label']
         .mean().reset_index(name='frac'))
    g = g[g['frac'] != 0.5]
    g['label'] = (g['frac'] > 0.5).astype(int)
    return g


def main():
    raw = load()
    sl = study_level(raw)
    counts = sl.groupby(['linear_sequence', 'HLA'])['reference_id'].nunique()
    multi = counts[counts >= 2].index
    sl = sl.set_index(['linear_sequence', 'HLA']).loc[multi].reset_index()
    print(f'usable records          : {len(raw):,}')
    print(f'pair-study observations : {len(sl):,}')
    print(f'pairs in >=2 studies    : {len(multi):,}\n')

    rng = np.random.default_rng(SEED)
    groups = [g for _, g in sl.groupby(['linear_sequence', 'HLA'])]
    agrees, kappas, aucs, pos_rates = [], [], [], []
    for _ in range(N_DRAWS):
        a, b = [], []
        for g in groups:
            i, j = rng.choice(len(g), 2, replace=False)
            a.append(g['label'].iat[i])
            b.append(g['label'].iat[j])
        a, b = np.array(a), np.array(b)
        agrees.append((a == b).mean())
        if len(set(a)) > 1 and len(set(b)) > 1:
            kappas.append(cohen_kappa_score(a, b))
            aucs.append(roc_auc_score(b, a))
        pos_rates.append(b.mean())

    def ci(v):
        return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    ag, ag_lo, ag_hi = ci(agrees)
    kp, kp_lo, kp_hi = ci(kappas)
    au, au_lo, au_hi = ci(aucs)
    print('Does one study\'s result predict another study\'s, on the same peptide-HLA pair?')
    print(f'  agreement      {ag:.3f}  [{ag_lo:.3f}, {ag_hi:.3f}]')
    print(f'  Cohen\'s kappa  {kp:.3f}  [{kp_lo:.3f}, {kp_hi:.3f}]   (0 = chance, 1 = perfect)')
    print(f'  AUC (CEILING)  {au:.3f}  [{au_lo:.3f}, {au_hi:.3f}]')
    print(f'  positive rate  {np.mean(pos_rates):.3f}')

    out = {'n_records': int(len(raw)), 'n_pairs_multi_study': int(len(multi)),
           'n_draws': N_DRAWS,
           'agreement': {'mean': ag, 'ci95': [ag_lo, ag_hi]},
           'cohens_kappa': {'mean': kp, 'ci95': [kp_lo, kp_hi]},
           'reproducibility_auc_ceiling': {'mean': au, 'ci95': [au_lo, au_hi]},
           'positive_rate': float(np.mean(pos_rates))}

    print('\nFor comparison, our best honest model performance:')
    print(f'  within-study + within-allele, held-out studies: 0.557 [0.526, 0.588]')
    frac = 0.557 / au if au > 0.5 else float('nan')
    print(f'  that is {frac:.0%} of the measured reproducibility ceiling')
    out['model_within_study_auc'] = 0.557
    out['fraction_of_ceiling'] = float(frac)

    (RESULTS / 'label_ceiling.json').write_text(json.dumps(out, indent=2))
    print('\nsaved', RESULTS / 'label_ceiling.json')


if __name__ == '__main__':
    main()
