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
N_DRAWS = 1000         # random study pairings; enough that the mean is stable to 4 dp
N_BOOT = 2000          # bootstrap resamples OF THE PAIRS, for the interval


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
    groups = [g['label'].values for _, g in sl.groupby(['linear_sequence', 'HLA'])]

    # Every ordered pair of studies within each pair, flattened, so that drawing a
    # random "study A vs study B" assignment for all pairs at once is one gather.
    flat = np.concatenate([[(g[i], g[j]) for i in range(len(g)) for j in range(len(g))
                            if i != j] for g in groups])
    counts = np.array([len(g) * (len(g) - 1) for g in groups])
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_groups = len(groups)

    def draw(group_idx):
        """One A-vs-B assignment for the given pairs."""
        picked = offsets[group_idx] + (rng.random(len(group_idx)) * counts[group_idx]).astype(int)
        return flat[picked, 0], flat[picked, 1]

    def stats(a, b):
        out = {'agreement': float((a == b).mean()), 'positive_rate': float(b.mean())}
        if len(set(a)) > 1 and len(set(b)) > 1:
            out['kappa'] = float(cohen_kappa_score(a, b))
            out['auc'] = float(roc_auc_score(b, a))
        return out

    # Point estimates: average over N_DRAWS re-pairings of the complete set, so the
    # headline does not depend on one arbitrary draw.
    all_pairs = np.arange(n_groups)
    point = [stats(*draw(all_pairs)) for _ in range(N_DRAWS)]

    # Interval: resample the PAIRS with replacement, with a fresh re-pairing each time.
    # The previous version took percentiles across re-pairings of the same fixed set of
    # pairs, which measures only how the study assignment is drawn and ignores sampling
    # error over pairs; it gave intervals about half this width.
    boot = [stats(*draw(rng.integers(0, n_groups, n_groups))) for _ in range(N_BOOT)]

    def ci(key):
        m = float(np.mean([d[key] for d in point if key in d]))
        v = [d[key] for d in boot if key in d]
        return m, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    agrees = [d['agreement'] for d in point]
    kappas = [d['kappa'] for d in point if 'kappa' in d]
    aucs = [d['auc'] for d in point if 'auc' in d]
    pos_rates = [d['positive_rate'] for d in point]

    ag, ag_lo, ag_hi = ci('agreement')
    kp, kp_lo, kp_hi = ci('kappa')
    au, au_lo, au_hi = ci('auc')
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
