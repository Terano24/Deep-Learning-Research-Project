"""
What, if anything, predicts immunogenicity INSIDE a single unseen study?

Every headline number in this project has collapsed once study structure was removed,
and `self_exact` turned out to separate studies rather than peptides. So this script
applies one test to everything: compute AUC inside each study that reported both
outcomes, then bootstrap OVER STUDIES to get a confidence interval on the mean. A
predictor only counts if that interval clears 0.5.

Tested:
  * each biochemical / self-similarity feature, on raw values
  * each trained model, from study-grouped out-of-fold predictions
    (so no model has seen its own study)

Bootstrapping over studies rather than rows is deliberate: resampling rows would treat
each study's peptides as independent and overstate confidence.

Run:  python -m src.analysis.what_survives
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.dataset import OUT_PATH
from src.data.features import get_features, FEATURE_COLS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
MIN_STUDY = 10
N_BOOT = 4000
SEED = 42
SCHEMES = {'ours_pointwise': 'oof_study_pointwise.npy',
           'ours_study_weighted': 'oof_study_pointwise_study_weighted.npy',
           'ours_drop_dominant': 'oof_study_pointwise_drop_dominant.npy',
           'ours_mixed_only': 'oof_study_pointwise_mixed_only.npy',
           'ours_ranking': 'oof_study_ranking.npy'}


def per_study(df, col, by_allele=False):
    """(auc, n) for every study (optionally study+allele) with both outcomes."""
    keys = ['reference_id', 'HLA'] if by_allele else ['reference_id']
    out = []
    for _, g in df.groupby(keys):
        v = g[col]
        if len(g) < MIN_STUDY or g['label'].nunique() < 2 or v.isna().any():
            continue
        out.append((roc_auc_score(g['label'], v), len(g)))
    return out


def boot_over_studies(stats, n=N_BOOT, seed=SEED):
    """Resample STUDIES (not rows): the unit of correlation is the study."""
    if len(stats) < 3:
        return None
    a = np.array([s[0] for s in stats])
    rng = np.random.default_rng(seed)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'n_studies': len(a), 'n_pairs': int(sum(s[1] for s in stats)),
            'macro_auc': float(a.mean()), 'ci95': [float(lo), float(hi)],
            'frac_above_chance': float((a > 0.5).mean()),
            'beats_chance': bool(lo > 0.5)}


def report(title, df, cols, by_allele=False):
    print(f'\n=== {title} ===')
    print(f'{"predictor":>24} {"AUC":>6}  {"95% CI (over studies)":>24} {"studies":>8} {">0.5":>6}')
    rows = {}
    for c in cols:
        if c not in df:
            continue
        r = boot_over_studies(per_study(df, c, by_allele))
        if r is None:
            continue
        rows[c] = r
        mark = '  **' if r['beats_chance'] else ''
        print(f'{c:>24} {r["macro_auc"]:>6.3f}  [{r["ci95"][0]:>7.3f}, {r["ci95"][1]:>7.3f}] '
              f'{r["n_studies"]:>8} {r["frac_above_chance"]:>5.0%}{mark}')
    return rows


def main():
    df = pd.read_csv(OUT_PATH)
    df = df.join(get_features(df['peptide']), on='peptide')
    for name, fn in SCHEMES.items():
        p = RESULTS / fn
        if p.exists():
            df[name] = np.load(p)
    models = [m for m in SCHEMES if m in df]

    out = {'min_study': MIN_STUDY, 'n_pairs': int(len(df))}
    out['features_within_study'] = report('FEATURES, within study', df, FEATURE_COLS)
    out['models_within_study'] = report('MODELS, within study (study-grouped OOF)', df, models)
    out['models_within_study_allele'] = report(
        'MODELS, within study AND allele', df, models, by_allele=True)

    lung = df[df['lung_cancer']]
    out['lung_features'] = report('FEATURES, within study (lung only)', lung, FEATURE_COLS)
    out['lung_models'] = report('MODELS, within study (lung only)', lung, models)

    winners = [k for sec in ('features_within_study', 'models_within_study',
                             'models_within_study_allele')
               for k, v in out[sec].items() if v['beats_chance']]
    print('\n' + '=' * 70)
    if winners:
        print('PREDICTORS WHOSE CI CLEARS 0.5:', ', '.join(sorted(set(winners))))
    else:
        print('NOTHING clears chance within study.')
    (RESULTS / 'what_survives.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'what_survives.json')


if __name__ == '__main__':
    main()
