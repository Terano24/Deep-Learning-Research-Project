"""
The decisive test: can the model rank peptides inside a single experiment it never saw?

Context from src/analysis/leakage.py: 75% of IEDB studies in this dataset are
label-homogeneous, and one all-negative study supplies 37% of all pairs. So "is this
pair positive?" is largely answerable from "which paper reported it?", and any pooled
AUC mixes real biology with study fingerprinting.

Within-study AUC removes that entirely. For each held-out study that reported *both*
outcomes, we ask whether the model ordered that study's own peptides correctly. A model
useful for screening candidates must do this; a model that only learned which labs
publish positives cannot.

Reported for both split regimes (peptide-grouped and study-grouped), with and without
the dominant study, and additionally stratified by allele inside each study.

Run:  python -m src.analysis.within_study
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
N_SPLITS = 5
MIN_STUDY = 20


def per_study_auc(df, score_col, min_n=MIN_STUDY, by_allele=False):
    """AUC inside each study (optionally inside each allele within each study)."""
    rows = []
    keys = ['reference_id', 'HLA'] if by_allele else ['reference_id']
    for key, g in df.groupby(keys):
        if len(g) < min_n or g['label'].nunique() < 2:
            continue
        rows.append({'key': key, 'n': len(g), 'pos_frac': float(g['label'].mean()),
                     'auc': roc_auc_score(g['label'], g[score_col])})
    per = pd.DataFrame(rows)
    if per.empty:
        return per, {}
    return per, {
        'n_groups': len(per),
        'n_pairs': int(per['n'].sum()),
        'macro_auc': float(per['auc'].mean()),
        'weighted_auc': float(np.average(per['auc'], weights=per['n'])),
        'frac_groups_above_chance': float((per['auc'] > 0.5).mean()),
    }


def main():
    df = pd.read_csv(OUT_PATH)
    df = df.join(get_features(df['peptide']), on='peptide')
    y = df['label'].values
    X = np.hstack([encoding_matrix(df), df[FEATURE_COLS].astype(float).values])

    dominant = df['reference_id'].value_counts().idxmax()
    dom_n = int((df['reference_id'] == dominant).sum())
    print(f'{len(df)} pairs; dominant study {dominant} holds {dom_n} pairs '
          f'({dom_n / len(df):.0%}, positive rate {df.loc[df.reference_id == dominant, "label"].mean():.3f})\n')

    out = {'dominant_study': int(dominant), 'dominant_study_pairs': dom_n}
    scored = {}
    for regime, key in [('peptide', 'peptide'), ('study', 'reference_id')]:
        scored[regime] = oof_predictions(X, y, GroupKFold(N_SPLITS), df[key].values)

    for regime, oof in scored.items():
        d = df.copy()
        d['score'] = oof
        res = {}
        pooled = roc_auc_score(d['label'], d['score'])
        res['pooled_auc'] = float(pooled)
        _, res['within_study'] = per_study_auc(d, 'score')
        _, res['within_study_and_allele'] = per_study_auc(d, 'score', by_allele=True)
        nd = d[d['reference_id'] != dominant]
        res['pooled_auc_excl_dominant'] = float(roc_auc_score(nd['label'], nd['score']))
        _, res['within_study_excl_dominant'] = per_study_auc(nd, 'score')
        out[regime] = res

        ws = res['within_study']
        wsa = res['within_study_and_allele']
        print(f'--- {regime}-grouped training ---')
        print(f'  pooled AUC                 {pooled:.3f}   (excl. dominant study {res["pooled_auc_excl_dominant"]:.3f})')
        if ws:
            print(f'  within-study AUC           macro {ws["macro_auc"]:.3f}  weighted {ws["weighted_auc"]:.3f}'
                  f'  ({ws["n_groups"]} studies, {ws["n_pairs"]} pairs)')
            print(f'  studies above chance       {ws["frac_groups_above_chance"]:.0%}')
        if wsa:
            print(f'  within-study+allele AUC    macro {wsa["macro_auc"]:.3f}'
                  f'  ({wsa["n_groups"]} groups, {wsa["n_pairs"]} pairs)')
        print()

    (RESULTS / 'within_study_analysis.json').write_text(json.dumps(out, indent=2, default=str))
    print('saved', RESULTS / 'within_study_analysis.json')


if __name__ == '__main__':
    main()
