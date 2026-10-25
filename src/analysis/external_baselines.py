"""
H1: compare against published external predictors on our held-out test split.

Baselines
  DeepImmuno    the published predictor, scored through its own released checkpoint
                (models/CNN_WEIGHT_OPSI; see src/models/deepimmuno.py). Not to be
                confused with cnn_retrained, which is that architecture trained by us.
  BigMHC_IM     ensemble of 7 pan-allelic networks, presentation-pretrained then
                transfer-learned on immunogenicity (Albert et al. 2023)
  MixMHCpred    binding/presentation only, no immunogenicity training (Gfeller lab).
                Included deliberately: if a pure binding predictor matches the
                immunogenicity models, that supports our claim that most apparent
                immunogenicity signal is binding plus allele identity.

Scores are also reported within-allele, where allele-prior signal cannot contribute,
and on the subset DeepImmuno was not trained on.

Run:  python -m src.analysis.external_baselines
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.confounds import within_allele_auc, paired_bootstrap

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
EXTERNAL = BASE_DIR / 'data' / 'external'
DI_TRAIN = EXTERNAL / 'deepimmuno_train.csv'



def load_external(df, name, column):
    """Attach a published predictor's scores from data/external/ (see its README)."""
    p = EXTERNAL / name
    if not p.exists():
        print(f'  {name} not found, skipping {column}')
        return df
    e = pd.read_csv(p)[['peptide', 'HLA', column]].drop_duplicates(['peptide', 'HLA'])
    return df.merge(e, on=['peptide', 'HLA'], how='left')


def metrics(y, p):
    return {'n': int(len(y)), 'n_pos': int(y.sum()),
            'auc': float(roc_auc_score(y, p)),
            'pr_auc': float(average_precision_score(y, p))}


def main():
    # 'deepimmuno' here is DeepImmuno's released checkpoint, scored in train.py via
    # src/models/deepimmuno.py. 'cnn_retrained' is our own model on the same
    # architecture. They are separate columns and neither is renamed into the other.
    df = pd.read_csv(RESULTS / 'test_predictions.csv')
    ours = pd.read_csv(RESULTS / 'improve_test_predictions_pretrained_lung_feat.csv')
    df = df.merge(ours.rename(columns={'pred': 'ours'})[['peptide', 'HLA', 'ours']],
                  on=['peptide', 'HLA'], how='inner')
    df = load_external(df, 'bigmhc_im_test.csv', 'bigmhc_im')
    df = load_external(df, 'mixmhcpred_test.csv', 'mixmhcpred')

    models = [m for m in ['ours', 'bigmhc_im', 'deepimmuno', 'mixmhcpred',
                          'gbm_baseline', 'allele_prior_baseline',
                          'cnn_retrained'] if m in df]
    missing = {m: int(df[m].isna().sum()) for m in models}
    print('test pairs:', len(df))
    print('missing scores per model:', {k: v for k, v in missing.items() if v})

    di = pd.read_csv(DI_TRAIN) if DI_TRAIN.exists() else None
    if di is not None:
        df['seen_by_di'] = df['peptide'].isin(set(di['peptide']))

    out = {'n_test_pairs': int(len(df)), 'missing': missing}
    subsets = {'all': np.ones(len(df), bool)}
    if di is not None:
        subsets['unseen_by_deepimmuno'] = ~df['seen_by_di'].values

    for name, mask in subsets.items():
        sub = df[mask]
        print(f'\n--- {name}: {len(sub)} pairs, {int(sub["label"].sum())} positive ---')
        print(f'  {"model":>22} {"AUC":>7} {"PR-AUC":>8} {"within-allele":>14}')
        res = {}
        for m in models:
            s = sub.dropna(subset=[m])
            if len(s) < 50 or s['label'].nunique() < 2:
                continue
            mm = metrics(s['label'].values, s[m].values)
            _, wa = within_allele_auc(s, m, min_n=20)
            mm['within_allele'] = wa.get('weighted_auc')
            mm['n_alleles'] = wa.get('n_alleles')
            res[m] = mm
            w = f'{mm["within_allele"]:.3f}' if mm['within_allele'] else '   n/a'
            print(f'  {m:>22} {mm["auc"]:>7.3f} {mm["pr_auc"]:>8.3f} {w:>14}')
        for m in models:
            if m == 'ours' or m not in res:
                continue
            s = sub.dropna(subset=['ours', m])
            for label, strat in [('overall', False), ('within_allele', True)]:
                b = paired_bootstrap(s, 'ours', m, stratified=strat)
                if b:
                    res[f'ours_vs_{m}_{label}'] = b
                    sig = 'sig' if b['ci95'][0] > 0 else ('sig(-)' if b['ci95'][1] < 0 else 'ns')
                    print(f'    ours - {m:<22} [{label:>13}] {b["mean_diff"]:+.3f} '
                          f'CI [{b["ci95"][0]:+.3f}, {b["ci95"][1]:+.3f}] {sig}')
        out[name] = res

    (RESULTS / 'external_baselines.json').write_text(json.dumps(out, indent=2))
    df.to_csv(RESULTS / 'test_predictions_with_baselines.csv', index=False)
    print('\nsaved', RESULTS / 'external_baselines.json')


if __name__ == '__main__':
    main()
