"""
A fair head-to-head against DeepImmuno.

models/CNN_WEIGHT_OPSI is byte-identical to DeepImmuno's published checkpoint
(cnn_model_331_3_7), so every "cnn" column in our results IS DeepImmuno. That makes the
comparison easy but raises a fairness problem in DeepImmuno's favour: it was trained on
IEDB T-cell assays, and our test split is drawn from IEDB too, so some of our test pairs
were in its training data. Scoring it there measures memorisation, not prediction.

This script quantifies the overlap and re-runs the comparison on the subset DeepImmuno
provably never saw, which is the only honest head-to-head.

Run:  python -m src.analysis.fair_vs_deepimmuno
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from src.analysis.confounds import within_allele_auc, paired_bootstrap

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
DI_TRAIN = Path('C:/Users/Terano/AppData/Local/Temp/claude/'
                'c--Users-Terano-OneDrive-Documents-CNN-Lung-Cancer-Deep-Learning-Research-Project/'
                '02bbbf46-fefb-4293-9822-36532cdb812c/scratchpad/di/di_train.csv')
MODELS = ['ours', 'deepimmuno', 'gbm_baseline', 'allele_prior_baseline']


def load():
    base = pd.read_csv(RESULTS / 'test_predictions.csv').rename(columns={'cnn': 'deepimmuno'})
    ours = pd.read_csv(RESULTS / 'improve_test_predictions_pretrained_lung_feat.csv')
    ours = ours.rename(columns={'pred': 'ours'})[['peptide', 'HLA', 'ours']]
    return base.merge(ours, on=['peptide', 'HLA'], how='inner')


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {'n': int(len(y)), 'n_pos': int(y.sum()),
            'auc': float(roc_auc_score(y, p)), 'pr_auc': float(average_precision_score(y, p)),
            'mcc': float(matthews_corrcoef(y, pred))}


def main():
    df = load()
    di = pd.read_csv(DI_TRAIN)
    di_pairs = set(zip(di['peptide'], di['HLA']))
    di_peptides = set(di['peptide'])

    df['in_di_pair'] = [(p, h) in di_pairs for p, h in zip(df['peptide'], df['HLA'])]
    df['in_di_peptide'] = df['peptide'].isin(di_peptides)

    out = {'test_pairs': len(df), 'deepimmuno_train_pairs': len(di_pairs),
           'overlap_exact_pair': int(df['in_di_pair'].sum()),
           'overlap_peptide': int(df['in_di_peptide'].sum())}
    print(f"our test split: {len(df)} pairs")
    print(f"DeepImmuno training set: {len(di)} rows, {len(di_pairs)} unique pairs")
    print(f"  test pairs DeepImmuno trained on (exact pair): {out['overlap_exact_pair']}"
          f" ({out['overlap_exact_pair']/len(df):.1%})")
    print(f"  test pairs whose PEPTIDE it trained on:        {out['overlap_peptide']}"
          f" ({out['overlap_peptide']/len(df):.1%})\n")

    subsets = {
        'all_test_pairs': np.ones(len(df), bool),
        'seen_by_deepimmuno': df['in_di_peptide'].values,
        'unseen_by_deepimmuno': ~df['in_di_peptide'].values,
    }
    for name, mask in subsets.items():
        sub = df[mask]
        if len(sub) < 50 or sub['label'].nunique() < 2:
            print(f'--- {name}: too few pairs ({len(sub)}) ---\n'); continue
        res = {}
        print(f'--- {name}: {len(sub)} pairs, {int(sub["label"].sum())} positive ---')
        print(f'  {"model":>22} {"AUC":>7} {"PR-AUC":>7} {"MCC":>7} {"within-allele":>14}')
        for m in MODELS:
            mm = metrics(sub['label'].values, sub[m].values)
            _, wa = within_allele_auc(sub, m, min_n=20)
            mm['within_allele'] = wa.get('weighted_auc')
            res[m] = mm
            wtxt = f'{mm["within_allele"]:.3f}' if mm['within_allele'] else '   n/a'
            print(f'  {m:>22} {mm["auc"]:>7.3f} {mm["pr_auc"]:>7.3f} {mm["mcc"]:>7.3f} {wtxt:>14}')
        for label, strat in [('overall', False), ('within_allele', True)]:
            b = paired_bootstrap(sub, 'ours', 'deepimmuno', stratified=strat)
            if b:
                res[f'ours_vs_deepimmuno_{label}'] = b
                sig = 'significant' if b['ci95'][0] > 0 else 'not significant'
                print(f'  ours - deepimmuno [{label:>13}]: {b["mean_diff"]:+.3f} '
                      f'CI [{b["ci95"][0]:+.3f}, {b["ci95"][1]:+.3f}]  {sig}')
        out[name] = res
        print()

    (RESULTS / 'fair_vs_deepimmuno.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'fair_vs_deepimmuno.json')


if __name__ == '__main__':
    main()
