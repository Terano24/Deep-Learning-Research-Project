"""
The paper's primary table: all predictors on every lung-cancer-associated pair.

Earlier lung numbers used only the 181 lung pairs that fell in the peptide-grouped test
split, which gave uselessly wide intervals. Here every one of the 1,109 lung pairs is
scored:

  externally trained models (DeepImmuno, BigMHC_IM, MixMHCpred) can score all pairs
  directly -- they never saw our split at all;

  locally trained models (ours, gradient boosting, allele prior) are scored from
  STUDY-grouped out-of-fold predictions, so each lung pair is predicted by a model that
  never saw its source publication.

Reported pooled, within-lung-allele, and within-lung-study, because the lung subset is
even more study-concentrated than IEDB overall (81% of lung studies are label-homogeneous).

Run:  python -m src.analysis.lung_benchmark
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.dataset import OUT_PATH
from src.analysis.confounds import within_allele_auc
from src.analysis.within_study import per_study_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
EXTERNAL = BASE_DIR / 'data' / 'external'
MIN_GROUP = 10


def score_deepimmuno(df):
    """DeepImmuno's published checkpoint, applied to every lung pair."""
    from src.models.deepimmuno import score
    return score(df)


def load_external(df, name, column):
    """Attach a published predictor's scores from data/external/ (see its README)."""
    p = EXTERNAL / name
    if not p.exists():
        print(f'  {name} not found, skipping {column}')
        return df
    e = pd.read_csv(p)[['peptide', 'HLA', column]].drop_duplicates(['peptide', 'HLA'])
    return df.merge(e, on=['peptide', 'HLA'], how='left')


def load_study_oof(df, full):
    """Study-grouped OOF from src/analysis/robustness.py, restricted to lung rows."""
    lung_mask = full['lung_cancer'].values
    for scheme, name in [('pointwise', 'ours_pointwise'),
                         ('pointwise_study_weighted', 'ours_study_weighted'),
                         ('ranking', 'ours_ranking')]:
        p = RESULTS / f'oof_study_{scheme}.npy'
        if p.exists():
            df[name] = np.load(p)[lung_mask]
    return df


def boot_auc(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def contamination(df):
    """Overlap with each published predictor's training set, and the cost of removing it.

    Overlap is reported two ways because the two are not interchangeable and the
    difference decides which pairs count as unseen: by shared peptide (strict -- the
    model saw that sequence against some allele) and by exact peptide-HLA pair (lax).
    The 'unseen by both' subset uses the strict rule for both predictors, so a pair
    survives only if neither model has encountered its sequence at all.

    These numbers were previously typed into the manuscript from a figure script that
    hard-coded them; nothing in results/ produced them. They are computed here.
    """
    train = {}
    for name, fn in [('deepimmuno', 'deepimmuno_train.csv'),
                     ('bigmhc_im', 'bigmhc_im_train.csv')]:
        p = EXTERNAL / fn
        if p.exists():
            tr = pd.read_csv(p)
            train[name] = (set(tr['peptide']),
                           set(map(tuple, tr[['peptide', 'HLA']].values)))

    pairs = list(zip(df['peptide'], df['HLA']))
    out, seen_any = {}, np.zeros(len(df), bool)
    for name, (peps, prs) in train.items():
        if name not in df:
            continue
        by_pep = np.array([p in peps for p, _ in pairs])
        by_pair = np.array([ph in prs for ph in pairs])
        seen_any |= by_pep
        row = {'overlap_peptide': float(by_pep.mean()),
               'overlap_exact_pair': float(by_pair.mean()),
               'n_seen': int(by_pep.sum()), 'n_unseen': int((~by_pep).sum())}
        for tag, mask in [('seen', by_pep), ('unseen', ~by_pep)]:
            s = df[mask].dropna(subset=[name])
            if len(s) >= 50 and s['label'].nunique() > 1:
                row[tag] = {'n': int(len(s)),
                            'auc': float(roc_auc_score(s['label'], s[name])),
                            'auc_ci95': boot_auc(s['label'], s[name])}
        out[name] = row
        print(f'  {name:>12}: overlap {by_pep.mean():.1%} by peptide, '
              f'{by_pair.mean():.1%} by exact pair | '
              f'seen {row.get("seen", {}).get("auc", float("nan")):.3f} '
              f'unseen {row.get("unseen", {}).get("auc", float("nan")):.3f}')

    u = df[~seen_any]
    out['unseen_by_all'] = {'n': int(len(u)), 'n_pos': int(u['label'].sum()),
                            'rule': 'shared peptide with either training set'}
    print(f'\n  unseen by every audited predictor: {len(u)} pairs, '
          f'{int(u["label"].sum())} positive')
    for m in ['deepimmuno', 'bigmhc_im', 'mixmhcpred', 'allele_prior']:
        if m not in u:
            continue
        s = u.dropna(subset=[m])
        if len(s) < 50 or s['label'].nunique() < 2:
            continue
        a = float(roc_auc_score(s['label'], s[m]))
        ci = boot_auc(s['label'], s[m])
        out['unseen_by_all'][m] = {'n': int(len(s)), 'auc': a, 'auc_ci95': ci}
        print(f'    {m:>14} {a:.3f}  CI [{ci[0]:.3f}, {ci[1]:.3f}]')
    return out


def evaluate(df, model):
    s = df.dropna(subset=[model])
    if len(s) < 50 or s['label'].nunique() < 2:
        return None
    _, wa = within_allele_auc(s, model, min_n=MIN_GROUP)
    _, ws = per_study_auc(s, model, min_n=MIN_GROUP)
    return {
        'n': int(len(s)), 'n_pos': int(s['label'].sum()),
        'auc': float(roc_auc_score(s['label'], s[model])),
        'pr_auc': float(average_precision_score(s['label'], s[model])),
        'within_allele': wa.get('weighted_auc'), 'n_alleles': wa.get('n_alleles'),
        'within_study': ws.get('macro_auc'), 'n_studies': ws.get('n_groups'),
    }


def main():
    full = pd.read_csv(OUT_PATH)
    df = full[full['lung_cancer']].reset_index(drop=True).copy()
    print(f'lung pairs: {len(df)}, positive {df["label"].mean():.1%}, '
          f'{df["reference_id"].nunique()} studies, {df["HLA"].nunique()} alleles\n')

    df = load_study_oof(df, full)
    df = load_external(df, 'bigmhc_im_lung.csv', 'bigmhc_im')
    df = load_external(df, 'mixmhcpred_lung.csv', 'mixmhcpred')
    print('scoring DeepImmuno on all lung pairs...', flush=True)
    df['deepimmuno'] = score_deepimmuno(df)
    # allele prior learned from NON-lung data only, so it never sees a lung label
    prior = full[~full['lung_cancer']].groupby('HLA')['label'].mean()
    df['allele_prior'] = df['HLA'].map(prior).fillna(full['label'].mean())

    models = [m for m in ['ours_ranking', 'ours_study_weighted', 'ours_pointwise',
                          'deepimmuno', 'bigmhc_im', 'mixmhcpred', 'allele_prior']
              if m in df]
    out = {'n_lung_pairs': int(len(df)), 'n_studies': int(df['reference_id'].nunique())}
    print(f'\n{"model":>22} {"n":>5} {"AUC":>7} {"PR-AUC":>8} {"w-allele":>9} {"w-study":>8}')
    for m in models:
        r = evaluate(df, m)
        if r is None:
            print(f'{m:>22}  insufficient data'); continue
        out[m] = r
        wa = f'{r["within_allele"]:.3f}' if r['within_allele'] else '  n/a'
        ws = f'{r["within_study"]:.3f}' if r['within_study'] else '  n/a'
        print(f'{m:>22} {r["n"]:>5} {r["auc"]:>7.3f} {r["pr_auc"]:>8.3f} {wa:>9} {ws:>8}')

    out['contamination'] = contamination(df)

    (RESULTS / 'lung_benchmark.json').write_text(json.dumps(out, indent=2))
    df.to_csv(RESULTS / 'lung_predictions.csv', index=False)
    print('\nsaved', RESULTS / 'lung_benchmark.json')


if __name__ == '__main__':
    main()
