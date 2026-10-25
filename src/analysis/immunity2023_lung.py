"""
Independent test of the lung claim, using Mueller et al. (Immunity 2023) supplements.

Our IEDB analysis says lung-cancer-associated epitopes are harder to predict than the
field average. That could be an artifact of how IEDB curates lung records. This checks it
against a completely independent, uniformly processed resource: the harmonized
multi-cancer dataset from Mueller et al., which reports per-patient neo-peptide screening
counts (mmc2) and the rank each immunogenic neo-peptide received from their classifiers
(mmc5).

From a positive's rank r among N candidates for a patient with P positives, the fraction
of negatives it outranks is (N - r) / (N - P). Averaging over a patient's positives gives
a per-patient AUC equivalent -- the same within-group quantity we compute on IEDB. We then
ask whether lung patients score lower than other cancers.

Run:  python -m src.analysis.immunity2023_lung
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
DL = Path('C:/Users/Terano/Downloads')


def read_sheet(path, sheet):
    """These supplements carry a title row, so locate the real header row first."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = None
    # match a cell that *is* "Patient", not a title sentence containing "patients"
    for i in range(min(12, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower()
        if (row == 'patient').any():
            hdr = i
            break
    if hdr is None:
        raise ValueError(f'no header row found in {path}:{sheet}')
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    return df.reset_index(drop=True)


def norm_cancer(s):
    s = str(s).lower()
    if 'lung' in s:
        return 'lung'
    if 'melanoma' in s or 'skin' in s:
        return 'melanoma/skin'
    if 'colon' in s or 'rectum' in s or 'colorectal' in s:
        return 'colorectal'
    if 'nan' in s or s.strip() in ('', 'none'):
        return 'unknown'
    return 'other'


def main():
    stats = read_sheet(DL / 'mmc2.xlsx', 'neo-peptides')
    rank = read_sheet(DL / 'mmc5.xlsx', 'Ranking neo-peptides')
    print(f'mmc2 patients: {len(stats)} | mmc5 ranking rows: {len(rank)}')

    cols = {c.lower(): c for c in stats.columns}
    pat = cols.get('patient')
    ctype = next((v for k, v in cols.items() if 'cancer type' in k), None)
    n_imm = next((v for k, v in cols.items() if 'imm count' in k), None)
    n_tested = next((v for k, v in cols.items() if 'tested' in k and 'not' not in k), None)
    stats = stats[[c for c in [pat, ctype, n_imm, n_tested] if c]].copy()
    stats.columns = ['patient', 'cancer_type', 'n_imm', 'n_tested'][:len(stats.columns)]
    stats['cancer'] = stats['cancer_type'].map(norm_cancer)
    for c in ('n_imm', 'n_tested'):
        stats[c] = pd.to_numeric(stats[c], errors='coerce')

    print('\npatients by cancer type:')
    print(stats['cancer'].value_counts().to_string())

    rank.columns = [c.strip() for c in rank.columns]
    rank['Rank'] = pd.to_numeric(rank['Rank'], errors='coerce')
    rank['Max_rank'] = pd.to_numeric(rank['Max_rank'], errors='coerce')
    rank = rank.dropna(subset=['Rank', 'Max_rank'])

    out = {}
    print('\n=== per-patient AUC equivalent, by classifier and cancer type ===')
    for clf, g in rank.groupby('Classifier'):
        rows = []
        for p, pg in g.groupby('Patient'):
            N = pg['Max_rank'].max()
            P = len(pg)
            if N <= P:
                continue
            # Mann-Whitney: a positive at rank r outranks N - r candidates, but some of
            # those are the patient's other positives and must not be counted as
            # negatives it beat. Sorting the positives by rank, the one lying k-th from
            # the bottom has P - k positives below it. Omitting that term lets the
            # estimate exceed 1.0, which it did for half the patients.
            r = np.sort(pg['Rank'].values)
            below = (N - r) - (P - np.arange(1, P + 1))
            auc = float(np.mean(below) / (N - P))
            rows.append({'patient': p, 'auc': auc, 'n_pos': P, 'n_cand': N})
        per = pd.DataFrame(rows).merge(stats[['patient', 'cancer']], on='patient', how='left')
        if per.empty:
            continue
        print(f'\n-- {clf} ({len(per)} patients) --')
        summ = {}
        for c, cg in per.groupby('cancer'):
            if len(cg) < 2:
                continue
            summ[c] = {'n_patients': int(len(cg)), 'mean_auc': float(cg['auc'].mean()),
                       'median_auc': float(cg['auc'].median())}
            print(f'   {c:>14}: n={len(cg):>3}  mean AUC {cg["auc"].mean():.3f}  '
                  f'median {cg["auc"].median():.3f}')
        if 'lung' in summ and len(per[per.cancer != 'lung']) > 1:
            lung = per[per.cancer == 'lung']['auc'].values
            rest = per[(per.cancer != 'lung') & (per.cancer != 'unknown')]['auc'].values
            if len(lung) >= 2 and len(rest) >= 2:
                rng = np.random.default_rng(0)
                d = [rng.choice(lung, len(lung)).mean() - rng.choice(rest, len(rest)).mean()
                     for _ in range(4000)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                summ['lung_minus_rest'] = {'diff': float(np.mean(d)), 'ci95': [float(lo), float(hi)],
                                           'lung_worse': bool(hi < 0)}
                print(f'   lung - others: {np.mean(d):+.3f} CI [{lo:+.3f}, {hi:+.3f}]'
                      f'{"  LUNG SIGNIFICANTLY WORSE" if hi < 0 else "  not significant"}')
        out[clf] = summ

    print('\n=== screening yield (immunogenic / tested) by cancer type ===')
    s = stats.dropna(subset=['n_imm', 'n_tested'])
    s = s[s['n_tested'] > 0]
    for c, cg in s.groupby('cancer'):
        if len(cg) < 2:
            continue
        rate = cg['n_imm'].sum() / cg['n_tested'].sum()
        print(f'   {c:>14}: {len(cg):>3} patients, {int(cg["n_imm"].sum()):>4} immunogenic / '
              f'{int(cg["n_tested"].sum()):>6} tested = {rate:.2%}')
        out.setdefault('yield', {})[c] = {'n_patients': int(len(cg)),
                                          'n_imm': int(cg['n_imm'].sum()),
                                          'n_tested': int(cg['n_tested'].sum()),
                                          'rate': float(rate)}

    (RESULTS / 'immunity2023_lung.json').write_text(json.dumps(out, indent=2))
    print('\nsaved', RESULTS / 'immunity2023_lung.json')


if __name__ == '__main__':
    main()
