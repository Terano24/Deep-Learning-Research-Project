"""
H5: does the grouping-unit effect reproduce on an independent dataset?

Our central claim is that the field's standard peptide-level split is too weak a
grouping unit, so reported performance is partly recovered clustering rather than
transferable biology. That was shown on IEDB using source publication as the group.

This tests the same principle on a dataset we did not assemble: the curated pathogenic
epitope set released with TRAP (Lee et al., Genome Medicine 2023, github.com/ChloeHJ/TRAP).
It carries no publication IDs, so the clustering axes available are the source antigen
(protein) and the source organism. If the principle holds, AUC should fall monotonically
as the grouping unit widens:

    random  >  peptide-grouped  >  antigen-grouped  >  organism-grouped

A single model class (gradient boosting on the AAindex encoding plus our biochemical and
self-similarity features) is used throughout, so only the split changes.

Run:  python -m src.analysis.external_validation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from src.data.dataset import HLA_PATH
from src.data.features import get_features, FEATURE_COLS
from src.data.processing import aaindex
from src.analysis.confounds import within_allele_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
TRAP = Path('C:/Users/Terano/AppData/Local/Temp/claude/'
            'c--Users-Terano-OneDrive-Documents-CNN-Lung-Cancer-Deep-Learning-Research-Project/'
            '02bbbf46-fefb-4293-9822-36532cdb812c/scratchpad/trap/pathogenic_db.csv')
SEED = 42
N_SPLITS = 5


def load_trap():
    df = pd.read_csv(TRAP)
    df = df.rename(columns={'Peptide': 'peptide', 'MHC_Restriction': 'mhc',
                            'Immunogenicity': 'outcome', 'Antigen_Name': 'antigen',
                            'Antigen_Organism': 'organism'})
    known = set(pd.read_csv(HLA_PATH, sep='\t')['HLA'])
    df = df.dropna(subset=['peptide', 'mhc', 'outcome'])
    df = df[df['mhc'].astype(str).str.match(r'^HLA-[ABC]\*\d\d:\d\d$')]
    df['HLA'] = df['mhc'].str.replace(':', '', regex=False)
    df = df[df['HLA'].isin(known)]
    df = df[df['peptide'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]{9,10}$')]
    df['label'] = df['outcome'].astype(str).str.startswith('Positive').astype(int)
    # collapse duplicate peptide-HLA pairs the same way as the main dataset
    g = df.groupby(['peptide', 'HLA']).agg(
        label=('label', 'mean'),
        antigen=('antigen', lambda s: s.mode().iat[0] if not s.mode().empty else 'NA'),
        organism=('organism', lambda s: s.mode().iat[0] if not s.mode().empty else 'NA'),
    ).reset_index()
    g = g[g['label'] != 0.5]
    g['label'] = (g['label'] > 0.5).astype(int)
    return g


def design_matrix(df):
    after_pca = np.loadtxt(BASE_DIR / 'data' / 'raw' / 'after_pca.txt')
    rows = []
    for p in df['peptide']:
        q = p[:5] + '-' + p[5:] if len(p) == 9 else p
        rows.append(aaindex(q, after_pca).ravel())
    enc = np.array(rows)
    allele = pd.Categorical(df['HLA'], categories=sorted(df['HLA'].unique())).codes[:, None]
    feats = df.join(get_features(df['peptide']), on='peptide')[FEATURE_COLS].astype(float).values
    return np.hstack([enc, allele, feats])


def oof(X, y, splitter, groups=None):
    out = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, va in it:
        if len(set(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced')
        m.fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
    return out


def main():
    df = load_trap()
    y = df['label'].values
    X = design_matrix(df)
    print(f'TRAP pathogenic set: {len(df)} peptide-HLA pairs, {y.mean():.1%} positive')
    print(f'  {df["antigen"].nunique()} antigens, {df["organism"].nunique()} organisms, '
          f'{df["HLA"].nunique()} alleles\n')

    regimes = {
        'random': (KFold(N_SPLITS, shuffle=True, random_state=SEED), None),
        'peptide': (GroupKFold(N_SPLITS), df['peptide'].values),
        'antigen': (GroupKFold(N_SPLITS), df['antigen'].values),
        'organism': (GroupKFold(min(N_SPLITS, df['organism'].nunique())), df['organism'].values),
    }
    out = {'n_pairs': int(len(df)), 'positive_rate': float(y.mean()),
           'n_antigens': int(df['antigen'].nunique()),
           'n_organisms': int(df['organism'].nunique())}
    print(f'{"grouping unit":>16}  {"pooled AUC":>11}  {"within-allele":>14}')
    for name, (splitter, groups) in regimes.items():
        p = oof(X, y, splitter, groups)
        d = df.copy()
        d['score'] = p
        d = d[~d['score'].isna()]
        auc = roc_auc_score(d['label'], d['score'])
        _, wa = within_allele_auc(d, 'score', min_n=20)
        out[name] = {'pooled_auc': float(auc), 'within_allele': wa, 'n_scored': int(len(d))}
        print(f'{name:>16}  {auc:>11.3f}  {wa.get("weighted_auc", float("nan")):>14.3f}')

    drop = out['random']['pooled_auc'] - out['organism']['pooled_auc']
    out['random_minus_organism'] = float(drop)
    print(f'\nrandom -> organism-grouped drop: {drop:+.3f}')
    (RESULTS / 'external_validation.json').write_text(json.dumps(out, indent=2))
    print('saved', RESULTS / 'external_validation.json')


if __name__ == '__main__':
    main()
