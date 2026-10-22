"""
Turn the raw IEDB export into a clean, leakage-safe labelled dataset.

Cleaning (mirrors the paper's strategy):
  1. keep only 4-digit HLA-A/B/C alleles that exist in the HLA pseudo-sequence table
  2. keep canonical 20-letter peptides of length 9/10
  3. label: any "Positive*" outcome -> 1, "Negative" -> 0
  4. collapse repeated (peptide, HLA) pairs by majority vote; ties are dropped
Split: whole peptides are assigned to a fold, so the same peptide never appears
in both train and test (even with a different HLA).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = BASE_DIR / 'data' / 'raw' / 'iedb_tcell_mhc1.csv'
HLA_PATH = BASE_DIR / 'data' / 'raw' / 'ParatopeIMGTopsi.txt'
OUT_PATH = BASE_DIR / 'data' / 'processed' / 'dataset.csv'

TEST_FRACTION = 0.15
N_FOLDS = 10
SEED = 42


def clean(raw_path=RAW_PATH, hla_path=HLA_PATH):
    raw = pd.read_csv(raw_path)
    known = set(pd.read_csv(hla_path, sep='\t')['HLA'])
    stats = {'raw_records': len(raw)}

    raw = raw.dropna(subset=['linear_sequence', 'mhc_restriction', 'qualitative_measure'])
    raw = raw[raw['mhc_restriction'].str.match(r'^HLA-[ABC]\*\d\d:\d\d$')]
    raw['HLA'] = raw['mhc_restriction'].str.replace(':', '', regex=False)
    raw = raw[raw['HLA'].isin(known)]
    raw = raw[raw['linear_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]{9,10}$')]
    stats['usable_records'] = len(raw)

    raw['label'] = raw['qualitative_measure'].str.startswith('Positive').astype(int)
    grouped = raw.groupby(['linear_sequence', 'HLA']).agg(
        n_records=('label', 'size'),
        pos_frac=('label', 'mean'),
        lung_cancer=('lung_cancer', 'any'),
        # the publication contributing most records for this pair, kept so we can also
        # split by study: IEDB records cluster heavily by reference (shared cohorts,
        # cell lines and protocols), which peptide-level splitting does not separate
        reference_id=('reference_id', lambda s: s.mode().iat[0] if not s.mode().empty else -1),
        n_references=('reference_id', 'nunique'),
    ).reset_index()
    stats['unique_pairs'] = len(grouped)
    grouped = grouped[grouped['pos_frac'] != 0.5]
    stats['pairs_after_tie_drop'] = len(grouped)
    grouped['label'] = (grouped['pos_frac'] > 0.5).astype(int)
    grouped = grouped.rename(columns={'linear_sequence': 'peptide'})
    return grouped[['peptide', 'HLA', 'label', 'lung_cancer', 'n_records',
                    'reference_id', 'n_references']], stats


def assign_splits(df, seed=SEED):
    """Two independent split schemes, both held in the same table.

    split/fold        grouped by peptide -- the headline evaluation
    study_split/fold  grouped by source publication -- the stricter check, since
                      records from one paper share cohorts, cell lines and protocols
    """
    df = df.copy()
    for prefix, key in [('', 'peptide'), ('study_', 'reference_id')]:
        groups = df[key].values
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
        _, test_idx = next(gss.split(df, groups=groups))
        scol, fcol = f'{prefix}split', f'{prefix}fold'
        df[scol] = 'train'
        df.iloc[test_idx, df.columns.get_loc(scol)] = 'test'
        df[fcol] = -1
        train = df[df[scol] == 'train']
        gkf = GroupKFold(n_splits=N_FOLDS)
        for k, (_, val_idx) in enumerate(gkf.split(train, groups=train[key].values)):
            df.loc[train.index[val_idx], fcol] = k
    return df


def build_dataset(out_path=OUT_PATH):
    df, stats = clean()
    df = assign_splits(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    stats.update({
        'train': int((df['split'] == 'train').sum()),
        'test': int((df['split'] == 'test').sum()),
        'test_lung_cancer': int(((df['split'] == 'test') & df['lung_cancer']).sum()),
        'positive_rate': round(float(df['label'].mean()), 4),
    })
    for k, v in stats.items():
        print(f'{k:>22}: {v}')
    return df


if __name__ == '__main__':
    build_dataset()
