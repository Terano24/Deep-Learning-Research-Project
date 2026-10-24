"""
Per-peptide biochemical and self-similarity features.

Motivation. Two signals are well established in the literature but absent from the
original CNN:

  1. Self-similarity / foreignness. T cells reactive to the host proteome are deleted
     during thymic selection, so a peptide that looks like "self" is unlikely to be
     immunogenic regardless of how well it binds. We measure this against the reviewed
     human proteome (UniProt UP000005640) at three resolutions: exact 9/10-mer match,
     a match within one substitution, and the fraction of the peptide's 4-mers that
     occur anywhere in the proteome (a graded "self-ness" score).

  2. Hydrophobicity at TCR-facing positions. Residues in the central bulge of the
     peptide contact the TCR rather than the MHC groove, and their hydrophobicity
     correlates with observed immunogenicity (Chowell et al., PNAS 2015).

Features are cached to data/processed/peptide_features.csv keyed by peptide.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROTEOME = BASE_DIR / 'data' / 'external' / 'human_proteome.fasta.gz'
CACHE = BASE_DIR / 'data' / 'processed' / 'peptide_features.csv'

AMINO = 'ACDEFGHIKLMNPQRSTVWY'
CODE = {a: i + 1 for i, a in enumerate(AMINO)}      # 0 reserved, 5 bits per residue
BITS = 5

# Kyte-Doolittle hydropathy
HYDRO = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
         'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
         'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
CHARGE = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
# positions that bulge toward the TCR rather than anchoring in the groove (1-based 4-6)
TCR_FACING = (3, 4, 5)
ANCHORS = (1, 2, -1, -2)


def _encode(seq):
    v = 0
    for c in seq:
        i = CODE.get(c)
        if i is None:
            return None
        v = (v << BITS) | i
    return v


def iter_proteins(path=PROTEOME):
    opener = gzip.open if str(path).endswith('.gz') else open
    seq = []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if seq:
                    yield ''.join(seq)
                    seq = []
            else:
                seq.append(line.strip())
    if seq:
        yield ''.join(seq)


def build_self_index(path=PROTEOME, lengths=(9, 10), kmer=7):
    """Hash sets of every human k-mer at the given lengths, plus a sorted short-kmer array.

    The short k-mer is 7 rather than 4: almost every 4-mer occurs somewhere in the
    proteome, so a 4-mer fraction saturates at 1.0 and carries no information.
    The short set is held as a sorted numpy array and queried with searchsorted to
    keep memory manageable (~11M entries).
    """
    # Every index is a sorted int64 array rather than a Python set: the proteome yields
    # ~11M k-mers per length, which costs ~90 MB as an array but close to 1 GB as a set.
    # Buffers are flushed and deduped in bounded chunks so peak memory stays modest.
    sizes = list(lengths) + [kmer]
    arrays = {n: np.empty(0, dtype=np.int64) for n in sizes}
    bufs = {n: [] for n in sizes}

    def flush(n, force=False):
        if bufs[n] and (force or len(bufs[n]) >= 500_000):
            arrays[n] = np.union1d(arrays[n], np.fromiter(bufs[n], dtype=np.int64,
                                                          count=len(bufs[n])))
            bufs[n].clear()

    for prot in iter_proteins(path):
        for n in sizes:
            b = bufs[n]
            for i in range(len(prot) - n + 1):
                v = _encode(prot[i:i + n])
                if v is not None:
                    b.append(v)
            flush(n)
    for n in sizes:
        flush(n, force=True)
    return {n: arrays[n] for n in lengths}, arrays[kmer], kmer


def _in_sorted(values, sorted_arr):
    """Vectorised membership test against a sorted int64 array."""
    v = np.asarray(values, dtype=np.int64)
    pos = np.searchsorted(sorted_arr, v)
    pos = np.clip(pos, 0, len(sorted_arr) - 1)
    return sorted_arr[pos] == v


def _within_one(peptide, index):
    """True if some human k-mer of the same length differs by at most one substitution.

    All single-substitution variants are tested in one vectorised membership query
    against the sorted index.
    """
    if len(index) == 0:
        return False
    variants = []
    for i, orig in enumerate(peptide):
        for a in AMINO:
            if a != orig:
                v = _encode(peptide[:i] + a + peptide[i + 1:])
                if v is not None:
                    variants.append(v)
    return bool(variants) and bool(_in_sorted(variants, index).any())


def peptide_features(peptides, path=PROTEOME):
    full, short, k = build_self_index(path)
    rows = []
    for p in peptides:
        n = len(p)
        idx = full.get(n, np.empty(0, dtype=np.int64))
        code = _encode(p)
        exact = code is not None and len(idx) and bool(_in_sorted([code], idx)[0])
        codes = [_encode(p[i:i + k]) for i in range(n - k + 1)]
        codes = [c for c in codes if c is not None]
        hits = int(_in_sorted(codes, short).sum()) if codes else 0
        h = [HYDRO.get(c, 0.0) for c in p]
        rows.append({
            'peptide': p,
            'self_exact': int(exact),
            # skip the expensive neighbour scan when the peptide is already self
            'self_within1': 1 if exact else int(_within_one(p, idx)),
            'self_7mer_frac': hits / max(n - k + 1, 1),
            'hydro_mean': float(np.mean(h)),
            'hydro_tcr_facing': float(np.mean([h[i] for i in TCR_FACING if i < n])),
            'hydro_anchor': float(np.mean([h[i] for i in ANCHORS if abs(i) <= n])),
            'charge': float(sum(CHARGE.get(c, 0) for c in p)),
            'aromatic_frac': sum(c in 'FWY' for c in p) / n,
            'length10': int(n == 10),
        })
    return pd.DataFrame(rows)


FEATURE_COLS = ['self_exact', 'self_within1', 'self_7mer_frac', 'hydro_mean',
                'hydro_tcr_facing', 'hydro_anchor', 'charge', 'aromatic_frac', 'length10']


def get_features(peptides, rebuild=False):
    """Cached lookup; computes only peptides missing from the cache."""
    peptides = pd.unique(pd.Series(list(peptides)))
    cached = pd.read_csv(CACHE) if CACHE.exists() and not rebuild else pd.DataFrame(columns=['peptide'] + FEATURE_COLS)
    missing = sorted(set(peptides) - set(cached['peptide']))
    if missing:
        print(f'computing self-similarity features for {len(missing)} peptides...', flush=True)
        new = peptide_features(missing)
        cached = pd.concat([cached, new], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(CACHE, index=False)
    return cached.set_index('peptide').loc[peptides, FEATURE_COLS]


if __name__ == '__main__':
    from src.data.dataset import OUT_PATH
    df = pd.read_csv(OUT_PATH)
    f = get_features(df['peptide'])
    j = df.join(f, on='peptide')
    print(f.astype(float).describe().T[['mean', 'std', 'min', 'max']].round(3))
    print('\nmean feature value by label:')
    print(j.groupby('label')[FEATURE_COLS].mean().T.round(3))
    print('\nlung-cancer subset by label:')
    print(j[j['lung_cancer']].groupby('label')[FEATURE_COLS].mean().T.round(3))
