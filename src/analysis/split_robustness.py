"""
Is the grouping-unit finding robust, or an artefact of one model, one split, or one study?

Section 3.4 compares random, peptide-grouped and study-grouped splits with a single model
class and an exact-match definition of "peptide". Four objections follow, and this script
answers each:

  1. "Peptide-level splitting should group by sequence similarity, not exact string."
       Peptides within `d` mismatches are clustered (single linkage) and whole clusters are
       held out together. d=1 is the "fewer than two mismatches" rule that T-SCAPE (Kim et
       al., Sci Adv 2025) applies; d=2 is stricter. A third rule links peptides that share
       any exact 6-residue substring, which groups overlapping windows of one source protein.
  2. "It is specific to gradient boosting."
       Every regime is repeated with logistic regression, and with an allele-prior baseline
       that sees no peptide at all, so the allele-identity share of each number is visible.
  3. "It is one odd study, or a difference between assays."
       The 11,486-pair all-negative study is dropped; separately, each regime is rerun inside
       each assay family, on pairs whose records all come from that family.
  4. "Study-grouped CV is still not a realistic deployment."
       Studies are ordered by PubMed ID (a proxy for publication date); the models train on
       the earliest 80% of pairs and are tested on the most recent studies.

Every number is pooled AUC and within-allele AUC, computed as in 3.4.

Run:  python -m src.analysis.split_robustness
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.confounds import within_allele_auc
from src.analysis.feature_ablation import encoding_matrix
from src.data.dataset import OUT_PATH
from src.data.features import FEATURE_COLS, get_features

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
RAW = BASE_DIR / 'data' / 'raw' / 'iedb_tcell_mhc1.csv'
SEED = 42
N_SPLITS = 5
MIN_STRATUM = 1000      # smallest assay stratum worth reporting (ICS is the small one)
MIN_STRATUM_STUDIES = 25


# ---------------------------------------------------------------------- homology clusters
def homology_groups(peptides, max_mismatch):
    """Cluster peptides of equal length that differ at <= max_mismatch positions.

    Pigeonhole: cut each peptide into max_mismatch + 1 segments; two peptides within that
    many mismatches must agree exactly on at least one segment, so candidate pairs only
    come from peptides sharing a segment. Candidates are then checked exactly and merged
    with union-find (single linkage). Peptides of different lengths are never merged.
    """
    uniq = sorted(set(peptides))
    idx = {p: i for i, p in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_len = {}
    for p in uniq:
        by_len.setdefault(len(p), []).append(p)
    for length, group in by_len.items():
        n_seg = max_mismatch + 1
        bounds = np.linspace(0, length, n_seg + 1).astype(int)
        codes = np.array([[ord(c) for c in p] for p in group], dtype=np.int16)
        gid = np.array([idx[p] for p in group])
        for s in range(n_seg):
            lo, hi = bounds[s], bounds[s + 1]
            buckets = {}
            for k, p in enumerate(group):
                buckets.setdefault(p[lo:hi], []).append(k)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                m = np.array(members)
                sub = codes[m]
                for start in range(0, len(m), 400):          # bound memory on big buckets
                    blk = sub[start:start + 400]
                    mism = (blk[:, None, :] != sub[None, :, :]).sum(axis=2)
                    rows, cols = np.where(mism <= max_mismatch)
                    for r, c in zip(rows, cols):
                        a, b = m[start + r], m[c]
                        if a < b:
                            ra, rb = find(gid[a]), find(gid[b])
                            if ra != rb:
                                parent[ra] = rb
    roots = np.array([find(idx[p]) for p in peptides])
    return roots


def cluster_stats(groups):
    _, counts = np.unique(groups, return_counts=True)
    return {'n_clusters': int(len(counts)),
            'largest_cluster_share_of_pairs': float(counts.max() / counts.sum()),
            'singleton_share_of_clusters': float((counts == 1).mean())}


def kmer_groups(peptides, k):
    """Cluster peptides that share any exact substring of k residues.

    Windows cut from the same protein overlap, so two 9-mers shifted by s positions share
    9 - s contiguous residues. Linking on a shared k-mer therefore groups neighbouring
    windows of one source protein, which Hamming distance cannot: a shifted window differs
    at almost every position.
    """
    uniq = sorted(set(peptides))
    idx = {p: i for i, p in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for p in uniq:
        i = idx[p]
        for s in range(len(p) - k + 1):
            j = first.setdefault(p[s:s + k], i)
            if j != i:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(idx[p]) for p in peptides])


# -------------------------------------------------------------------------------- models
def make_models():
    return {
        'gbm': lambda: HistGradientBoostingClassifier(random_state=SEED, class_weight='balanced'),
        'logreg': lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=300, class_weight='balanced')),
    }


def cv_scores(df, X, y, splitter, groups, models):
    """Out-of-fold scores for every model plus the peptide-blind allele prior."""
    n = len(df)
    oof = {m: np.full(n, np.nan) for m in list(models) + ['allele_prior']}
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    hla = df['HLA'].values
    for tr, va in it:
        for name, make in models.items():
            mdl = make().fit(X[tr], y[tr])
            oof[name][va] = mdl.predict_proba(X[va])[:, 1]
        prior = pd.Series(y[tr]).groupby(hla[tr]).mean()
        oof['allele_prior'][va] = pd.Series(hla[va]).map(prior).fillna(y[tr].mean()).values
    return oof


def summarise(df, oof):
    out = {}
    for name, score in oof.items():
        ok = ~np.isnan(score)
        d = df[ok].copy()
        d['score'] = score[ok]
        if d['label'].nunique() < 2:
            continue
        if name == 'allele_prior':
            wa = {}      # constant within an allele inside a fold, so this is not meaningful
        else:
            _, wa = within_allele_auc(d, 'score')
        out[name] = {'pooled_auc': float(roc_auc_score(d['label'], d['score'])),
                     'within_allele': float(wa['weighted_auc']) if wa else None,
                     'n_scored': int(len(d))}
    return out


def study_bootstrap_diff(df, score_a, score_b, n_boot=1000, seed=SEED):
    """AUC(a) - AUC(b) on the same rows, resampling whole studies with replacement.

    Pairs within a study are not independent, so resampling rows would understate the
    uncertainty; the unit that varies between replications is the study.
    """
    rng = np.random.default_rng(seed)
    label = df['label'].values
    codes, uniq = pd.factorize(df['reference_id'])
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    full = roc_auc_score(label, score_a) - roc_auc_score(label, score_b)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        y_b = label[idx]
        if y_b.min() == y_b.max():
            continue
        diffs.append(roc_auc_score(y_b, score_a[idx]) - roc_auc_score(y_b, score_b[idx]))
    return {'diff': float(full), 'ci95': [float(np.percentile(diffs, 2.5)),
                                          float(np.percentile(diffs, 97.5))]}


def study_bootstrap_auc(df, score, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    label = df['label'].values
    codes, uniq = pd.factorize(df['reference_id'])
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if label[idx].min() == label[idx].max():
            continue
        vals.append(roc_auc_score(label[idx], score[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_regimes(df, X, y, regimes, models, tag='', keep=None):
    res = {}
    for rname, (splitter, groups) in regimes.items():
        t0 = time.time()
        oof = cv_scores(df, X, y, splitter, groups, models)
        if keep is not None:
            keep[rname] = oof['gbm']
        res[rname] = summarise(df, oof)
        line = '  '.join(f"{m}: {v['pooled_auc']:.3f}/{v['within_allele']:.3f}"
                         for m, v in res[rname].items() if v['within_allele'] is not None)
        print(f'  {tag}{rname:<12} {line}   ({time.time() - t0:.0f}s)', flush=True)
    return res


def standard_regimes(df, extra=None):
    reg = {'random': (KFold(N_SPLITS, shuffle=True, random_state=SEED), None),
           'peptide': (GroupKFold(N_SPLITS), df['peptide'].values)}
    reg.update(extra or {})
    reg['study'] = (GroupKFold(N_SPLITS), df['reference_id'].values)
    return reg


# ------------------------------------------------------------------------------ main
def assay_family_by_pair():
    """Set of assay families behind each pair, using the same family definition as H5."""
    from src.analysis.ceiling_robustness import load
    raw = load()
    raw['assay_family'] = raw['assay_names'].fillna('unknown').astype(str).str.split('|').str[-1]
    raw['pair'] = raw['linear_sequence'] + '|' + raw['HLA']
    fam = raw.groupby('pair')['assay_family'].agg(lambda s: set(s))
    return fam, raw['assay_family'].value_counts().head(3).index.tolist()


def main():
    df = pd.read_csv(OUT_PATH)
    df = df.join(get_features(df['peptide']), on='peptide')
    y = df['label'].values.astype(int)
    X = np.hstack([encoding_matrix(df), df[FEATURE_COLS].astype(float).values])
    print(f'{len(df)} pairs, {df["reference_id"].nunique()} studies, X = {X.shape}', flush=True)
    models = make_models()
    out = {'n_pairs': int(len(df)), 'n_studies': int(df['reference_id'].nunique()),
           'folds': N_SPLITS}

    # ---- 1 + 2: homology-aware splits, three model types
    print('\n[1,2] homology-aware splits, three model types  (pooled/within-allele AUC)', flush=True)
    extra, out['homology'] = {}, {}
    for d in (1, 2):
        g = homology_groups(df['peptide'].tolist(), d)
        out['homology'][f'd{d}'] = cluster_stats(g)
        print(f'  d<={d} mismatches: {out["homology"][f"d{d}"]}', flush=True)
        extra[f'homology_d{d}'] = (GroupKFold(N_SPLITS), g)
    g = kmer_groups(df['peptide'].tolist(), 6)
    out['homology']['shared_6mer'] = cluster_stats(g)
    print(f'  shared 6-mer: {out["homology"]["shared_6mer"]}', flush=True)
    extra['shared_6mer'] = (GroupKFold(N_SPLITS), g)
    oof_full = {}
    out['regimes'] = run_regimes(df, X, y, standard_regimes(df, extra), models, keep=oof_full)
    print('  bootstrapping the gaps over studies ...', flush=True)
    out['gaps'] = {
        'random_minus_study': study_bootstrap_diff(df, oof_full['random'], oof_full['study']),
        'peptide_minus_study': study_bootstrap_diff(df, oof_full['peptide'], oof_full['study']),
        'random_minus_peptide': study_bootstrap_diff(df, oof_full['random'], oof_full['peptide']),
        'peptide_minus_shared_6mer': study_bootstrap_diff(
            df, oof_full['peptide'], oof_full['shared_6mer']),
    }
    for k, v in out['gaps'].items():
        print(f'    {k:<28} {v["diff"]:+.3f}  CI [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]', flush=True)

    # ---- 3a: drop the dominant study
    print('\n[3a] without the largest study', flush=True)
    top = df['reference_id'].value_counts().index[0]
    keep = (df['reference_id'] != top).values
    d2 = df[keep].reset_index(drop=True)
    oof_nd = {}
    out['drop_dominant'] = {'dropped_study_pairs': int((~keep).sum()), 'n_pairs': int(keep.sum()),
                            'regimes': run_regimes(d2, X[keep], y[keep], standard_regimes(d2),
                                                   {'gbm': models['gbm']}, tag='[no top study] ',
                                                   keep=oof_nd)}
    out['drop_dominant']['gaps'] = {
        'random_minus_study': study_bootstrap_diff(d2, oof_nd['random'], oof_nd['study']),
        'peptide_minus_study': study_bootstrap_diff(d2, oof_nd['peptide'], oof_nd['study']),
    }
    for k, v in out['drop_dominant']['gaps'].items():
        print(f'    {k:<28} {v["diff"]:+.3f}  CI [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]', flush=True)

    # ---- 3b: within a single assay family
    print('\n[3b] inside one assay family', flush=True)
    fam, top3 = assay_family_by_pair()
    pairkey = (df['peptide'] + '|' + df['HLA']).values
    out['assay_strata'] = {}
    for a in top3:
        sel = np.array([fam.get(k) == {a} for k in pairkey])
        sub = df[sel].reset_index(drop=True)
        n_st = sub['reference_id'].nunique()
        if len(sub) < MIN_STRATUM or n_st < MIN_STRATUM_STUDIES:
            print(f'  {a[:30]}: skipped (n={len(sub)}, studies={n_st})', flush=True)
            continue
        print(f'  {a[:30]}: {len(sub)} pairs from {n_st} studies, {sub["label"].mean():.1%} positive',
              flush=True)
        out['assay_strata'][a] = {'n_pairs': int(len(sub)), 'n_studies': int(n_st),
                                  'regimes': run_regimes(sub, X[sel], y[sel], standard_regimes(sub),
                                                         {'gbm': models['gbm']}, tag='  ')}

    # ---- 4: forward-in-time holdout by publication order
    print('\n[4] train on earlier publications, test on later ones', flush=True)
    raw = pd.read_csv(RAW, usecols=['reference_id', 'pubmed_id'])
    pm = (raw.dropna().assign(pubmed_id=lambda r: pd.to_numeric(r['pubmed_id'], errors='coerce'))
          .dropna().groupby('reference_id')['pubmed_id'].min())
    df['pmid'] = df['reference_id'].map(pm)
    dated = df['pmid'].notna().values
    order = df[dated].sort_values('pmid')
    cut = order['pmid'].iloc[int(0.8 * len(order))]
    tr = (df['pmid'] <= cut).values & dated
    te = (df['pmid'] > cut).values & dated
    print(f'  dated pairs {dated.sum()}  train {tr.sum()} (PMID <= {int(cut)})  test {te.sum()}',
          flush=True)
    temporal = {'n_dated': int(dated.sum()), 'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                'cutoff_pmid': int(cut), 'test_studies': int(df[te]['reference_id'].nunique()),
                'test_share_positive': float(y[te].mean())}
    # how concentrated the test pairs are decides how wide a study-level interval must be
    sizes = df[te]['reference_id'].value_counts()
    temporal['test_top10_study_share'] = float(sizes.iloc[:10].sum() / sizes.sum())
    temporal['test_largest_study_pairs'] = int(sizes.iloc[0])
    scores = {}
    for name, make in models.items():
        scores[name] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    prior = pd.Series(y[tr]).groupby(df['HLA'].values[tr]).mean()
    scores['allele_prior'] = pd.Series(df['HLA'].values[te]).map(prior).fillna(y[tr].mean()).values
    dt = df[te].reset_index(drop=True)
    for name, s in scores.items():
        d = dt.copy()
        d['score'] = s
        wa = {} if name == 'allele_prior' else within_allele_auc(d, 'score')[1]
        temporal[name] = {'pooled_auc': float(roc_auc_score(d['label'], s)),
                          'pooled_auc_ci95': study_bootstrap_auc(dt, s),
                          'within_allele': float(wa['weighted_auc']) if wa else None}
        ci = temporal[name]['pooled_auc_ci95']
        wtxt = f'{temporal[name]["within_allele"]:.3f}' if wa else ' n/a '
        print(f'  {name:<13} pooled {temporal[name]["pooled_auc"]:.3f} '
              f'[{ci[0]:.3f}, {ci[1]:.3f}]  within-allele {wtxt}', flush=True)
    out['temporal'] = temporal

    (RESULTS / 'split_robustness.json').write_text(json.dumps(out, indent=2))
    print('\nsaved', RESULTS / 'split_robustness.json', flush=True)


if __name__ == '__main__':
    main()
