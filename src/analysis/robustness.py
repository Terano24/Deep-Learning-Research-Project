"""
H2 + H3 + H4 in one framework: is the ranking objective actually needed?

The obvious reviewer challenge is that study-level confounding has cheap fixes -- drop
the one study that dominates the data, reweight by study size, or train only on studies
that reported both outcomes. If any of those matches the pairwise ranking objective,
the method is not worth a paper.

All variants share the SAME architecture (pocket cross-attention, binding-pretrained,
aux features) and differ only in how they are trained. Every variant is evaluated on
study-grouped out-of-fold predictions, so each row is scored by a model that never saw
its publication. That also fixes the small-evaluation problem: instead of the 10 groups
available in one held-out split, within-study AUC is measured across every study.

  pointwise                standard per-record cross-entropy (the baseline)
  pointwise_drop_dominant  same, minus the all-negative study holding 37% of pairs
  pointwise_study_weighted same, rows weighted by 1/sqrt(study size)
  pointwise_mixed_only     same, trained only on studies reporting both outcomes
  ranking                  study+allele-matched pairwise ranking (our method)

Run:  python -m src.analysis.robustness --folds 5
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models.improve import load_dataset, compile_model, balanced_weights, _load_pretrained
from src.models.pocket_attention import build_model
from src.models.rank import build_scorer, build_ranker, sample_pairs
from src.analysis.confounds import within_allele_auc
from src.analysis.within_study import per_study_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
SEED = 42
MIN_GROUP = 10          # lower than the earlier 20, to widen the evaluation base
SCHEMES = ['pointwise', 'pointwise_drop_dominant', 'pointwise_study_weighted',
           'pointwise_mixed_only', 'ranking']
# '_nofeat' suffix: identical training, but the biochemical/self-similarity feature
# vector is withheld. Used to test whether the feature gain survives study-grouped
# evaluation or was a study artifact (see src/analysis/what_survives.py).


def standardise(aux, fit_mask):
    mu, sd = aux[fit_mask].mean(0), aux[fit_mask].std(0)
    return ((aux - mu) / np.where(sd > 0, sd, 1.0)).astype('float32')


def study_val_split(df, train_mask, rng, frac=0.15):
    """Hold out whole studies from the training portion, for early stopping."""
    studies = df.loc[train_mask, 'reference_id'].unique()
    n = max(1, int(len(studies) * frac))
    val_studies = set(rng.choice(studies, n, replace=False))
    val = train_mask & df['reference_id'].isin(val_studies).values
    return train_mask & ~val, val


def train_pointwise(scheme, df, data, fit, val, af, lf, epochs=25):
    _, pep, hla, _, aux = data
    y = df['label'].values.astype('float32')
    dominant = df['reference_id'].value_counts().idxmax()

    use_feat = not scheme.endswith('_nofeat')
    scheme = scheme.replace('_nofeat', '')

    use = fit.copy()
    if scheme == 'pointwise_drop_dominant':
        use &= (df['reference_id'] != dominant).values
    elif scheme == 'pointwise_mixed_only':
        mixed = df.groupby('reference_id')['label'].transform('nunique') > 1
        use &= mixed.values
    if use.sum() == 0 or len(set(y[use])) < 2:
        return None

    w = balanced_weights(y[use])
    if scheme == 'pointwise_study_weighted':
        counts = df['reference_id'].value_counts()
        sizes = df.loc[use, 'reference_id'].map(counts).values.astype(float)
        sw = 1.0 / np.sqrt(sizes)
        w = w * (sw / sw.mean())

    tf.keras.utils.set_random_seed(SEED)
    n_aux = aux.shape[1] if use_feat else 0
    if not use_feat:
        af = np.zeros((len(df), 1), 'float32')
    model = build_model(use_lung_adapter=True, n_aux=n_aux)
    _load_pretrained(model)
    compile_model(model, 'finetune', 3e-4)
    dummy = lambda m: np.zeros(int(m.sum()), 'float32')
    xs = lambda m: [pep[m], hla[m], lf[m], af[m]]
    model.fit(xs(use), {'binding': dummy(use), 'immunogenicity': y[use]},
              validation_data=(xs(val), {'binding': dummy(val), 'immunogenicity': y[val]}),
              epochs=epochs, batch_size=128, verbose=0,
              sample_weight={'binding': dummy(use), 'immunogenicity': w.astype('float32')},
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                          restore_best_weights=True)])
    return lambda mask: model.predict(xs(mask), batch_size=512, verbose=0)[1].ravel()


def train_ranking(df, data, fit, val, af, lf, rng, pairs_per_group=800, epochs=15,
                  use_feat=True):
    _, pep, hla, _, aux = data
    n_aux = aux.shape[1] if use_feat else 0
    if not use_feat:
        af = np.zeros((len(df), 1), 'float32')
    tr_p, tr_n = sample_pairs(df, fit, rng, pairs_per_group)
    va_p, va_n = sample_pairs(df, val, rng, pairs_per_group)
    if len(tr_p) == 0:
        return None
    tf.keras.utils.set_random_seed(SEED)
    scorer = build_scorer(n_aux, use_lung_adapter=True, pretrained=True)
    ranker = build_ranker(scorer)
    ranker.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy')
    at = lambda rows: [pep[rows], hla[rows], lf[rows], af[rows]]
    kw, cb = {}, []
    if len(va_p):
        kw['validation_data'] = (at(va_p) + at(va_n), np.ones(len(va_p), 'float32'))
        cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                   restore_best_weights=True))
    ranker.fit(at(tr_p) + at(tr_n), np.ones(len(tr_p), 'float32'),
               epochs=epochs, batch_size=256, verbose=0, callbacks=cb, **kw)
    return lambda mask: scorer.predict(at(np.where(mask)[0]), batch_size=512, verbose=0).ravel()


def evaluate(df, oof, tag):
    d = df.copy()
    d['score'] = oof
    d = d[~d['score'].isna()]
    _, ws = per_study_auc(d, 'score', min_n=MIN_GROUP)
    _, wsa = per_study_auc(d, 'score', min_n=MIN_GROUP, by_allele=True)
    _, wa = within_allele_auc(d, 'score', min_n=MIN_GROUP)
    res = {
        'n_scored': int(len(d)),
        'pooled_auc': float(roc_auc_score(d['label'], d['score'])),
        'pooled_pr_auc': float(average_precision_score(d['label'], d['score'])),
        'within_allele': wa, 'within_study': ws, 'within_study_and_allele': wsa,
    }
    nan = float('nan')
    print(f'  {tag:>26}  pooled {res["pooled_auc"]:.3f}  within-allele '
          f'{wa.get("weighted_auc", nan):.3f}  within-study {ws.get("macro_auc", nan):.3f}'
          f'  within-study+allele {wsa.get("macro_auc", nan):.3f}'
          f'  ({wsa.get("n_groups", 0)} groups)', flush=True)
    return res


def main(args):
    data = load_dataset()
    df = data[0]
    _, pep, hla, lung, aux = data
    lf = lung.astype('float32')
    rng = np.random.default_rng(SEED)
    folds = list(GroupKFold(args.folds).split(df, groups=df['reference_id'].values))

    out_path = RESULTS / 'robustness.json'
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.update({'folds': args.folds, 'min_group': MIN_GROUP})
    for scheme in args.schemes:
        oof = np.full(len(df), np.nan)
        for k, (tr_idx, va_idx) in enumerate(folds):
            train_mask = np.zeros(len(df), bool); train_mask[tr_idx] = True
            test_mask = np.zeros(len(df), bool); test_mask[va_idx] = True
            fit, val = study_val_split(df, train_mask, rng)
            if val.sum() == 0 or len(set(df['label'].values[val])) < 2:
                fit, val = train_mask, train_mask
            af = standardise(aux, fit)
            if scheme.startswith('ranking'):
                predict = train_ranking(df, data, fit, val, af, lf, rng,
                                        use_feat=not scheme.endswith('_nofeat'))
            else:
                predict = train_pointwise(scheme, df, data, fit, val, af, lf)
            if predict is None:
                print(f'  {scheme} fold {k}: not trainable, skipped', flush=True)
                continue
            oof[test_mask] = predict(test_mask)
            print(f'  {scheme} fold {k} done', flush=True)
        if np.isnan(oof).all():
            print(f'  {scheme}: no predictions produced', flush=True)
            continue
        out[scheme] = evaluate(df, oof, scheme)
        np.save(RESULTS / f'oof_study_{scheme}.npy', oof)
        out_path.write_text(json.dumps(out, indent=2))
    print('\nsaved', out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--schemes', nargs='+', default=SCHEMES)
    main(ap.parse_args())
