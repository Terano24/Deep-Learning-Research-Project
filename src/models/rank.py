"""
Study-stratified pairwise ranking (RankNet-style) for peptide immunogenicity.

Why. src/analysis/within_study.py shows the pointwise model scores ~0.83 pooled but only
~0.55 when asked to rank peptides inside a single unseen study. The pointwise objective is
partly to blame: because most IEDB studies are label-homogeneous, minimising per-record
cross-entropy rewards guessing the study's base rate, and study identity is inferable from
the peptide/allele distribution. Nothing in that loss asks the model to separate two
peptides measured under the same conditions.

Method. Train on *comparisons instead of labels*. Sample (immunogenic, non-immunogenic)
peptide pairs that share a study and an HLA allele, and optimise

    L = BCE( sigmoid(s(x+) - s(x-)), 1 )

with one shared scorer s (the pretrained pocket-attention trunk, pre-sigmoid logit).
Study base rates cancel in the difference, so the only way to reduce the loss is to find
peptide-level features that distinguish the two members of a pair. Evaluation is on
held-out studies, scoring each peptide independently with s.

Run:
  python -m src.models.rank --epochs 15 --pairs-per-group 400
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src.data.dataset import OUT_PATH
from src.models.improve import load_dataset, _load_pretrained, PRETRAIN_DIR
from src.models.pocket_attention import build_model
from src.analysis.within_study import per_study_auc
from src.analysis.confounds import within_allele_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
SEED = 42
INPUT_NAMES = ['peptide', 'hla', 'lung_flag', 'aux']


def build_scorer(n_aux, use_lung_adapter=True, pretrained=True):
    """The pocket-attention model, exposed as a scalar scorer (pre-sigmoid logit)."""
    model = build_model(use_lung_adapter=use_lung_adapter, n_aux=n_aux)
    if pretrained and (PRETRAIN_DIR / 'weights.index').exists():
        _load_pretrained(model)
    logit = model.get_layer('immunogenicity').input
    return tf.keras.Model(model.inputs, logit, name='scorer')


def build_ranker(scorer):
    """Siamese wrapper: P(x_a ranks above x_b) = sigmoid(s(a) - s(b))."""
    a = [tf.keras.Input(shape=t.shape[1:], dtype=t.dtype, name=f'a_{n}')
         for t, n in zip(scorer.inputs, INPUT_NAMES)]
    b = [tf.keras.Input(shape=t.shape[1:], dtype=t.dtype, name=f'b_{n}')
         for t, n in zip(scorer.inputs, INPUT_NAMES)]
    diff = tf.keras.layers.Subtract()([scorer(a), scorer(b)])
    out = tf.keras.layers.Activation('sigmoid', name='ranked_above')(diff)
    return tf.keras.Model(a + b, out)


def sample_pairs(df, mask, rng, per_group, group_cols=('reference_id', 'HLA')):
    """Index pairs (positive, negative) drawn from within one study+allele group.

    Returns *positional* row indices into the feature arrays, carried explicitly so that
    group-local offsets are never confused with global positions.
    """
    sub = df.loc[mask, list(group_cols) + ['label']].copy()
    sub['_row'] = np.where(mask)[0]
    pos_rows, neg_rows = [], []
    for _, g in sub.groupby(list(group_cols)):
        p = g.loc[g['label'] == 1, '_row'].values
        n = g.loc[g['label'] == 0, '_row'].values
        if len(p) == 0 or len(n) == 0:
            continue
        k = int(min(per_group, len(p) * len(n)))
        pos_rows.append(rng.choice(p, k, replace=len(p) < k))
        neg_rows.append(rng.choice(n, k, replace=len(n) < k))
    if not pos_rows:
        return np.array([], int), np.array([], int)
    return np.concatenate(pos_rows), np.concatenate(neg_rows)


def inputs_at(data, rows, af, lf):
    _, pep, hla, _, _ = data
    return [pep[rows], hla[rows], lf[rows], af[rows]]


def main(args):
    data = load_dataset()
    df, pep, hla, lung, aux = data
    seed = getattr(args, 'seed', SEED)
    rng = np.random.default_rng(seed)

    # study-grouped split: held-out studies are entirely unseen
    train = (df['study_split'] == 'train').values
    test = (df['study_split'] == 'test').values
    folds = df['study_fold'].values
    # one whole study (all-negative, 37% of pairs) lands in a single fold and yields no
    # comparable pairs, so pick the first validation fold that actually produces some
    val_fold, va_p, va_n = None, np.array([], int), np.array([], int)
    for k in sorted({f for f in folds if f >= 0}):
        cand = train & (folds == k)
        p, n = sample_pairs(df, cand, rng, args.pairs_per_group)
        if len(p):
            val_fold, va_p, va_n = k, p, n
            break
    if val_fold is None:
        raise SystemExit('no validation fold yields comparable pairs')
    val = train & (folds == val_fold)
    fit = train & (folds != val_fold) & (folds >= 0)
    print(f'validation fold: {val_fold}', flush=True)

    mu, sd = aux[fit].mean(0), aux[fit].std(0)
    af = ((aux - mu) / np.where(sd > 0, sd, 1.0)).astype('float32')
    lf = lung.astype('float32')

    tr_p, tr_n = sample_pairs(df, fit, rng, args.pairs_per_group)
    print(f'training pairs: {len(tr_p)}  validation pairs: {len(va_p)}', flush=True)
    if len(tr_p) == 0:
        raise SystemExit('no comparable pairs found -- check study/allele grouping')

    tf.keras.utils.set_random_seed(seed)
    scorer = build_scorer(aux.shape[1], use_lung_adapter=True, pretrained=not args.no_pretrain)
    ranker = build_ranker(scorer)
    ranker.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='binary_crossentropy')

    x_tr = inputs_at(data, tr_p, af, lf) + inputs_at(data, tr_n, af, lf)
    x_va = inputs_at(data, va_p, af, lf) + inputs_at(data, va_n, af, lf)
    ranker.fit(x_tr, np.ones(len(tr_p), 'float32'),
               validation_data=(x_va, np.ones(len(va_p), 'float32')),
               epochs=args.epochs, batch_size=args.batch, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                           restore_best_weights=True)])

    all_rows = np.arange(len(df))
    scores = scorer.predict(inputs_at(data, all_rows, af, lf), batch_size=512, verbose=0).ravel()
    d = df.copy()
    d['score'] = scores

    out = {'n_train_pairs': int(len(tr_p)), 'epochs': args.epochs,
           'pairs_per_group': args.pairs_per_group}
    for name, mask in [('held_out_studies', test), ('all_pairs', np.ones(len(df), bool))]:
        sub = d[mask]
        _, ws = per_study_auc(sub, 'score')
        _, wsa = per_study_auc(sub, 'score', by_allele=True)
        _, wa = within_allele_auc(sub, 'score')
        pooled = roc_auc_score(sub['label'], sub['score']) if sub['label'].nunique() > 1 else None
        out[name] = {'n': int(mask.sum()), 'pooled_auc': float(pooled) if pooled else None,
                     'within_study': ws, 'within_study_and_allele': wsa, 'within_allele': wa}
        print(f'\n--- {name} (n={int(mask.sum())}) ---')
        if pooled:
            print(f'  pooled AUC              {pooled:.3f}')
        if wa:
            print(f'  within-allele AUC       {wa["weighted_auc"]:.3f}')
        if ws:
            print(f'  within-study AUC        macro {ws["macro_auc"]:.3f}  weighted {ws["weighted_auc"]:.3f}'
                  f'  ({ws["n_groups"]} studies)')
            print(f'  studies above chance    {ws["frac_groups_above_chance"]:.0%}')
        if wsa:
            print(f'  within-study+allele     macro {wsa["macro_auc"]:.3f}  ({wsa["n_groups"]} groups)')

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'rank_model{args.tag}.json').write_text(json.dumps(out, indent=2, default=str))
    d[['peptide', 'HLA', 'label', 'lung_cancer', 'reference_id', 'study_split', 'score']].to_csv(
        RESULTS / f'rank_scores{args.tag}.csv', index=False)
    out_dir = BASE_DIR / 'models' / 'POCKET_RANKER'
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer.save_weights(str(out_dir / 'weights'))
    print('\nsaved', RESULTS / 'rank_model.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--pairs-per-group', type=int, default=400)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--no-pretrain', action='store_true')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--tag', default='')
    main(ap.parse_args())
