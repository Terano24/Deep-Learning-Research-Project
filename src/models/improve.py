"""
Experiments for the pocket-anchor cross-attention model.

  python -m src.models.improve pretrain
  python -m src.models.improve cv --variants scratch pretrained pretrained_lung --folds 3
  python -m src.models.improve final --variant pretrained_lung --seeds 5

Variants
  scratch          cross-attention net, trained only on the immunogenicity data
  pretrained       encoder pretrained on IEDB MHC binding/presentation, then fine-tuned
  pretrained_lung  pretrained + lung-cancer adapter head (gated residual logit)
  scratch_lung     adapter head without pretraining (ablation)
All CV model selection uses the training folds only; the test split is touched by `final`.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset import OUT_PATH, HLA_PATH
from src.data.features import get_features, FEATURE_COLS
from src.models.pocket_attention import build_model, tokenize_peptide, tokenize_hla
from src.models.train import metrics, bootstrap_auc
from src.analysis.confounds import within_allele_auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BINDING_PATH = BASE_DIR / 'data' / 'raw' / 'iedb_mhc_binding.csv'
PRETRAIN_DIR = BASE_DIR / 'models' / 'POCKET_PRETRAINED'
RESULTS = BASE_DIR / 'results'
VARIANTS = {
    'scratch': dict(pretrained=False, lung=False),
    'scratch_lung': dict(pretrained=False, lung=True),
    'pretrained': dict(pretrained=True, lung=False),
    'pretrained_lung': dict(pretrained=True, lung=True),
    'pretrained_feat': dict(pretrained=True, lung=False, feat=True),
    'pretrained_lung_feat': dict(pretrained=True, lung=True, feat=True),
    'scratch_lung_feat': dict(pretrained=False, lung=True, feat=True),
}
for _c in VARIANTS.values():
    _c.setdefault('feat', False)


def hla_table():
    t = pd.read_csv(HLA_PATH, sep='\t')
    return {h: np.array(tokenize_hla(s), dtype='int32') for h, s in zip(t['HLA'], t['pseudo'])}


def tensors(df, hla_tok):
    pep = np.array([tokenize_peptide(p) for p in df['peptide']], dtype='int32')
    hla = np.stack([hla_tok[h] for h in df['HLA']])
    return pep, hla


def load_dataset():
    df = pd.read_csv(OUT_PATH)
    pep, hla = tensors(df, hla_table())
    lung = df['lung_cancer'].values.astype('float32')[:, None]
    # join on the peptide key: get_features returns one row per *unique* peptide
    aux = df.join(get_features(df['peptide']), on='peptide')[FEATURE_COLS].astype('float32').values
    assert not np.isnan(aux).any(), 'aux feature join left NaNs'
    return df, pep, hla, lung, aux


def load_binding(exclude_peptides):
    b = pd.read_csv(BINDING_PATH).dropna(subset=['linear_sequence', 'mhc_restriction', 'qualitative_measure'])
    hla_tok = hla_table()
    b = b[b['mhc_restriction'].str.match(r'^HLA-[ABC]\*\d\d:\d\d$')]
    b['HLA'] = b['mhc_restriction'].str.replace(':', '', regex=False)
    b = b[b['HLA'].isin(hla_tok) & b['linear_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]{9,10}$')]
    b = b[~b['linear_sequence'].isin(exclude_peptides)]          # no peptide overlap with the task data
    b['label'] = b['qualitative_measure'].str.startswith('Positive').astype(int)
    g = b.groupby(['linear_sequence', 'HLA'])['label'].mean().reset_index()
    g = g[g['label'] != 0.5]
    g['label'] = (g['label'] > 0.5).astype(int)
    return g.rename(columns={'linear_sequence': 'peptide'}), hla_tok


def compile_model(model, task, lr):
    w = {'binding': 1.0, 'immunogenicity': 0.0} if task == 'pretrain' else {'binding': 0.0, 'immunogenicity': 1.0}
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={'binding': 'binary_crossentropy', 'immunogenicity': 'binary_crossentropy'},
                  loss_weights=w)


def balanced_weights(y):
    pos = max(float(y.sum()), 1.0)
    return np.where(y == 1, (len(y) - pos) / pos, 1.0).astype('float32')


def pretrain(args):
    df = pd.read_csv(OUT_PATH)
    b, hla_tok = load_binding(set(df['peptide']))
    print('binding pretrain pairs:', len(b), 'positive rate', round(b['label'].mean(), 3), flush=True)
    pep, hla = tensors(b, hla_tok)
    y = b['label'].values.astype('float32')
    rng = np.random.default_rng(42)
    val = rng.random(len(y)) < 0.05
    tf.keras.utils.set_random_seed(42)
    model = build_model()
    compile_model(model, 'pretrain', 1e-3)
    z = np.zeros((len(y), 1), 'float32')
    ds = lambda m: ([pep[m], hla[m], z[m], z[m]], {'binding': y[m], 'immunogenicity': np.zeros(m.sum(), 'float32')})
    hist = model.fit(*ds(~val), validation_data=ds(val), epochs=args.epochs, batch_size=256,
                     sample_weight={'binding': balanced_weights(y[~val]), 'immunogenicity': np.zeros((~val).sum(), 'float32')},
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
                     verbose=2)
    pv = model.predict([pep[val], hla[val], z[val], z[val]], batch_size=1024, verbose=0)[0].ravel()
    from sklearn.metrics import roc_auc_score
    print('pretrain val AUC (binding):', round(roc_auc_score(y[val], pv), 4))
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(PRETRAIN_DIR / 'weights'))


def fit_variant(variant, tr, va, data, seed, epochs=25):
    df, pep, hla, lung, aux = data
    cfg = VARIANTS[variant]
    tf.keras.utils.set_random_seed(seed)
    n_aux = aux.shape[1] if cfg['feat'] else 0
    model = build_model(use_lung_adapter=cfg['lung'], n_aux=n_aux)
    if cfg['pretrained']:
        _load_pretrained(model)
    compile_model(model, 'finetune', 3e-4 if cfg['pretrained'] else 1e-3)
    y = df['label'].values.astype('float32')
    lf = lung if cfg['lung'] else np.zeros_like(lung)
    # standardise aux features with TRAINING-fold statistics only
    if n_aux:
        mu, sd = aux[tr].mean(0), aux[tr].std(0)
        af = ((aux - mu) / np.where(sd > 0, sd, 1.0)).astype('float32')
    else:
        af = np.zeros((len(df), 1), 'float32')
    dummy = lambda m: np.zeros(m.sum(), 'float32')
    xs = lambda m: [pep[m], hla[m], lf[m], af[m]]
    ys = lambda m: {'binding': dummy(m), 'immunogenicity': y[m]}
    kw = {}
    cb = []
    if va is not None:
        kw['validation_data'] = (xs(va), ys(va))
        cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))
    hist = model.fit(xs(tr), ys(tr), epochs=epochs, batch_size=128, callbacks=cb, verbose=0,
                     sample_weight={'binding': dummy(tr), 'immunogenicity': balanced_weights(y[tr])}, **kw)
    best = int(np.argmin(hist.history['val_loss'])) + 1 if va is not None else epochs
    return model, best, (lf, af)


def _load_pretrained(model):
    """Copy pretrained weights by layer name. Layers absent from the pretrained graph
    (the lung adapter, the aux projection) and any whose shape changed (the trunk, when
    aux features widen its input) keep their fresh initialisation."""
    src = build_model(use_lung_adapter=False, n_aux=0)
    src.load_weights(str(PRETRAIN_DIR / 'weights'))
    copied, skipped = 0, []
    for layer in src.layers:
        if not layer.weights:
            continue
        try:
            target = model.get_layer(layer.name)
        except ValueError:
            skipped.append(layer.name)
            continue
        if [w.shape for w in target.weights] == [w.shape for w in layer.weights]:
            target.set_weights(layer.get_weights())
            copied += 1
        else:
            skipped.append(layer.name)
    if copied == 0:
        raise RuntimeError('pretrained transfer copied nothing -- layer names changed?')
    return copied, skipped


def predict(model, data, mask, lfaf):
    df, pep, hla, lung, aux = data
    lf, af = lfaf
    return model.predict([pep[mask], hla[mask], lf[mask], af[mask]],
                         batch_size=512, verbose=0)[1].ravel()


def cv(args):
    data = load_dataset()
    df = data[0]
    train = (df['split'] == 'train').values
    out = {}
    for v in args.variants:
        oof = np.full(len(df), np.nan)
        fold_auc, best_eps = [], []
        for k in range(args.folds):
            va = train & (df['fold'].values == k)
            tr = train & (df['fold'].values != k)
            model, best, lf = fit_variant(v, tr, va, data, seed=42)
            oof[va] = predict(model, data, va, lf)
            m = metrics(df['label'].values[va], oof[va])
            fold_auc.append(m['auc']); best_eps.append(best)
            print(f'{v:>16} fold {k}: AUC {m["auc"]:.3f} PR {m["pr_auc"]:.3f} MCC {m["mcc"]:.3f} ep {best}', flush=True)
        done = ~np.isnan(oof)
        lung = df['lung_cancer'].values
        scored = df[done][['peptide', 'HLA', 'label', 'lung_cancer']].copy()
        scored['score'] = oof[done]
        _, wa = within_allele_auc(scored, 'score')
        res = {'cv_auc_mean': float(np.mean(fold_auc)), 'cv_auc_std': float(np.std(fold_auc)),
               'all': metrics(df['label'].values[done], oof[done]),
               'lung': metrics(df['label'].values[done & lung], oof[done & lung]),
               'within_allele': wa, 'best_epochs': best_eps}
        out[v] = res
        print(f'== {v}: AUC {res["cv_auc_mean"]:.3f}±{res["cv_auc_std"]:.3f} | within-allele '
              f'{wa.get("weighted_auc", float("nan")):.3f} | pooled lung AUC '
              f'{res["lung"]["auc"]:.3f} (n={res["lung"]["n"]}, pos={res["lung"]["n_pos"]})', flush=True)
        scored.to_csv(RESULTS / f'cv_oof_{v}.csv', index=False)
    (RESULTS / f'improve_cv_{"_".join(args.variants)}_f{args.folds}.json').write_text(json.dumps(out, indent=2))


def final(args):
    data = load_dataset()
    df, pep, hla, lung, aux = data
    train = (df['split'] == 'train').values
    test = (df['split'] == 'test').values
    v = args.variant
    preds, epochs = [], []
    for s in range(args.seeds):
        # early-stop epoch from one CV fold per seed, then refit on all training data
        va = train & (df['fold'].values == s % 10)
        _, best, _ = fit_variant(v, train & ~va, va, data, seed=100 + s)
        epochs.append(best)
        model, _, lf = fit_variant(v, train, None, data, seed=100 + s, epochs=best)
        preds.append(predict(model, data, test, lf))
        print('seed', s, 'epochs', best, flush=True)
    p = np.mean(preds, axis=0)
    y = df['label'].values[test]
    lg = df['lung_cancer'].values[test]
    res = {'variant': v, 'seeds': args.seeds, 'epochs': epochs,
           'all': metrics(y, p), 'lung': metrics(y[lg], p[lg])}
    res['all']['auc_ci95'] = bootstrap_auc(y, p)
    res['lung']['auc_ci95'] = bootstrap_auc(y[lg], p[lg])
    print(json.dumps(res, indent=2))
    (RESULTS / f'improve_final_{v}.json').write_text(json.dumps(res, indent=2))
    out = df[test][['peptide', 'HLA', 'label', 'lung_cancer']].copy()
    out['pred'] = p
    out.to_csv(RESULTS / f'improve_test_predictions_{v}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['pretrain', 'cv', 'final'])
    ap.add_argument('--variants', nargs='+', default=['scratch'])
    ap.add_argument('--variant', default='pretrained_lung')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args()
    {'pretrain': pretrain, 'cv': cv, 'final': final}[a.stage](a)
