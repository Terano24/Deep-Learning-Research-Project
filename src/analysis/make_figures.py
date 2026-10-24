"""
Publication figures for the manuscript.

Rendered for print rather than for screen: greyscale fills with hatching and distinct
markers, so every figure survives black-and-white printing and is colourblind-safe by
construction rather than by palette choice. Serif type matches the manuscript. Each
figure is written as both PDF (vector, for submission) and PNG (300 dpi, for preprint
servers and slides).

Numbers are read from results/*.json so the figures cannot drift from the analyses.

Run:  python -m src.analysis.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
FIGDIR = BASE_DIR / 'docs' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.edgecolor': '0.2',
    'axes.linewidth': 0.7,
    'xtick.color': '0.2',
    'ytick.color': '0.2',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# greyscale steps: light -> dark, used as an ordered sequence, never cycled
G = {'light': '0.82', 'mid': '0.58', 'dark': '0.30', 'ink': '0.12'}
CHANCE = 0.5


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{stem}.{ext}')
    plt.close(fig)
    print(f'  wrote {stem}.pdf / .png', flush=True)


def chance_line(ax, y=CHANCE, label='chance'):
    """Dashed reference at 0.5, labelled at the left margin where marks never reach."""
    ax.axhline(y, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(0.006, y), xycoords=('axes fraction', 'data'),
                xytext=(0, 2), textcoords='offset points',
                ha='left', va='bottom', fontsize=7, color=G['ink'])


# ----------------------------------------------------------------------------- Fig 1
def fig1_dataset():
    """Composition of the benchmark: study sizes and label homogeneity."""
    df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'dataset.csv')
    sizes = df.groupby('reference_id').size().sort_values(ascending=False)
    comp = df.groupby('reference_id')['label'].mean()
    allneg = int((comp == 0).sum()); allpos = int((comp == 1).sum())
    mixed = int(len(comp) - allneg - allpos)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.7),
                               gridspec_kw={'width_ratios': [1.45, 1]})

    a.plot(np.arange(1, len(sizes) + 1), sizes.values, color=G['ink'], lw=1.1)
    a.set_yscale('log')
    a.set_xlabel('study, ranked by size')
    a.set_ylabel('peptide–HLA pairs (log)')
    a.set_title('a  One study dominates the dataset', loc='left', fontweight='bold')
    top = sizes.iloc[0]
    a.scatter([1], [top], s=26, facecolor='white', edgecolor=G['ink'], zorder=3, lw=1.1)
    a.annotate(f'{top:,} pairs\n= {top/len(df):.0%} of all data\n(all negative)',
               xy=(1, top), xytext=(14, top * 0.42), fontsize=7.5, color=G['ink'],
               arrowprops=dict(arrowstyle='-', lw=0.7, color=G['ink']))

    bars = [('all negative', allneg, G['light'], '///'),
            ('all positive', allpos, G['mid'], '\\\\\\'),
            ('both outcomes', mixed, G['dark'], '')]
    y = np.arange(len(bars))
    for i, (lab, v, c, h) in enumerate(bars):
        b.barh(i, v, color=c, edgecolor=G['ink'], lw=0.7, hatch=h, height=0.62)
        b.annotate(f'{v}  ({v/len(comp):.0%})', xy=(v, i), xytext=(5, 0),
                   textcoords='offset points', va='center', fontsize=7.5, color=G['ink'])
    b.set_yticks(y); b.set_yticklabels([x[0] for x in bars])
    b.set_xlim(0, max(v for _, v, _, _ in bars) * 1.32)
    b.set_xlabel('number of studies')
    b.set_title(f'b  {(allneg+allpos)/len(comp):.0%} of studies report one outcome only',
                loc='left', fontweight='bold')
    b.invert_yaxis()
    save(fig, 'fig1_dataset_composition')


# ----------------------------------------------------------------------------- Fig 2
def fig2_splits():
    """The grouping unit, not the model, sets the reported number."""
    leak, ext = load('leakage_analysis.json'), load('external_validation.json')
    iedb = [('random', leak.get('random', {}).get('overall_auc')),
            ('peptide', leak.get('peptide', {}).get('overall_auc')),
            ('study', leak.get('study', {}).get('overall_auc'))]
    trap = [('random', ext.get('random', {}).get('pooled_auc')),
            ('peptide', ext.get('peptide', {}).get('pooled_auc')),
            ('antigen', ext.get('antigen', {}).get('pooled_auc')),
            ('organism', ext.get('organism', {}).get('pooled_auc'))]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for ax, data, title in zip(axes, [iedb, trap],
                               ['a  IEDB (this study)', 'b  TRAP (independent)']):
        data = [(k, v) for k, v in data if v is not None]
        x = np.arange(len(data))
        shades = [G['light'], G['light'], G['mid'], G['dark']][:len(data)]
        hatches = ['', '', '...', 'xxx'][:len(data)]
        for i, ((lab, v), c, h) in enumerate(zip(data, shades, hatches)):
            ax.bar(i, v, color=c, edgecolor=G['ink'], lw=0.8, hatch=h, width=0.62)
            ax.annotate(f'{v:.3f}', xy=(i, v), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8, color=G['ink'])
        ax.set_xticks(x); ax.set_xticklabels([k for k, _ in data])
        ax.set_ylim(0.45, 0.99)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('grouping unit for the split')
        chance_line(ax)
    axes[0].set_ylabel('pooled AUC')
    # the point of the figure: the first two bars are the same height. The bracket sits
    # above the value labels so it cannot collide with them.
    for ax in axes:
        ax.plot([0, 0, 1, 1], [0.925, 0.937, 0.937, 0.925], color=G['ink'], lw=0.7,
                clip_on=False)
        ax.annotate('same height — no protection', xy=(0.5, 0.945),
                    ha='center', va='bottom', fontsize=7.5, color=G['ink'])
    save(fig, 'fig2_grouping_unit')


# ----------------------------------------------------------------------------- Fig 3
def fig3_ceiling():
    """Measured label reliability, and its robustness."""
    rob, base = load('ceiling_robustness.json'), load('label_ceiling.json')
    rows = []
    hl = base.get('reproducibility_auc_ceiling', {})
    if hl:
        rows.append(('majority vote (headline)', hl['mean'], hl['ci95'], True))
    for k, lab in [('any_positive', 'any-positive'), ('ties_positive', 'ties → positive'),
                   ('ties_negative', 'ties → negative'),
                   ('raw_records_no_aggregation', 'raw records, no aggregation')]:
        r = rob.get('H3_rules', {}).get(k)
        if r:
            rows.append((lab, r['auc'], r.get('auc_ci95'), False))
    rows.append((None, None, None, None))
    for k in ['exactly 2 studies', 'exactly 3 studies', '4+ studies']:
        r = rob.get('H2_by_n_studies', {}).get(k)
        if r:
            rows.append((k, r['auc'], r.get('auc_ci95'), k == 'exactly 2 studies'))
    rows.append((None, None, None, None))
    for k, r in (rob.get('H5_slices') or {}).items():
        if r:
            rows.append((k.replace('assay::', '').replace('_', ' '), r['auc'],
                         r.get('auc_ci95'), False))

    fig, ax = plt.subplots(figsize=(7.1, 0.3 * len(rows) + 1.5))
    ypos, labels = [], []
    for i, (lab, v, ci, emph) in enumerate(rows):
        y = len(rows) - i
        if lab is None:
            continue
        ypos.append(y); labels.append(lab)
        if ci:
            ax.plot(ci, [y, y], color=G['ink'], lw=1.0, solid_capstyle='butt')
        ax.scatter([v], [y], s=34 if emph else 22, zorder=3,
                   facecolor=G['ink'] if emph else 'white',
                   edgecolor=G['ink'], lw=1.0)
        ax.annotate(f'{v:.3f}', xy=(v, y), xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=7.2, color=G['ink'])
    ax.axvspan(0.85, 0.90, color=G['light'], alpha=0.75, zorder=0)
    ax.annotate('published claims\n0.85–0.90', xy=(0.875, max(ypos) + 0.4),
                ha='center', va='bottom', fontsize=7.2, color=G['ink'])
    ax.axvline(CHANCE, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate('chance', xy=(CHANCE, min(ypos) - 0.6), ha='center', fontsize=7,
                color=G['ink'])
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlim(0.42, 0.95)
    ax.set_ylim(min(ypos) - 1.1, max(ypos) + 1.5)
    ax.set_xlabel('AUC of one study’s outcome predicting another’s, on the same pair')
    ax.set_title('Measured reliability of the labels, and its robustness',
                 loc='left', fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig, 'fig3_label_ceiling')


# ----------------------------------------------------------------------------- Fig 4
def fig4_models():
    """Predictors on the held-out test split, pooled vs within-allele."""
    eb = load('external_baselines.json').get('all', {})
    order = ['ours', 'gbm_baseline', 'deepimmuno', 'allele_prior_baseline',
             'bigmhc_im', 'mixmhcpred']
    pretty = {'ours': 'this study', 'gbm_baseline': 'gradient boosting',
              'deepimmuno': 'DeepImmuno', 'allele_prior_baseline': 'allele prior\n(no peptide)',
              'bigmhc_im': 'BigMHC_IM', 'mixmhcpred': 'MixMHCpred\n(binding only)'}
    rows = [(pretty[k], eb[k]['auc'], eb[k].get('within_allele'))
            for k in order if k in eb]

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    x = np.arange(len(rows)); w = 0.37
    pooled = [r[1] for r in rows]
    within = [r[2] if r[2] else np.nan for r in rows]
    ax.bar(x - w / 2, pooled, w, color=G['light'], edgecolor=G['ink'], lw=0.8,
           label='pooled AUC')
    ax.bar(x + w / 2, within, w, color=G['dark'], edgecolor=G['ink'], lw=0.8,
           hatch='///', label='within-allele AUC')
    for i, (p, v) in enumerate(zip(pooled, within)):
        ax.annotate(f'{p:.3f}', xy=(i - w / 2, p), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=7, color=G['ink'])
        if not np.isnan(v):
            ax.annotate(f'{v:.3f}', xy=(i + w / 2, v), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=7, color=G['ink'])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel('AUC')
    chance_line(ax)
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.set_title('A peptide-blind baseline outscores two published predictors',
                 loc='left', fontweight='bold')
    save(fig, 'fig4_model_comparison')


# ----------------------------------------------------------------------------- Fig 5
def fig5_contamination():
    """Seen vs unseen: training-set overlap inflates reported performance."""
    fv = load('fair_vs_deepimmuno.json')
    lb = load('lung_benchmark.json')
    pairs = []
    s = fv.get('seen_by_deepimmuno', {}).get('deepimmuno', {})
    u = fv.get('unseen_by_deepimmuno', {}).get('deepimmuno', {})
    if s and u:
        pairs.append(('DeepImmuno\n(37.5% overlap)', s['auc'], u['auc']))
    pairs.append(('BigMHC_IM\n(26.5% overlap)', 0.807, 0.549))

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (lab, seen, unseen) in enumerate(pairs):
        ax.plot([0, 1], [seen, unseen], color=G['ink'], lw=1.1, zorder=2,
                marker='o', ms=6, mfc='white', mec=G['ink'], mew=1.1)
        ax.annotate(f'{seen:.3f}', xy=(0, seen), xytext=(-6, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=7.5)
        ax.annotate(f'{unseen:.3f}', xy=(1, unseen), xytext=(6, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=7.5)
        ax.annotate(lab, xy=(0.5, (seen + unseen) / 2), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=7.5, color=G['ink'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['pairs it\ntrained on', 'pairs it\nnever saw'])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel('AUC')
    chance_line(ax)
    ax.set_title('Training-set overlap', loc='left', fontweight='bold')
    save(fig, 'fig5_contamination')


# ----------------------------------------------------------------------------- Fig 6
def fig6_honest():
    """Leakage-free performance against the measured ceiling."""
    df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'dataset.csv')
    variants = [('ranking (no features)', 'oof_study_ranking_nofeat.npy'),
                ('ranking (+ features)', 'oof_study_ranking.npy'),
                ('CNN (original architecture)', 'oof_study_cnn.npy'),
                ('study reweighting', 'oof_study_pointwise_study_weighted.npy'),
                ('pointwise (no features)', 'oof_study_pointwise_nofeat.npy'),
                ('pointwise', 'oof_study_pointwise.npy')]
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    rows = []
    for lab, fn in variants:
        p = RESULTS / fn
        if not p.exists():
            continue
        df['_s'] = np.load(p)
        groups = [g for _, g in df.groupby(['reference_id', 'HLA'])
                  if len(g) >= 10 and g['label'].nunique() > 1 and g['_s'].notna().all()]
        if len(groups) < 5:
            continue
        a = np.array([roc_auc_score(g['label'], g['_s']) for g in groups])
        boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
        rows.append((lab, a.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

    fig, ax = plt.subplots(figsize=(7.1, 0.42 * len(rows) + 1.6))
    ceil = load('label_ceiling.json').get('reproducibility_auc_ceiling', {}).get('mean', 0.637)
    cons = 0.681
    ax.axvspan(ceil, cons, color=G['light'], alpha=0.8, zorder=0)
    ax.annotate('measured label reliability\n0.637 – 0.681', xy=((ceil + cons) / 2, len(rows) + 0.55),
                ha='center', va='bottom', fontsize=7.2, color=G['ink'])
    ax.axvspan(0.85, 0.90, color=G['mid'], alpha=0.55, zorder=0)
    ax.annotate('published\nclaims', xy=(0.875, len(rows) + 0.55), ha='center',
                va='bottom', fontsize=7.2, color=G['ink'])
    for i, (lab, v, lo, hi) in enumerate(rows):
        y = len(rows) - i
        ax.plot([lo, hi], [y, y], color=G['ink'], lw=1.0)
        ax.scatter([v], [y], s=32, zorder=3, facecolor=G['ink'] if i == 0 else 'white',
                   edgecolor=G['ink'], lw=1.0)
        ax.annotate(f'{v:.3f}', xy=(v, y), xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=7.2, color=G['ink'])
    ax.axvline(CHANCE, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate('chance', xy=(CHANCE, 0.35), ha='center', fontsize=7, color=G['ink'])
    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0.44, 0.95)
    ax.set_ylim(0.1, len(rows) + 1.5)
    ax.set_xlabel('within-study, within-allele AUC (bootstrap over studies)')
    ax.set_title('Leakage-free performance against the measured ceiling',
                 loc='left', fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig, 'fig6_honest_performance')


if __name__ == '__main__':
    print('rendering figures ->', FIGDIR, flush=True)
    for fn in (fig1_dataset, fig2_splits, fig3_ceiling, fig4_models,
               fig5_contamination, fig6_honest):
        try:
            fn()
        except Exception as e:
            print(f'  !! {fn.__name__} failed: {type(e).__name__}: {e}', flush=True)
    print('done')
