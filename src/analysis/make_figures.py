"""
Publication figures for the manuscript.

Two figures only. Every result a table states adequately -- the split-scheme
comparison (3.4), the predictor comparison (3.5), the training-set overlap (3.6),
the negative-definition counts (3.7) -- is left to its table. A figure is drawn only
where the display does something a table cannot: show the shape of a distribution
(Fig 1), or place many interval estimates against shared reference bands (Fig 2).

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
CLAIMS = (0.85, 0.943)  # reported under random cross-validation; see Table 1 of the
                        # manuscript and data/external/reported_performance.csv
# labels that sit on top of a reference line punch a hole in it rather than cross it
KNOCKOUT = dict(facecolor='white', edgecolor='none', pad=1.2)

# Each figure is drawn twice. The full-width version is for the single-column
# manuscript and for preprint servers; the column version is for the two-column
# build, where a 7.1-inch figure scaled into a 3.2-inch column is unreadable. The
# narrow version is not a shrunk copy: type stays at print size and the wordier
# annotations are shortened, which is why NARROW is consulted inside each figure.
FULL_W, COL_W = 7.1, 3.25
NARROW = False


def width():
    return COL_W if NARROW else FULL_W


def pt(full, narrow):
    """Pick a size or a string by the variant being drawn."""
    return narrow if NARROW else full


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def save(fig, stem):
    stem = f'{stem}_col' if NARROW else stem
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{stem}.{ext}')
    plt.close(fig)
    print(f'  wrote {stem}.pdf / .png', flush=True)


def claims_band(ax, ytop, shade=G['light'], label=None):
    """The 0.85-0.90 band every estimate in this paper is measured against."""
    if label is None:
        label = pt('published claims\n0.85-0.90', 'published\nclaims')
    ax.axvspan(*CLAIMS, color=shade, alpha=0.8, zorder=0)
    ax.annotate(label, xy=(sum(CLAIMS) / 2, ytop), ha='center', va='bottom',
                fontsize=pt(7.2, 6.0), color=G['ink'])


def chance_line(ax, ytop):
    """Dashed reference at 0.5, labelled at the top where no estimate reaches."""
    ax.axvline(CHANCE, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate('chance', xy=(CHANCE, ytop), ha='center', va='bottom',
                fontsize=pt(7.2, 6.0), color=G['ink'], zorder=5, bbox=KNOCKOUT)


def forest_row(ax, y, value, ci=None, emphasis=False):
    """One estimate: interval, marker, value label.

    The label sits above the marker at full width, where rows are far enough apart.
    In the narrow variant rows are close enough that a label above one row lands on
    the interval of the row above it, so there it goes beside the interval instead.
    """
    if ci:
        ax.plot(ci, [y, y], color=G['ink'], lw=1.0, solid_capstyle='butt', zorder=2)
    ax.scatter([value], [y], s=(34 if emphasis else 22) * pt(1.0, 0.7), zorder=3,
               facecolor=G['ink'] if emphasis else 'white',
               edgecolor=G['ink'], lw=pt(1.0, 0.8))
    if NARROW:
        ax.annotate(f'{value:.3f}', xy=(max(ci) if ci else value, y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=6.0,
                    color=G['ink'], zorder=5, bbox=KNOCKOUT)
    else:
        ax.annotate(f'{value:.3f}', xy=(value, y), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=7.2,
                    color=G['ink'])


# ----------------------------------------------------------------------------- Fig 1
def fig1_dataset():
    """How unevenly the benchmark is spread across its source studies.

    A 1,365-study size distribution is either 1,365 table rows or a handful of
    arbitrary quantiles; the shape is the point, so this one stays a figure. Rank is
    on a log axis because everything that matters happens in the first 1% of studies,
    which a linear axis crushes against the origin.
    """
    df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'dataset.csv')
    sizes = df.groupby('reference_id').size().sort_values(ascending=False)
    n_pairs, n_studies = len(df), len(sizes)
    rank = np.arange(1, n_studies + 1)
    cum = np.cumsum(sizes.values) / n_pairs

    comp = df.groupby('reference_id')['label'].mean()
    single = comp[(comp == 0) | (comp == 1)].index
    single_share = df['reference_id'].isin(single).mean()

    fig, ax = plt.subplots(figsize=(width(), pt(3.1, 1.85)))
    ax.plot(rank, sizes.values, color=G['ink'], lw=1.2, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, n_studies * 1.35)
    ax.set_ylim(0.6, sizes.iloc[0] * pt(4.0, 6.0))
    ax.set_xlabel('study, ranked by size (log)', fontsize=pt(8.5, 7.0))
    ax.set_ylabel('peptide–HLA pairs\ncontributed (log)', fontsize=pt(8.5, 7.0))
    ax.tick_params(labelsize=pt(8, 6.5))

    top = sizes.iloc[0]
    ax.scatter([1], [top], s=pt(28, 20), facecolor='white', edgecolor=G['ink'],
               zorder=4, lw=1.1)
    ax.annotate(pt(f'largest study: {top:,} pairs,\n{top / n_pairs:.0%} of the benchmark,\n'
                   'every outcome negative',
                   f'largest study:\n{top:,} pairs ({top / n_pairs:.0%}),\nall negative'),
                xy=(1, top), xytext=(pt(2.4, 2.0), top * pt(0.95, 4.6)),
                fontsize=pt(7.5, 6.0), color=G['ink'], va='top', ha='left',
                arrowprops=dict(arrowstyle='-', lw=0.7, color=G['ink'],
                                shrinkA=0, shrinkB=4))

    # cumulative share on its own axis: the skew as a number the text can cite
    ax2 = ax.twinx()
    ax2.plot(rank, cum * 100, color=G['mid'], lw=1.3, ls=(0, (5, 2)), zorder=2)
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel(pt('cumulative share of all pairs (%)', 'cumulative share (%)'),
                   color=G['dark'], fontsize=pt(8.5, 7.0))
    ax2.tick_params(axis='y', colors=G['dark'], labelsize=pt(8, 6.5))
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(G['dark'])
    ax2.spines['top'].set_visible(False)
    # the top-10 label goes left of its marker: to the right the size curve crosses it
    for k, (dx, ha) in [(10, (-7, 'right')), (100, (7, 'left'))]:
        ax2.scatter([k], [cum[k - 1] * 100], s=pt(20, 14), facecolor=G['mid'],
                    edgecolor=G['ink'], lw=0.8, zorder=4)
        ax2.annotate(pt(f'top {k} studies = {cum[k - 1]:.0%}', f'top {k} = {cum[k - 1]:.0%}'),
                     xy=(k, cum[k - 1] * 100),
                     xytext=(dx, -3), textcoords='offset points', ha=ha, va='top',
                     fontsize=pt(7.2, 6.0), color=G['dark'], zorder=5, bbox=KNOCKOUT)

    ax.set_title(pt(f'{n_studies:,} studies, {n_pairs:,} pairs: {single_share:.0%} of pairs '
                    'come from studies that report a single outcome',
                    f'{n_studies:,} studies, {n_pairs:,} pairs; {single_share:.0%} of pairs\n'
                    'come from single-outcome studies'),
                 loc='left', fontweight='bold', fontsize=pt(9, 7.2))
    save(fig, 'fig1_dataset_composition')


# ----------------------------------------------------------------------------- Fig 2
def _ceiling_groups():
    """Label-reliability estimates, grouped by the robustness check each comes from."""
    rob = load('ceiling_robustness.json')
    groups = []

    rows = []
    # the headline comes from the same file as the other rows: label_ceiling.py computes
    # it too, but from an independent RNG, and the two differ in the fourth decimal
    hl = rob.get('H3_rules', {}).get('majority')
    if hl:
        rows.append(('majority vote (headline)', hl['auc'], hl.get('auc_ci95'), True))
    for k, lab in [('any_positive', 'any-positive'), ('ties_positive', 'ties to positive'),
                   ('ties_negative', 'ties to negative'),
                   ('raw_records_no_aggregation', 'raw records, no aggregation')]:
        r = rob.get('H3_rules', {}).get(k)
        if r:
            rows.append((lab, r['auc'], r.get('auc_ci95'), False))
    groups.append(('aggregation rule', rows))

    rows = []
    for k in ['exactly 2 studies', 'exactly 3 studies', '4+ studies']:
        r = rob.get('H2_by_n_studies', {}).get(k)
        if r:
            rows.append((k, r['auc'], r.get('auc_ci95'), k == 'exactly 2 studies'))
    groups.append(('how often the pair was retested', rows))

    rows = []
    for k, r in (rob.get('H5_slices') or {}).items():
        if r:
            rows.append((k.replace('assay::', '').replace('_', ' '), r['auc'],
                         r.get('auc_ci95'), False))
    groups.append(('assay family and HLA locus', rows))

    return [(g, rs) for g, rs in groups if rs]


def _model_rows():
    """Within-study, within-allele AUC per training scheme, bootstrapped over studies."""
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
    return rows


def fig2_ceiling_and_performance():
    """What the labels can support, and what a leakage-free model achieves.

    Both panels are AUC on one scale, so the two reference bands — the measured
    reliability of the labels and the 0.85-0.90 range reported in the literature — run
    through both and are labelled once. That shared geometry is the argument, and it is
    why these are one figure rather than two: panel (a) establishes the light band, and
    panel (b) is read against it. The underlying numbers are tabulated in 3.8; what a
    table cannot place beside them is the distance drawn at the foot of panel (b).
    """
    groups = _ceiling_groups()
    rows = _model_rows()

    gap = pt(1.0, 0.5)
    # a header line per group, plus a gap between groups; the narrow variant uses a
    # half-height gap so the figure still fits beside a column of text
    slots_a = sum(len(rs) for _, rs in groups) + len(groups) * (1 + gap)
    slots_b = len(rows) + 1.5          # the rows, plus the strip the gap arrow sits in
    pitch = pt(0.27, 0.185)
    head = pt(2.0, 1.75)               # band labels, shared axis, two panel titles

    fig, (axa, axb) = plt.subplots(
        2, 1, sharex=True, figsize=(width(), pitch * (slots_a + slots_b) + head),
        gridspec_kw={'height_ratios': [slots_a, slots_b], 'hspace': pt(0.22, 0.30)})

    # both edges of the band are the two filled markers in panel (a), read from the
    # same file so the markers land exactly on them
    rob = load('ceiling_robustness.json')
    h3 = rob.get('H3_rules', {}).get('majority') or {}
    h2 = rob.get('H2_by_n_studies', {}).get('exactly 2 studies') or {}
    ceil_lo = h3.get('auc', 0.637)
    ceil_hi = h2.get('auc', 0.681)   # conservative bound: pairs assayed by exactly two

    # the bands and the chance line carry across both panels; only the top one is labelled
    for ax in (axa, axb):
        ax.axvspan(ceil_lo, ceil_hi, color=G['light'], alpha=0.8, zorder=0)
        ax.axvspan(*CLAIMS, color=G['mid'], alpha=0.8, zorder=0)
        ax.axvline(CHANCE, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0.42, 0.95)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    # ----- panel a: how reliable the labels are
    ypos, labels = [], []
    y = slots_a
    for gname, rs in groups:
        axa.annotate(gname, xy=(0, y), xycoords=('axes fraction', 'data'),
                     xytext=(0, -1), textcoords='offset points', ha='left', va='center',
                     fontsize=pt(7.6, 6.2), style='italic', color=G['dark'], zorder=5,
                     bbox=KNOCKOUT)
        y -= 1
        for lab, v, ci, emph in rs:
            ypos.append(y)
            labels.append(lab)
            forest_row(axa, y, v, ci, emph)
            y -= 1
        y -= gap

    ytop = slots_a + 0.4
    axa.annotate(pt(f'what the labels\ncan support\n{ceil_lo:.3f}-{ceil_hi:.3f}',
                    f'labels can\nsupport\n{ceil_lo:.2f}-{ceil_hi:.2f}'),
                 xy=((ceil_lo + ceil_hi) / 2, ytop), ha='center', va='bottom',
                 fontsize=pt(7.2, 6.0), color=G['ink'])
    axa.annotate(pt('what is reported\nin the literature\n0.85-0.90',
                    'reported in\nliterature\n0.85-0.90'),
                 xy=(sum(CLAIMS) / 2, ytop), ha='center', va='bottom',
                 fontsize=pt(7.2, 6.0), color=G['ink'])
    axa.annotate('chance', xy=(CHANCE, ytop), ha='center', va='bottom',
                 fontsize=pt(7.2, 6.0), color=G['ink'], zorder=5, bbox=KNOCKOUT)
    # the two panels are far apart vertically, so panel (a) carries its own tick
    # labels rather than making the reader track down to panel (b)'s axis
    axa.tick_params(axis='x', labelbottom=True, labelsize=pt(8, 6.5))
    axa.set_yticks(ypos)
    axa.set_yticklabels(labels, fontsize=pt(8, 6.2))
    axa.set_ylim(y + 0.5, ytop + pt(2.4, 3.0))
    axa.set_title(pt('a  Reliability of the labels: one study’s outcome predicting '
                     'another’s on the same pair',
                     'a  Reliability of the labels: one study’s\n'
                     '    outcome predicting another’s'),
                  loc='left', fontweight='bold', fontsize=pt(9, 7.2))

    # ----- panel b: what a leakage-free model achieves against it
    n = len(rows)
    for i, (lab, v, lo, hi) in enumerate(rows):
        forest_row(axb, n - i, v, (lo, hi), emphasis=(i == 0))

    best = rows[0][1] if rows else 0.573
    ygap = 0.2
    axb.annotate('', xy=(CLAIMS[0], ygap), xytext=(best, ygap),
                 arrowprops=dict(arrowstyle='<->, head_width=0.16', lw=0.8,
                                 color=G['ink'], shrinkA=0, shrinkB=0))
    axb.annotate(pt(f'{CLAIMS[0] - best:.2f} AUC between the best leakage-free result '
                    'and the reported range',
                    f'{CLAIMS[0] - best:.2f} AUC unaccounted for'),
                 xy=((best + CLAIMS[0]) / 2, ygap), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=pt(7.2, 6.0), color=G['ink'])

    axb.set_yticks(range(1, n + 1))
    axb.set_yticklabels([r[0] for r in rows][::-1], fontsize=pt(8, 6.2))
    axb.set_ylim(-0.45, n + 0.9)
    axb.set_xlabel('AUC', fontsize=pt(8.5, 7.0))
    axb.tick_params(axis='x', labelsize=pt(8, 6.5))
    axb.set_title(pt('b  Leakage-free model performance: within-study, within-allele AUC',
                     'b  Leakage-free model performance:\n    within-study, within-allele AUC'),
                  loc='left', fontweight='bold', fontsize=pt(9, 7.2))

    save(fig, 'fig2_ceiling_and_performance')


if __name__ == '__main__':
    print('rendering figures ->', FIGDIR, flush=True)
    for NARROW in (False, True):
        for fn in (fig1_dataset, fig2_ceiling_and_performance):
            try:
                fn()
            except Exception as e:
                print(f'  !! {fn.__name__} failed: {type(e).__name__}: {e}', flush=True)
    print('done')
