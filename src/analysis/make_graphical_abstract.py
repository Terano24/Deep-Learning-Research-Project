"""
Graphical abstract for the manuscript.

Elsevier asks for a single image that carries the article's message on its own, legible
at 5 x 13 cm. The message here is a comparison of three quantities on one AUC axis: what
the literature reports, what the labels can support, and what a leakage-free evaluation
yields. Everything else in the paper explains the distance between them.

Same conventions as make_figures: greyscale with distinct markers so it survives
black-and-white printing, serif type to match the manuscript, and every number read from
results/*.json so the image cannot drift from the analyses.

Run:  python -m src.analysis.make_graphical_abstract
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
FIGDIR = BASE_DIR / 'docs' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.edgecolor': '0.2',
    'axes.linewidth': 0.7,
    'xtick.color': '0.2',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'savefig.dpi': 300,
})

G = {'light': '0.82', 'mid': '0.58', 'dark': '0.30', 'ink': '0.12'}
CLAIMS = (0.85, 0.943)   # reported under random cross-validation (manuscript Table 1)
CHANCE = 0.5
KNOCKOUT = dict(facecolor='white', edgecolor='none', pad=1.0)


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    robust = load('ceiling_robustness.json')
    splits = load('split_robustness.json')
    ext = load('external_baselines.json')

    # Label reliability band: headline estimate -> conservative (2-study) bound. Read from
    # the same file and keys the manuscript and Figure 2 use, so the rounding agrees.
    head = robust['H3_rules']['majority']['auc']
    cons = robust['H2_by_n_studies']['exactly 2 studies']['auc']
    lo, hi = min(head, cons), max(head, cons)

    reg = splits['regimes']
    random_auc = reg['random']['gbm']['pooled_auc']
    peptide_auc = reg['peptide']['gbm']['pooled_auc']
    study_auc = reg['study']['gbm']['pooled_auc']

    allele_prior = ext['all']['allele_prior_baseline']['auc']
    deepimmuno = ext['all']['deepimmuno']['auc']

    # best leakage-free estimate: same computation that produces Figure 2b, so the two
    # cannot disagree
    from src.analysis.make_figures import _model_rows
    leakfree = max(r[1] for r in _model_rows())

    fig, ax = plt.subplots(figsize=(5.12, 1.97))   # 13 x 5 cm

    ax.axvspan(*CLAIMS, color=G['mid'], alpha=0.85, zorder=0)
    ax.axvspan(lo, hi, color=G['light'], alpha=0.9, zorder=0)
    ax.axvline(CHANCE, color=G['ink'], lw=0.8, ls=(0, (4, 3)), zorder=1)

    # three rows, top to bottom: what is reported, what the labels allow, what we measure
    rows = [
        (2, [(random_auc, 'random split', 'o'),
             (peptide_auc, 'peptide split', 's')],
         'benchmark as usually built'),
        (1, [(allele_prior, 'allele prior\n(no peptide)', '^'),
             (deepimmuno, 'published\npredictor', 'D')],
         'what that performance rests on'),
        (0, [(study_auc, 'study split', 'v'),
             (leakfree, 'leakage-free', '*')],
         'benchmark split by publication'),
    ]
    # Within a row the two points can sit 0.014 apart, so their labels alternate above and
    # below the marker rather than overlapping each other.
    for y, pts, caption in rows:
        for i, (x, lab, mk) in enumerate(pts):
            ax.scatter([x], [y], s=46 if mk == '*' else 26, marker=mk,
                       facecolor=G['ink'] if mk in ('*', 'o') else 'white',
                       edgecolor=G['ink'], lw=0.9, zorder=4)
            above = i == 1
            ax.annotate(f'{lab} {x:.2f}', (x, y), textcoords='offset points',
                        xytext=(0, 8 if above else -9),
                        ha='center', va='bottom' if above else 'top', fontsize=5.4,
                        color=G['ink'], zorder=5, bbox=KNOCKOUT, linespacing=0.95)
        ax.annotate(caption, (0.335, y), ha='left', va='center', fontsize=5.2,
                    style='italic', color=G['dark'], zorder=5)

    ax.annotate('reported\nrandom CV', (sum(CLAIMS) / 2, 2.72), ha='center', va='center',
                fontsize=5.6, color='white', fontweight='bold', linespacing=1.0, zorder=5)
    ax.annotate(f'labels support\n{lo:.3f}-{hi:.3f}', ((lo + hi) / 2, 2.72), ha='center',
                va='center', fontsize=5.6, color=G['ink'], linespacing=1.0, zorder=5)
    ax.annotate('chance', (CHANCE, 2.72), ha='center', va='center', fontsize=5.4,
                color=G['ink'], zorder=5, bbox=KNOCKOUT)

    ax.set_xlim(0.325, 0.95)   # left gutter holds the row captions, clear of the data
    ax.set_ylim(-0.75, 3.05)
    ax.set_yticks([])
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(axis='x', labelsize=6, length=2.5, pad=1.5)
    ax.set_xlabel('ROC-AUC', fontsize=6.5, labelpad=1.5)

    for ext_ in ('pdf', 'png', 'tif'):
        out = FIGDIR / f'graphical_abstract.{ext_}'
        fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
        print('wrote', out)
    plt.close(fig)

    print(f'\nvalues used: random {random_auc:.3f} peptide {peptide_auc:.3f} '
          f'study {study_auc:.3f} allele-prior {allele_prior:.3f} '
          f'deepimmuno {deepimmuno:.3f} leakage-free {leakfree:.3f} '
          f'band {lo:.3f}-{hi:.3f}')


if __name__ == '__main__':
    main()
