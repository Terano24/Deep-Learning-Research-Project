"""
Does every headline number in the manuscript still match the analysis that produces it?

Two errors in this project's history were numbers that sat in the manuscript with nothing
computing them: a column of our own retrained model reported as DeepImmuno's, and seen/
unseen AUCs typed in from a figure script. This reads results/*.json, formats each number
the way the manuscript writes it, and checks that the string is present.

It does not prove the analyses are right, only that the text has not drifted from them.

Run:  python -m src.analysis.check_manuscript          (exit status 1 on any mismatch)
"""
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE_DIR / 'results'
MANUSCRIPT = BASE_DIR / 'docs' / 'MANUSCRIPT.md'


def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def f3(x):
    return f'{x:.3f}'


def ci(c):
    return f'{c[0]:.3f}–{c[1]:.3f}'


def checks():
    sr, eb = load('split_robustness.json'), load('external_baselines.json')['all']
    lb, lc = load('lung_benchmark.json'), load('label_ceiling.json')
    lk, cr = load('leakage_analysis.json'), load('ceiling_robustness.json')
    R, G, ND, T, A = sr['regimes'], sr['gaps'], sr['drop_dominant'], sr['temporal'], sr['assay_strata']
    ct = lb['contamination']
    out = []

    def add(label, needle):
        out.append((label, needle))

    # ---- 3.2 / abstract: label reliability
    # The headline AUC comes from ceiling_robustness.json, the file Figure 2 is drawn from.
    # label_ceiling.py computes the same quantity with its own random stream; the guard in
    # main() fails if the two ever drift apart by more than Monte Carlo noise.
    a, k = lc['agreement'], lc['cohens_kappa']
    u = cr['H3_rules']['majority']
    add('ceiling AUC and CI', f'{f3(u["auc"])} (95% CI {ci(u["auc_ci95"])}')
    add('ceiling table row', f'| **AUC** | **{f3(u["auc"])}** ({ci(u["auc_ci95"])}) |')
    add('agreement row', f'| Agreement | {f3(a["mean"])} ({ci(a["ci95"])}) |')
    add("kappa row", f'{f3(k["mean"])}** ({ci(k["ci95"])}) |')
    add('two-study bound', f'{f3(cr["H2_by_n_studies"]["exactly 2 studies"]["auc"])}')

    # ---- 3.4: grouping unit, IEDB column
    add('IEDB random / peptide / study',
        f'| Random | {f3(lk["random"]["overall_auc"])} |')
    add('IEDB peptide', f'| **Peptide** | **{f3(lk["peptide"]["overall_auc"])}** |')
    add('IEDB study', f'| Source (study / organism) | **{f3(lk["study"]["overall_auc"])}**')

    # ---- 3.4: robustness table, one check per row
    labels = {'random': 'Random', 'peptide': 'Exact peptide',
              'homology_d1': 'Peptides within 1 mismatch',
              'homology_d2': 'Peptides within 2 mismatches',
              'shared_6mer': 'Peptides sharing a 6-residue stretch', 'study': 'Source publication'}
    for key, lab in labels.items():
        cells = [f3(R[key]['gbm']['pooled_auc']), f3(R[key]['gbm']['within_allele']),
                 f3(R[key]['logreg']['pooled_auc']), f3(R[key]['allele_prior']['pooled_auc'])]
        if key in ('peptide', 'study'):          # these two rows are bold in the manuscript
            lab_, cells = f'**{lab}**', [f'**{c}**' for c in cells]
        else:
            lab_ = lab
        add(f'robustness row: {lab}', '| ' + ' | '.join([lab_] + cells) + ' |')
    rp, ps, p6 = G['random_minus_peptide'], G['peptide_minus_study'], G['peptide_minus_shared_6mer']
    add('random - peptide', f'{f3(rp["diff"])} (95% CI {ci(rp["ci95"])}')
    add('peptide - study', f'{f3(ps["diff"])} ({ci(ps["ci95"])})')
    add('6-mer vs peptide', f'{f3(p6["diff"])}, 95% CI {ci(p6["ci95"])}')
    nd = ND['gaps']['random_minus_study']
    add('gap without top study', f'{f3(nd["diff"])} ({ci(nd["ci95"])})')
    add('gap with top study', f'{f3(G["random_minus_study"]["diff"])} (95% CI '
        f'{ci(G["random_minus_study"]["ci95"])})')
    for fam in ('ELISPOT', 'multimer/tetramer', 'ICS'):
        r = A[fam]['regimes']
        add(f'assay family {fam}',
            f'from {f3(r["peptide"]["gbm"]["pooled_auc"])} to {f3(r["study"]["gbm"]["pooled_auc"])}')
    add('forward-in-time GBM', f'{f3(T["gbm"]["pooled_auc"])} (95% CI {ci(T["gbm"]["pooled_auc_ci95"])}')
    add('forward-in-time concentration',
        f'{T["test_top10_study_share"]:.0%} of the test pairs come from ten studies')
    cm = load('cnn_study_eval.json')['cnn_minus_attention']
    add('CNN minus attention (sign and CI)',
        f'the convolutional model scores {cm["diff"]:.3f} above the attention model '
        f'(95% CI {cm["ci95"][0]:+.3f} to {cm["ci95"][1]:+.3f})')

    # ---- 3.5: predictors on the held-out test split
    for key, lab in [('ours', 'This study'), ('gbm_baseline', 'Gradient boosting on engineered'),
                     ('deepimmuno', 'DeepImmuno [2]'), ('bigmhc_im', 'BigMHC_IM [9]'),
                     ('mixmhcpred', 'MixMHCpred 3.0')]:
        add(f'3.5 {key}', f'{f3(eb[key]["auc"])} | {f3(eb[key]["within_allele"])}')
    add('3.5 allele prior', f'{f3(eb["allele_prior_baseline"]["auc"])}**')
    add('3.5 retrained CNN', f'| {f3(eb["cnn_retrained"]["auc"])} | {f3(eb["cnn_retrained"]["within_allele"])} |')

    # ---- 3.6: contamination
    di, bm = ct['deepimmuno'], ct['bigmhc_im']
    add('BigMHC overlap', f'{bm["overlap_peptide"]:.1%} (shared peptide)')
    add('BigMHC seen/unseen', f'{f3(bm["seen"]["auc"])} | **{f3(bm["unseen"]["auc"])}**')
    ub = ct['unseen_by_all']
    add('unseen-by-all n', f'On the {ub["n"]} lung-cancer-associated pairs')
    for m, lab in [('deepimmuno', 'DeepImmuno'), ('bigmhc_im', 'BigMHC'),
                   ('mixmhcpred', 'MixMHCpred'), ('allele_prior', 'allele prior')]:
        add(f'unseen {m}', f'{lab} {f3(ub[m]["auc"])} ({ci(ub[m]["auc_ci95"])})')
    return out


def drift_guard():
    """The two ceiling scripts must agree to within Monte Carlo noise."""
    lc = load('label_ceiling.json')['reproducibility_auc_ceiling']['mean']
    cr = load('ceiling_robustness.json')['H3_rules']['majority']['auc']
    ok = abs(lc - cr) < 0.002
    print(f'  {"OK      " if ok else "MISMATCH"} ceiling scripts agree          '
          f'label_ceiling {lc:.4f} vs ceiling_robustness {cr:.4f}')
    return 0 if ok else 1


def citation_guard():
    """Every author-year citation must have a reference entry, and the reverse.

    The journal uses author-year style, so there are no numeric markers to keep in step;
    the failure mode is instead a citation whose entry was never added, or an entry left
    behind when the sentence citing it was cut. Both are invisible until a reader checks.
    """
    import unicodedata
    raw = MANUSCRIPT.read_text(encoding='utf-8')
    if '## References' not in raw:
        print('  MISMATCH citation cross-check             no reference section found')
        return 1
    body, refs = raw.split('## References')

    def norm(s):
        return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode().lower()

    rkeys = set()
    for entry in re.split(r'\n\s*\n', refs.strip()):
        entry = entry.strip()
        if not entry:
            continue
        m = re.search(r'\b(1\d{3}|20\d{2})\.', entry)
        if m:
            rkeys.add((norm(entry.split(',')[0]), m.group(1)))

    name = r"[A-Z][\wÀ-ɏ'-]+"
    cites = set()
    # parenthetical: (Author et al., 2023) and (A, 2008; B, 2009)
    for m in re.finditer(r'\(([^()]*?\b(?:19|20)\d{2}[a-z]?)\)', body):
        for part in m.group(1).split(';'):
            mm = re.match(r"^(%s)(?: and %s)?(?: et al\.)?,?\s*((?:19|20)\d{2})" % (name, name),
                          part.strip())
            if mm:
                cites.add((norm(mm.group(1)), mm.group(2)))
    # narrative: Author et al. (2023), Author and Author (2023)
    for pat in (r"\b(%s)(?: and %s)? et al\.\s*\(((?:19|20)\d{2})\)" % (name, name),
                r"\b(%s) and %s\s*\(((?:19|20)\d{2})\)" % (name, name)):
        for m in re.finditer(pat, body):
            cites.add((norm(m.group(1)), m.group(2)))

    missing = sorted(c for c in cites if c not in rkeys)
    unused = sorted(r for r in rkeys if r not in cites)
    ok = not missing and not unused
    print('  %s citation cross-check             %d cited, %d listed%s%s' % (
        'OK      ' if ok else 'MISMATCH', len(cites), len(rkeys),
        ' | uncited: %s' % unused if unused else '',
        ' | missing entry: %s' % missing if missing else ''))
    return 0 if ok else 1


def main():
    text = re.sub(r'\s+', ' ', MANUSCRIPT.read_text(encoding='utf-8'))
    bad = drift_guard()
    bad += citation_guard()
    for label, needle in checks():
        ok = re.sub(r'\s+', ' ', needle) in text
        bad += not ok
        print(f'  {"OK      " if ok else "MISMATCH"} {label:<34} {needle[:70]}')
    print(f'\n{"all" if not bad else "NOT all"} checks pass'
          + (f' ({bad} mismatched)' if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
