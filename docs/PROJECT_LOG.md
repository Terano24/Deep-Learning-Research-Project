# Project log: from a lung cancer CNN to a leakage-aware benchmark

This is the working history of the project — what we started with, what broke, what we
checked, and what survived. It exists so that any later session (human or agent) can pick
up without re-deriving the reasoning or repeating mistakes already made.

Everything here is backed by a file in `results/`. Numbers quoted are from those files.

---

## 1. Starting point (inherited, 2024)

An OPSI 2024 student research entry: *Prediction of Immunogenic Peptides for T-Cell
Immunity … in Lung Cancer Based on CNN*. The PDF is in `docs/`. It reported **94%
accuracy, 0.90 ROC-AUC**, and claimed novelty in "modifying" a CNN with 2D convolutions.

### What we found when we opened it

| Claim | Reality |
|---|---|
| A trained CNN model | `models/CNN_WEIGHT_OPSI/` is **byte-identical (MD5) to DeepImmuno's published `cnn_model_331_3_7`** |
| "Modified Conv2D architecture" | `arsitekturCNN()` matches DeepImmuno's `seperateCNN()` layer for layer |
| 10-fold cross-validation | `MatrixCNN.ipynb` runs `KFold` over a CSV of **already-computed predictions**; no model is ever retrained |
| 94% accuracy / 0.90 AUC | Not reproducible; derived from the above |

There was no training code and no training data anywhere in the repo. The "epoch" CSVs
contain model output, not training curves.

**Consequence:** the original paper cannot be submitted anywhere. Reusing DeepImmuno is
legitimate; presenting it as a model you built and trained is not. The 2024 PDF should be
treated as superseded, not revised.

---

## 2. What we rebuilt

| Component | File | Note |
|---|---|---|
| IEDB fetch | `src/data/fetch_data.py` | 92,721 T-cell assay records; 400,000 binding records |
| Dataset + splits | `src/data/dataset.py` | 31,274 unique pairs, 31.7% positive, **two** split schemes |
| Leakage-safe retrain of the original CNN | `src/models/train.py` | real per-fold retraining, held-out test, baselines |
| Pocket cross-attention model | `src/models/pocket_attention.py` | replaces Conv2D with peptide↔HLA cross-attention |
| Pretraining / fine-tuning / ensembling | `src/models/improve.py` | binding pretraining reaches **0.958** held-out AUC |
| Self-similarity + biochemical features | `src/data/features.py` | human proteome (UniProt UP000005640) |
| Study-stratified ranking objective | `src/models/rank.py` | trains on same-study, same-allele comparisons |

Reproduction commands: `docs/PIPELINE.md`.

---

## 3. The findings, in the order they appeared

### 3.1 The allele prior is most of the "signal"
A baseline predicting each allele's base rate, **with no peptide information at all**,
scores **0.752 AUC**. Measuring AUC *within* each allele removes that shortcut.
→ `results/confound_analysis.json`

### 3.2 Peptide-level splitting does not protect against anything
| Grouping unit | IEDB | TRAP (independent) |
|---|---|---|
| random | 0.840 | 0.857 |
| **peptide** | **0.826** | **0.862** |
| antigen | — | 0.817 |
| study / organism | **0.607** | **0.552** |

On an independent dataset, peptide-grouping scores *higher* than random splitting. The
precaution the field actually uses buys nothing; the grouping unit that matters is the
source. → `results/leakage_analysis.json`, `results/external_validation.json`

### 3.3 Why: study identity is nearly the label
- One **all-negative study contributes 11,486 pairs — 37% of the dataset**
- **75% of studies are label-homogeneous** (81% within the lung subset)
- Within-study ranking — the operationally relevant task — is ≈ **0.55**
→ `results/within_study_analysis.json`

### 3.4 A peptide-blind baseline beats published predictors
On the same held-out test split (4,641 pairs):

| Model | AUC | Within-allele |
|---|---|---|
| Ours | 0.832 | 0.769 |
| Gradient boosting | 0.810 | 0.721 |
| DeepImmuno | 0.792 | 0.661 |
| **Allele prior (no peptide info)** | **0.752** | 0.500 |
| BigMHC_IM | 0.647 | 0.609 |
| MixMHCpred (binding only) | 0.559 | 0.570 |

**This is the paper's strongest single line.** Locally trained models (ours, GBM) learn
the benchmark's artifacts; externally trained models (BigMHC, MixMHCpred) cannot, and
score below a peptide-blind baseline. Our margin over BigMHC is evidence of the problem,
**not** evidence that our model is better.
→ `results/external_baselines.json`

### 3.5 DeepImmuno memorisation
**37.5%** of our test peptides are in DeepImmuno's training set. It scores **0.835** on
those and **0.763** on ones it never saw — a 0.072 gap. Its published figure is 0.85.
→ `results/fair_vs_deepimmuno.json`

### 3.6 Mitigations: the simple fix wins
Five training schemes, same architecture, 205 evaluation groups:

| Scheme | Pooled | Within-allele | Within-study+allele |
|---|---|---|---|
| pointwise (baseline) | 0.623 | 0.417 | 0.532 |
| drop dominant study | 0.554 | 0.479 | 0.540 |
| **study reweighting** | **0.700** | 0.584 | 0.548 |
| mixed-label studies only | 0.648 | 0.541 | 0.543 |
| ranking (ours) | 0.649 | **0.630** | 0.557 |

→ `results/robustness.json`

### 3.7 Lung-specific failure has two causes
**Data:** 1,109 pairs, 208 positives, 67 studies but only **13 with both classes**; 59%
of records from 3 papers; **81% of lung studies label-homogeneous**.

**Biology:** in the lung subset, exact self-match is **0.43 in positives vs 0.05 in
negatives** — lung immunogenic peptides look *more* like self, the opposite of the
"dissimilarity to self" principle most predictors encode. Consistent with lung
tumour-associated antigens being self-proteins, and with published reports that
immunoediting in smoker-associated NSCLC reduces neoantigen-based predictive power.

---

## 4. Claims we had to withdraw

Recorded so they are not revived. Each was asserted before a literature check and
deflated by one.

| Withdrawn claim | Why |
|---|---|
| HLA/allele bias is our finding | **Lee et al. 2023** (TRAP, Genome Medicine) name it in their abstract and demonstrate it *on DeepImmuno specifically* |
| The cross-species/organism collapse is ours | Lee et al. ran random vs cross-species on this exact data |
| Critiquing DeepImmuno's inflated 0.85 is ours | Lee et al. already wrote that its "reported performance does not reflect real-world accuracy" |
| The ranking objective beats simple fixes (+0.146) | Measured on 10 groups / 457 pairs. Across 205 groups the gap is **+0.009** |
| Binding alone explains the signal | MixMHCpred (binding only) scores **0.559**, near chance |
| A lung-cancer-specific predictor is achievable here | Lung AUC 0.555, CI 0.438–0.670 — chance is inside the interval |

**Rule adopted:** every novelty claim gets a literature check *before* it enters a draft.

---

## 5. Bugs found and fixed

These produced plausible-looking wrong numbers and are the reason results get re-verified.

1. **Features joined by position, not key** (`feature_ablation.py`). `get_features` returns
   one row per *unique* peptide; joining positionally against duplicated peptides scrambled
   the alignment and made the features look worthless (AUC 0.499). Fixed → **+0.034**
   within-allele. Always `df.join(feats, on='peptide')`.
2. **Group-local vs global indices** (`rank.py:sample_pairs`). Group-relative positions were
   used to index the global array, producing mismatched pairs and zero validation pairs.
   Now carries an explicit `_row` column.
3. **4-mer self-similarity saturated at 1.0** — nearly every 4-mer occurs in the human
   proteome. Changed to 7-mers, which separates the classes.
4. **MemoryError in the proteome index** — two Python sets of ~11M ints (~1 GB each).
   Rewritten as sorted numpy arrays (~90 MB) with vectorised lookup. **Verified the
   refactor reproduces the cache exactly on 200 peptides** before trusting any result.
5. **IEDB binding fetch died on a read timeout** after 200k rows, saving nothing. Now
   appends each page and resumes.

---

## 6. Current position

**Paper:** the first leakage-aware benchmark of immunogenicity predictors for lung
cancer-associated MHC-I epitopes, with a curated 1,109-pair resource, four predictors
compared, and a mechanistic account of failure.

**Gap check (3 web searches):** no lung-specific benchmark of immunogenicity predictors
exists. Tumour-specific benchmarks exist for glioblastoma (TEIP 2025) and GI cancer
(NeoPrecis 2026). Lung has none.

**Scoping caveat:** our data is *lung-cancer-associated epitopes from IEDB T-cell assays*
(tumour-associated antigens plus neoantigens). Much of the clinical literature concerns
*patient-specific neoantigens from tumour sequencing*. Related, not identical — scope
claims accordingly and cite the vaccine trials as motivation only.

**Target venue:** Computational Biology and Chemistry (Elsevier, Scopus Q3). Free on the
traditional track; open access free via the corresponding author's UofT Elsevier agreement
(expires 31 Dec 2026 — confirm renewal).

**In progress:** rerunning all predictors on the full 1,109 lung pairs via study-grouped
out-of-fold (`src/analysis/lung_benchmark.py`) to become the paper's primary table.

**Not done:** BigMHC's training set (Mendeley `10.17632/dvmz6pkzvb`) was not downloaded, so
its scores were never checked for overlap with our test split the way DeepImmuno's were.
BigMHC_IM was transfer-learned on IEDB immunogenicity data and is probably contaminated too.

---

## 7. Standing cautions

- The headline AUC of any model trained on this data is **not** a measure of immunogenicity
  prediction. Always report within-allele and within-study alongside it.
- Split by `reference_id`, not only by peptide.
- A result computed on fewer than ~50 evaluation groups should be treated as provisional
  until it is reproduced on the full out-of-fold set. This is how the ranking claim failed.
- "Our model wins" on this benchmark is a warning sign, not an achievement.
