# Label Reliability and Study-Level Leakage Bound Measurable Performance in T-Cell Epitope Immunogenicity Prediction

Code, data and analyses for the manuscript of the same name
(`docs/MANUSCRIPT.pdf`). I Nyoman Agung Ananda Terano, Mirrabell Frederica Hadiwijono.

Published predictors of CD8⁺ T-cell epitope immunogenicity report ROC-AUC of 0.85–0.90 on
IEDB-derived data. This repository measures how much of that is transferable biology and
how much is the structure of the benchmark itself.

## What it finds

**The grouping unit, not the model, sets the reported number.** One model class, one
dataset, only the split changes:

| split by | IEDB | TRAP (independent) |
|---|---|---|
| random rows | 0.840 | 0.857 |
| peptide | 0.826 | 0.862 |
| peptides sharing a 6-residue stretch | 0.818 | n/a |
| source publication / organism | **0.607** | **0.552** |

Peptide-level splitting, the field's standard precaution, lowers AUC by only
0.014 against random splitting on IEDB (95% CI 0.009–0.024) and scores
*above* it on the independent dataset. Grouping peptides by sequence similarity changes
AUC by under 0.02, so similarity is not the cause. Holding out the source publication is
what removes the effect: one study contributes 37% of all pairs with no positive outcome
among them, and 75% of studies report a single outcome class. That one study accounts for
about 60% of the gap; without it the gap is 0.09 (95% CI 0.066–0.118).
The ordering also holds with logistic regression, within single assay families, and in a
forward-in-time test (`results/split_robustness.json`).

**A baseline that cannot see the peptide beats every published predictor tested.**
Predicting each allele's training-set positive rate, using no sequence information at
all, reaches 0.752 against BigMHC_IM 0.647, DeepImmuno 0.606, MixMHCpred 0.559.

**The labels support less than is claimed of them.** Two laboratories assaying the same
peptide–HLA pair agree at κ = 0.273, an AUC-equivalent of 0.636 (0.617–0.656).

**Under leakage-free evaluation** — within study and within allele, across 205 groups —
the best model here reaches 0.573 (0.544–0.603), 84% of the conservative reliability
bound. The original convolutional architecture does not differ significantly from
it.

## Verify the central claim in one command

```bash
python -m src.models.deepimmuno
# DeepImmuno (published checkpoint) on 4641 held-out pairs: AUC 0.606
```

### The two CNN checkpoints are not interchangeable

| path | what it is | test AUC |
|---|---|---|
| `models/CNN_WEIGHT_OPSI` | DeepImmuno's released `cnn_model_331_3_7`, byte for byte | 0.606 |
| `models/CNN_RETRAINED` | the same architecture, trained here on our split | 0.792 |

Only the first may be called DeepImmuno, and it is loaded only through
`src/models/deepimmuno.py`. The second is written to a column named `cnn_retrained`.
The 0.19 between them is not an architectural difference — it is what exposure to this
dataset's study structure is worth, and it is one of the paper's results.

## Reproducing

`docs/PIPELINE.md` lists every command that produces every file in `results/`, in order.
Python 3.11 or 3.12; `pip install -r requirements.txt`.

The comparisons against published predictors need no third-party installs: BigMHC and
MixMHCpred outputs on our pairs, both of their training sets, and the TRAP dataset are
committed under `data/external/`, with provenance in that directory's README.

Two inputs are not in the repository because of size: the IEDB export
(`python -m src.data.fetch_data`) and the human proteome used for self-similarity
features (see `docs/PIPELINE.md`).

## If you are evaluating an immunogenicity predictor

The recommendations in §4.1 of the manuscript, as a checklist:

1. Split by source publication, not only by peptide, and state the grouping unit.
2. Report within-study and within-allele AUC alongside any pooled figure.
3. Include a peptide-blind allele-prior baseline. If it is competitive, the benchmark is
   measuring dataset structure.
4. State the negative-set definition. Untested candidates and assayed negatives are
   different tasks, worth ≈0.4 AUC.
5. Audit training-set overlap before reporting a comparison.
6. Bootstrap over studies, not rows.

`src/analysis/confounds.py` and `src/analysis/within_study.py` implement 2 and 6;
`src/data/dataset.py` implements 1.

## Scope

9–10mers and HLA-A/B/C only, inherited from the DeepImmuno encoding. The reliability
bound is measured on IEDB alone. No improved predictor is proposed: the model here is a
leakage-free reference point, and its apparent advantage over published tools on the
peptide-grouped split is an artifact of the split this work argues against. §5 of the
manuscript states the limitations in full.

## Provenance

This began as an entry to Olimpiade Penelitian Siswa Indonesia (OPSI) 2024 by the same
authors, reimplementing DeepImmuno-CNN for lung-cancer neoantigens. The benchmark audit
grew out of attempting to validate that model properly. The original CNN inference entry
point is still `main.py`.

## Citation

See `CITATION.cff`. Please cite the archived release rather than the repository
directly.
