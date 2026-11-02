# DOME checklist

Supplementary reporting checklist for *Label Reliability and Study-Level Leakage Bound
Measurable Performance in T-Cell Epitope Immunogenicity Prediction*.

Structured according to the DOME recommendations for supervised machine learning validation
in biology (Walsh et al., 2021; Nat. Methods 18, 1122–1127;
[10.1038/s41592-021-01205-4](https://doi.org/10.1038/s41592-021-01205-4)), using the item
names given in that paper under **D**ata, **O**ptimization, **M**odel and **E**valuation.

Section references point to the manuscript. DOME version 1.0.

---

## Data

| Item | Response |
|---|---|
| **Provenance** | T-cell assay records from the Immune Epitope Database (Vita et al., 2019) Query API, restricted to MHC class I, human host, linear peptides of length 9–10 (92,721 records). Filtering to a four-digit HLA-A/B/C allele present in our pseudo-sequence table and a canonical twenty-letter peptide left 65,732 records over 32,167 unique peptide–HLA pairs. Collapsed to one label per pair by majority vote across records; 893 exact ties dropped. Final benchmark: **31,274 pairs, 31.7% positive, from 1,365 source publications**. A further 400,000 MHC binding and ligand-elution records were retrieved under the same constraints for encoder pretraining. Data not previously used in this grouping. §3.1 |
| **Dataset splits** | Two schemes stored in the same table. (i) **Peptide-grouped**: `split`/`fold` grouped by peptide. (ii) **Study-grouped**: `study_split`/`study_fold` grouped by `reference_id` (source publication). Each uses a 15% grouped hold-out and 10 grouped cross-validation folds. Random (ungrouped) splitting is reported alongside both as the comparator. Robustness checks add three similarity-based grouping rules with five grouped folds each, plus a forward-in-time split where studies are dated by smallest PubMed ID and training uses studies up to the 80th percentile of dated pairs. §3.2, §4.4 |
| **Redundancy between data splits** | Addressed directly, and is a principal result rather than a precaution. Three similarity rules were applied: single-linkage merging of equal-length peptides within one or two mismatches (candidates from a pigeonhole index, then checked exactly), and merging of peptides sharing any exact six-residue substring. A five-residue rule was rejected because it joined 14.5% of pairs into one cluster. Peptides of different lengths are never merged. **Sequence-similarity grouping changes AUC by under 0.02**, whereas grouping by source publication reduces AUC from 0.840 to 0.607 — i.e. the operative redundancy is publication structure, not sequence similarity. §3.2, §4.4 |
| **Availability of data** | Yes. Curated benchmark, both split schemes, all model predictions, and the outputs of every audited third-party predictor are released. Repository: <https://github.com/Terano24/Deep-Learning-Research-Project>. Archived at Zenodo: concept DOI [10.5281/zenodo.23719454](https://doi.org/10.5281/zenodo.23719454) (all versions); version DOIs [10.5281/zenodo.23719455](https://doi.org/10.5281/zenodo.23719455) (v1.0.0) and [10.5281/zenodo.23840985](https://doi.org/10.5281/zenodo.23840985) (v1.0.1). Code under MIT licence; redistributed third-party data retains its own terms, listed in `LICENSE`. |

## Optimization

| Item | Response |
|---|---|
| **Algorithm** | Three model families are evaluated. (i) A bidirectional cross-attention neural network (this study). (ii) Gradient boosting on an AAindex/PCA encoding. (iii) Logistic regression, used to test whether findings are specific to the model class. Published predictors (DeepImmuno, BigMHC_IM, MixMHCpred 3.0) are scored from released checkpoints, not retrained, except where explicitly labelled as "DeepImmuno architecture, retrained here". §3.4 |
| **Meta-predictions** | No. No predictor's output is used as an input feature to another. Published predictors are evaluated as independent comparators on identical pairs. |
| **Data encoding** | Peptides: learned residue and position embeddings over 10 positions. HLA: 46-residue pseudo-sequence with the same embedding scheme. Gradient-boosting baseline uses an AAindex/PCA encoding. The allele-identity baseline uses no peptide information at all. §3.4 |
| **Parameters** | Cross-attention network: **65,987 parameters**, against 31,274 labelled pairs. Architecture: self-attention within peptide and within HLA pseudo-sequence, cross-attention in both directions, dense trunk, sigmoid output. §3.4 |
| **Features** | No feature selection was performed. Inputs are the peptide sequence and the HLA pseudo-sequence only; no expression, tumour-sequencing or patient-context features are used, by design (§2 excludes patient-specific neoantigen prioritisation pipelines, which use such features). |
| **Fitting** | The encoder is pretrained on the 400,000-record binding/elution corpus (held-out binding AUC 0.958) and fine-tuned on immunogenicity. Over- and under-fitting are addressed principally through the grouped split schemes rather than through capacity control: the leakage-free estimate (0.573) is stable across architectures, and the convolutional and attention models do not differ significantly (+0.013, 95% CI −0.017 to +0.043). §3.4, §4.8 |
| **Regularization** | Dropout within the network. Class weighting applied identically across compared models. Additional training schemes tested: dropping the dominant study, study-size reweighting, training on mixed-outcome studies only, and a study-stratified pairwise ranking objective. §3.4 |
| **Availability of configuration** | Yes. Architectures, training schemes, folds, class weighting and seeds are in `src/models`; the full pipeline is documented in `docs/PIPELINE.md` and reproduced end-to-end by `notebooks/Reproduce_Manuscript.ipynb`. |

## Model

| Item | Response |
|---|---|
| **Interpretability** | Black box for the neural models; no attempt is made to interpret attention weights as biology. Interpretability is instead addressed by comparison against a fully transparent baseline: an **allele-identity prior** that predicts each allele's training-set positive rate using no peptide information. That baseline reaches 0.752 and outscores every published predictor tested on the same pairs, which bounds how much of the apparent signal requires any sequence model at all. §3.4, §4.5 |
| **Output** | Regression output in [0,1] (sigmoid), interpreted as a ranking score for immunogenicity, evaluated by ROC-AUC. The operational task is ranking candidates *within one experiment* (§2). |
| **Execution time** | Not a contribution of this work and not optimised; the released pipeline trains and evaluates all reported models on a single consumer GPU. `python -m src.models.deepimmuno` scores the released DeepImmuno checkpoint and prints its held-out AUC in one command. |
| **Availability of software** | Yes. Full source at <https://github.com/Terano24/Deep-Learning-Research-Project>, archived at Zenodo (DOIs above), MIT licence. PRIME 2.0 was obtained but its bundled Windows binary did not execute; it is excluded and this is documented rather than silently omitted. §3.4 |

## Evaluation

| Item | Response |
|---|---|
| **Evaluation method** | Grouped cross-validation under three grouping units (random, peptide, source publication), plus a held-out grouped test set, plus a forward-in-time split. Models are scored from study-grouped out-of-fold predictions. Pooled AUC is reported alongside two stricter quantities: within-allele AUC and within-study (and within-study-and-allele) AUC. Groups require ≥10 pairs and both outcome classes. §3.2, §3.5 |
| **Performance measures** | ROC-AUC throughout, for comparability with the published figures compiled in Table 1. ROC-AUC and PR-AUC can rank methods differently under class imbalance (Saito and Rehmsmeier, 2015); **PR-AUC for every predictor is included in the released result files** rather than tabulated. Agreement between laboratories is additionally reported as raw agreement and Cohen's κ. §3.5 |
| **Comparison** | Against three published immunogenicity predictors scored on identical pairs (DeepImmuno, BigMHC_IM, MixMHCpred 3.0 as a binding-only control), and against two deliberately simple baselines (allele-identity prior; gradient boosting). Published literature figures are compiled with their evaluation protocols in Table 1, with every source sentence recorded verbatim in `data/external/reported_performance.csv`. A contamination audit reports the proportion of evaluation pairs present in each predictor's released training set. §3.4, §4.5, §4.6 |
| **Confidence** | All headline quantities carry 95% confidence intervals. Intervals for within-study quantities, for differences between grouping rules, and for the forward-in-time test **bootstrap over whole studies rather than rows** (1,000 replicates), because pairs within a study are not independent. The reliability estimate uses a separate bootstrap resampling pairs with replacement (2,000 replicates), drawing a fresh study pairing each time; the point estimate is the mean over 200 random pairings so it does not depend on one draw. Differences are computed from unrounded values. No p-values are reported; claims rest on interval estimation. §3.3, §3.5 |
| **Availability of evaluation** | Yes. All model predictions, out-of-fold assignments, both split schemes, the outputs of every audited third-party predictor, and the result JSON files behind every reported number are released with the repository and archived at Zenodo. Every number in the manuscript is verified against those files by `python -m src.analysis.check_manuscript`, which also cross-checks citations against the reference list and recomputes the chance-corrected performance fraction. |

---

## Note on a reporting choice

The chance-corrected fraction in §4.8 and §5 is computed as (AUC − 0.5) / (reference − 0.5),
not as a raw ratio of AUCs. A raw ratio treats zero rather than chance as the floor, under
which a model with no predictive signal (AUC 0.5) would appear to reach 73% of a 0.681
reference. The reported figure of 40% is the fraction of the signal available *above chance*.

The reliability figure (AUC 0.636, or 0.681 restricted to pairs assayed by exactly two
studies) is used as an **empirical cross-publication reproducibility reference**, not as a
theoretical cap on achievable prediction. Inter-annotator agreement does not bound what a
well-specified model can achieve when annotators carry independent noise (Richie et al.,
2022); §4.2 and §5 state this explicitly.
