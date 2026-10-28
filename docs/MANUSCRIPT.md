# Leakage, Label Noise, and Overestimated Performance in T-Cell Epitope Immunogenicity Prediction

**I Nyoman Agung Ananda Terano**¹  **Mirrabell Frederica Hadiwijono**²

## Abstract

Computational predictors of CD8⁺ T-cell epitope immunogenicity report ROC-AUC values of
0.85–0.90 on data derived from the Immune Epitope Database (IEDB), and are used to
prioritise candidates for neoantigen vaccines. We show that these values are inflated by
leakage from how the benchmarks are built, and that the labels could not support them in
any case. From 65,732 human MHC-I T-cell assay records we identified 2,914 peptide–HLA
pairs assayed independently by two or more publications, and measured how well one
laboratory's outcome predicts another's on the same pair. Agreement is 72.2% and Cohen's
κ is 0.273, an AUC-equivalent of 0.636 (95% CI 0.617–0.656), or 0.681 on the least-selected
subset of pairs assayed by exactly two studies. A repeat experiment observes the
biology directly, so reported performance exceeds what these labels can support. Three
benchmark-construction choices produce the excess. First, splitting by peptide, the field's
standard precaution, performs indistinguishably from not splitting at all: 0.826 against
0.840 for random splits on IEDB, and 0.862 against 0.857 on the independent TRAP dataset.
Grouping by source publication instead gives 0.607. That gap of 0.219 is the leakage. It survives
peptide-level holdout because a held-out peptide's study-mates stay in training, and study
identity nearly determines the label: one all-negative study supplies 37% of pairs, 75% of
studies report a single outcome class, and an allele-identity baseline that never sees the
peptide reaches 0.752, above BigMHC_IM (0.647), DeepImmuno (0.606) and MixMHCpred 3.0
(0.559). Second, DeepImmuno and BigMHC_IM had themselves been trained
on 37.5% and 27.1% of our evaluation pairs, and fall by 0.23 and 0.26 AUC on pairs they had
not seen, where neither clears chance. Third, counting untested candidates as negatives
rather than assayed negatives raises per-patient AUC from ≈0.55 to ≈0.97 for the same
nominal task. Under leakage-free evaluation, within study and within allele across 205
groups, the best model we construct reaches 0.573 (95% CI 0.544–0.603), 84% of the
conservative bound; the original convolutional architecture reaches 0.544, a difference
that is not significant. The bound is measured on IEDB alone and limits what is measurable
against these labels, not what is biologically achievable. We release the benchmark, both
split schemes, all model predictions, and a protocol for reporting.

**Keywords:** T-cell epitope · immunogenicity prediction · data leakage · benchmark design
· label reliability · reproducibility · IEDB


^¹Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia. ²University of Toronto, Toronto, Canada. Correspondence to: I Nyoman Agung Ananda Terano (teranoace@gmail.com).

## 1. Overview

Predictors of which peptide–MHC complexes trigger a CD8⁺ T-cell response are used to
design neoantigen vaccines and to choose targets for adoptive cell therapy. Experimental
screening is slow and costly, so predictors trained on IEDB have multiplied, and recent ones
report ROC-AUC values between 0.85 and 0.90. Two established facts argue against taking
those numbers at face value.

First, peptide and allele do not determine the outcome. A response also depends on the
individual's T-cell receptor repertoire, prior antigen exposure, the rest of their HLA
genotype and the tissue context. Two people with the same restricting allele can respond
differently to the same peptide, so a model given only a peptide and an allele has a ceiling
below 1.0 whatever its architecture.

Second, laboratories measure the response inconsistently. International proficiency panels
run by the Cancer Vaccine Consortium and the CIMT Immunoguiding Programme found
inter-laboratory variation of roughly 35-fold for ELISpot, 20-fold for intracellular
cytokine staining and 100-fold for tetramer assays, and some laboratories could not detect
all predefined responses "even on a yes/no basis" [1].

Harmonisation guidelines reduce this variation in prospective, coordinated studies [1], but
they cannot be applied retroactively to a database that aggregates decades of heterogeneous
protocols. To our knowledge, what that heterogeneity does to machine-learning benchmarks
built on IEDB has not been measured. Predictors deal with label disagreement by convention.
PredIG, for example, discards peptide–HLA pairs whose outcomes disagree across experiments
[8].

Related problems have been reported. Lee et al. documented HLA bias in immunogenicity
predictors and showed that cross-species splits lower performance [3]. Benchmarks of
TCR–epitope predictors find near-chance performance on unseen epitopes and warn that
external negatives "introduce batch-like confounders, causing models to learn
dataset-specific artifacts" [9]. Kapoor and Narayanan reviewed 17 scientific fields and found
leakage affecting 294 papers; in one case study, complex models claimed to beat logistic
regression stopped doing so once leakage was corrected [6].

We are not aware of a published measurement of what the labels themselves can support, or
of which construction choices open the gap between that limit and reported performance.
This paper does both. Our
contributions are:

1. A measured reliability bound for IEDB immunogenicity labels, with sensitivity analyses
   over aggregation rule, retesting frequency, assay family and HLA locus.
2. A demonstration that peptide-level splitting is ineffective, replicated on an independent
   dataset, and identification of source publication as the grouping unit that determines
   the result.
3. A contamination audit of two widely used published predictors against our evaluation set.
4. A leakage-free performance estimate, an architecture comparison showing the choice makes
   no measurable difference, and a worked example of a feature that standard validation
   endorses and grouped validation rejects.

## 2. Scope and definitions

**Scope.** We examine benchmarks for CD8⁺ T-cell epitope immunogenicity built from IEDB
assay records, where a model receives a peptide and an MHC-I allele and predicts whether a
T-cell response was observed. We do not examine MHC binding or ligand presentation
prediction, which are better-posed problems with more reproducible labels and where
reported performance is correspondingly higher [9]. We also leave out patient-specific
neoantigen prioritisation pipelines, which use tumour sequencing, expression and clonality
in addition to sequence [4]; §3.7 shows why conflating the two inflates apparent
performance.

**Grouping unit.** A split is defined by the unit held out. Splitting *by peptide* means no
peptide appears in both training and test data; splitting *by study* means no record from a
given publication appears in both. Leakage here is shared structure: records from one
publication share a cohort, a protocol and often an outcome, so a model can identify the
source rather than the biology. We use "grouping unit" throughout to name the choice,
because it is the choice that determines the reported number (§3.4).

**Label reliability.** We define the reliability of a label set as the agreement between two
independent measurements of the same quantity, here two publications assaying the same
peptide–HLA pair. Expressed as an AUC, this is the score a repeat experiment achieves when
used to predict another experiment's outcome. Because a repeat experiment observes the
biology rather than inferring it from sequence, it is the most informative predictor
obtainable for that pair.

**Limits of the bound.** The reliability figure limits what can be *measured* against a
given label set. It does not limit how accurate immunogenicity prediction could be in
principle: when annotators carry independent noise, a well-specified model can exceed their
mutual agreement [7]. We return to this in §4.

**Operational task.** We take the task that matters for vaccine design to be ranking
candidates *within one experiment*: given the peptides a laboratory is considering for one
patient or one antigen, which should be screened first. This motivates reporting
within-study and within-allele AUC alongside pooled figures (§7.5).

## 3. Results

### 3.1 One study supplies a third of the data, and most studies report a single outcome

After retrieval and cleaning (§7.1), the benchmark comprises 31,274 unique peptide–HLA
pairs, 31.7% positive, drawn from 1,365 publications. Its composition is uneven
(**Figure 1**). A single study contributes 11,486 pairs, 37% of the dataset, and every one
of its outcomes is negative; the ten largest studies supply 54% of all pairs and the
hundred largest 81%. Across all studies, 75% report only one outcome class, and those
studies account for 57% of the pairs. Study identity is therefore highly informative about
the label before any peptide is examined, which motivates the analyses that follow.

### 3.2 Immunogenicity labels reproduce poorly between laboratories

Restricting to the 2,914 pairs assayed by two or more distinct publications, and taking
each study's majority outcome, we asked how well one study's result predicts another's on
the same pair (§7.3).

| Quantity | Value (95% CI) |
|---|---|
| Agreement | 0.722 (0.706–0.738) |
| Cohen's κ | **0.273** (0.235–0.312) |
| **AUC** | **0.636** (0.617–0.656) |

A κ of 0.273 denotes fair agreement only. The agreement between two experiments on the same
pair bounds what can be *measured* against these labels.

### 3.3 The reliability estimate is robust to how it is measured

We tested three threats to this estimate (**Figure 2**, panel a).

**Aggregation rule.** Majority vote gives 0.636; any-positive 0.652; ties-to-positive 0.646;
ties-to-negative 0.556; and raw records with no aggregation at all 0.595. The range is
0.556–0.652, and the least-processed variant falls below the headline figure.

**Selection bias.** Pairs are not retested at random. Reproducibility declines with
retesting frequency: 0.681 for pairs assayed by exactly two studies (n = 1,806), 0.530 for
exactly three, 0.543 for four or more. We therefore adopt 0.681 as a conservative bound.
Separately, repeat-tested pairs are 72.5% positive against 26.5% for singly-tested pairs, so
the decision to retest is driven by the previous result.

**Slice dependence.** By assay family, ELISpot 0.702, ICS 0.655, multimer/tetramer 0.605;
by locus, HLA-A 0.594, HLA-B 0.775. Every slice falls far below reported performance. The
ordering across assay families independently reproduces the proficiency-panel finding that
tetramer assays vary more than ELISpot [1].

### 3.4 Peptide-level splitting does not prevent leakage

Holding out data by peptide is the standard precaution in this field. We compared grouping
units using one model class throughout, so that only the split changes.

| Grouping unit | IEDB (this study) | TRAP (independent) [3] |
|---|---|---|
| Random | 0.840 | 0.857 |
| **Peptide** | **0.826** | **0.862** |
| Antigen | — | 0.817 |
| Source (study / organism) | **0.607** | **0.552** |

On the independent dataset, peptide grouping scores *above* random splitting. In both
datasets, widening the grouping unit to the data source collapses performance. Peptide
grouping leaves the structure that inflates these results untouched.

### 3.5 A baseline that cannot see the peptide outscores published predictors

On the held-out test split (4,641 pairs, 1,470 positive), an allele-prior baseline reaches
0.752. It predicts each allele's training-set positive rate and uses no peptide information
whatsoever. On the same pairs BigMHC_IM reaches 0.647, DeepImmuno 0.606 and MixMHCpred
0.559: every published predictor scores below the baseline that never sees the peptide.

| Predictor | Pooled AUC | Within-allele AUC |
|---|---|---|
| This study | 0.832 | 0.769 |
| Gradient boosting on engineered features | 0.810 | 0.721 |
| DeepImmuno architecture, retrained here | 0.792 | 0.661 |
| **Allele prior (no peptide information)** | **0.752** | 0.500 |
| BigMHC_IM [9] | 0.647 | 0.609 |
| DeepImmuno [2] | 0.606 | 0.633 |
| MixMHCpred 3.0 (binding only) [11] | 0.559 | 0.570 |

The two DeepImmuno rows are different models and the gap between them is itself a result:
the published checkpoint scores 0.606, while the same architecture retrained on our
training split reaches 0.792. The architecture is not what separates them; exposure to
this dataset's study structure is.

The allele prior is constant within an allele, so its within-allele AUC is 0.500 by
construction: the entire 0.752 is allele frequency. Holding the allele fixed costs the
models trained here between 0.06 and 0.13 AUC, which is the allele signal they had been
using. It costs the externally trained predictors nothing, and DeepImmuno and MixMHCpred
score marginally *higher* within allele than pooled, because they never learned this
dataset's allele skew in the first place. What looks like an advantage for the local
models over published ones is, to a large extent, that signal.

A peptide-blind baseline that beats published predictors is diagnostic: on this benchmark,
allele identity and dataset structure carry more signal than the peptide sequence does.

### 3.6 Published predictors were trained on the evaluation data

Intersecting our evaluation pairs with released training sets (§7.5) shows substantial
overlap, and performance falls sharply outside it.

| Predictor | Evaluation pairs it had seen | Seen | Unseen |
|---|---|---|---|
| DeepImmuno [2] | 37.5% (shared peptide) | 0.746 | **0.512** |
| BigMHC_IM [9] | 27.1% (shared peptide) | 0.807 | **0.549** |

On the 778 lung-cancer-associated pairs whose sequence appears in neither training set,
every published predictor's interval includes chance: DeepImmuno 0.528 (0.476–0.583),
BigMHC 0.502 (0.438–0.561), MixMHCpred 0.446 (0.383–0.502), allele prior 0.465
(0.415–0.514).

### 3.7 The definition of a negative changes the task by ≈0.4 AUC

Using the harmonised multi-cancer resource of Müller et al. [4], across 99 patients:

| What counts as a negative | Peptides compared | Per-patient AUC |
|---|---|---|
| Every candidate neo-peptide, screened or not | 1,780,481 (76% never assayed) | ≈ 0.97 |
| **Assayed negatives only** | **421,017 assayed (23.6% of candidates)** | **≈ 0.55** |

Of the 421,017 screened peptides, 178 were immunogenic. Ranking those 178 against *all*
candidates gives per-patient AUC ≈ 0.97; discriminating them from assayed negatives gives
≈ 0.55. Both are published as immunogenicity prediction. Most "negatives" in the former are
peptides nobody screened, overwhelmingly non-binders that any filter rejects.

### 3.8 Leakage-free performance approaches the measured ceiling, and architecture is immaterial

We scored every model from study-grouped out-of-fold predictions, within study and allele,
across 205 groups, bootstrapping over studies (**Figure 2**, panel b):

| Training scheme | AUC | 95% CI |
|---|---|---|
| **Ranking objective, no engineered features** | **0.573** | 0.544–0.603 |
| Ranking objective, with features | 0.557 | 0.526–0.588 |
| **CNN (original Conv2D architecture)** | **0.544** | 0.516–0.572 |
| Study reweighting | 0.548 | 0.519–0.576 |
| Pointwise, no features | 0.538 | 0.511–0.566 |
| Pointwise baseline | 0.532 | 0.504–0.560 |

All intervals clear chance, and the best result is 84% of the conservative reliability bound
(0.681). Given identical pretraining, folds and seeds, the convolutional model differs from
the attention model by +0.013 (95% CI −0.017 to +0.043), which is not significant.
Architecture made no measurable difference. Only one comparison reaches significance: the
ranking objective beats pointwise training by +0.035 (CI +0.001 to +0.069). The lower bound
sits almost on zero, so we call that result marginal.

Study-aware reweighting recovers much of the loss on pooled metrics (0.623 → 0.700) and costs
less than a bespoke objective.

### 3.9 A worked example: a feature that standard validation endorses and grouped validation rejects

Self-similarity to the human proteome looks strongly predictive under standard validation.
Pooled across lung-cancer-associated pairs it reaches AUC 0.671 (CI 0.630–0.712), and under
peptide-grouped validation it raised within-allele AUC by +0.034.

Grouping by study removes the effect. Within study its AUC is 0.483, and no study scores
above chance. Withholding the whole feature block changes within-study AUC by +0.005
(CI −0.015 to +0.024) and within-study-and-allele AUC by −0.007; neither is significant. The
feature tells studies apart, and says nothing about which peptides within a study are
immunogenic.

Of nine feature families tested within study, only two clear chance: mean hydrophobicity
0.550 (0.518–0.582) and aromatic content 0.542 (0.517–0.568), both consistent with known
determinants of TCR recognition [8].


## 4. Discussion

Reported AUCs of 0.85–0.90 are higher than the measured reliability of the labels (0.636, or
0.681 under the conservative bound) and higher than any model we could build under
leakage-free evaluation (0.573). We attribute the gap to three things: peptide-level
splitting, which behaves like no splitting; training-set overlap, which we found in both
audited predictors; and negatives defined as untested candidates, which changes the task.

AUC is measured against noisy labels. A model that learns only transferable biology should
therefore score near the label reliability, and one that scores well above it is probably
also fitting the noise. Our results place that noise at the level of the study.

**Agreement as a bound.** Agreement between annotators does not cap model performance in
general. If annotators err independently, a well-specified model can beat their mutual
agreement [7]. Our figures therefore describe performance against these labels, not how
accurate prediction could be in principle. Random label noise would make a high measured
AUC unattainable. High AUCs are attainable here, which indicates that the noise is
systematic and learnable, as §3.4–3.6 show.

**Relation to prior work.** Kapoor and Narayanan documented this pattern across 17 fields
[6], and ours is an instance in computational immunology. In their case study, complex
models claimed to beat logistic regression and ceased to do so once leakage was corrected;
in ours, published predictors are outscored by a baseline that cannot see the peptide. Our
setting also lets us measure the reliability of the labels directly instead of assuming it,
because IEDB records which publication produced each measurement and some pairs were
measured more than once. We extend Lee et al. [3], who identified HLA bias and cross-species
degradation, by identifying the source publication as the operative grouping unit and by
quantifying the label ceiling.

**What bounds performance here.** At 0.573 against a conservative bound of 0.681, we are at
84% of what the labels permit, leaving ≈0.11 AUC of headroom. We do not think architecture
can recover it: the convolutional and attention models are statistically indistinguishable,
and methods that do substantially better use information that sequence-only benchmarks
lack, such as tumour expression, clonality, or the patient's own tumour-infiltrating
repertoire [4]. Approaches that add TCR sequence without patient context report near-chance
performance on unseen epitopes [9], so the limit is not specific to sequence-only models.

Raising the ceiling requires harmonised assays of the kind proficiency panels have shown to
reduce variance [1]. Broadening coverage requires more distinct epitopes, currently limited
by peptide-library production, and patient-specific prediction requires patient-specific
measurement.

### 4.1 Recommendations

1. Split by source publication, not only by peptide, and state the grouping unit used.
2. Report within-study and within-allele AUC alongside any pooled figure.
3. Include a peptide-blind allele-prior baseline. If it is competitive, the benchmark is
   measuring dataset structure.
4. State the negative-set definition explicitly; untested candidates and assayed negatives
   are different tasks.
5. Audit training-set overlap before reporting comparative performance.
6. Bootstrap over studies, not rows; peptides within a study are not independent.


## 5. Limitations

- Label reliability is measured on IEDB alone. The independent dataset available to us
  lacks publication identifiers, so §3.4 but not §3.2 could be replicated externally.
- Repeat-tested pairs are a selected subset. We address this by stratification and adopt the
  most favourable bound, but the estimate remains conditional on that subset.
- MixMHCpred's training set is not public and could not be audited for contamination.
- Our protocol is stricter than the usual one, but we do not show that it predicts clinical
  success better. That would require prospective experimental validation.
- Analysis is restricted to 9–10mers and HLA-A/B/C, inherited from the encoding of the
  predictor we reimplemented.
- We propose no improved predictor. Our best model is a leakage-free reference point, and
  its apparent advantage over published tools on the peptide-grouped split (0.832 against
  0.606 for DeepImmuno's released checkpoint) is an artifact of the split we argue against,
  and of our model having been fitted to this dataset's study structure.


## 6. Conclusion

The labels underlying T-cell epitope immunogenicity benchmarks reproduce across
laboratories at AUC 0.636 (κ = 0.273; conservatively 0.681). Published predictors report
0.85–0.90. We trace the difference to peptide-level splitting that provides no protection,
training-set contamination of 26–38%, and negative-set definitions that alter the task
being measured. A leakage-free model reaches 0.573, attaining 84% of what the labels permit,
and the original convolutional architecture is statistically indistinguishable from it.
Progress in this area is currently limited by measurement reproducibility and epitope
coverage, and model architecture is not the constraint.


## 7. Methods

### 7.1 Data retrieval and cleaning

T-cell assay records were retrieved from the IEDB Query API restricted to MHC class I,
human host, and linear peptides of length 9–10 (92,721 records). Records were required to
carry a four-digit HLA-A/B/C allele present in our pseudo-sequence table and a canonical
twenty-letter peptide, leaving 65,732 records over 32,167 unique pairs. Any `Positive*`
qualitative measure was mapped to 1 and `Negative` to 0. Records were collapsed to one label
per pair by majority vote; 893 exact ties were dropped, giving 31,274 pairs, 31.7%
positive. A further 400,000 MHC binding and ligand-elution records were retrieved under
the same constraints for encoder pretraining.

### 7.2 Split schemes

Two schemes are stored in the same table: `split`/`fold` grouped by peptide, and
`study_split`/`study_fold` grouped by `reference_id`. Each uses a 15% grouped hold-out and
10 grouped CV folds.

### 7.3 Reliability measurement

Pairs assayed in ≥2 distinct publications were identified. Each study contributes its
majority outcome for that pair; within-study ties are dropped. Two studies are drawn at
random per pair, and we compute agreement, Cohen's κ, and the AUC of study A's label
predicting study B's. The point estimate is the mean over 200 random pairings of the full
set, so it does not depend on one arbitrary draw. The interval is a separate bootstrap that
resamples the pairs themselves with replacement, 2,000 times, drawing a fresh pairing each
time; percentiles taken across re-pairings of one fixed set of pairs would capture only how
the two studies are chosen and would be about half as wide. Because both labels are binary,
this AUC equals balanced accuracy. Sensitivity was assessed over four aggregation rules plus an
unaggregated variant, over assay families, over HLA locus, and stratified by the number of
studies testing a pair.

### 7.4 Models

*Published predictors.* DeepImmuno [2] (two-branch Conv2D), scored through its released
`cnn_model_331_3_7` checkpoint, which we verified byte-identical to the copy in this
repository; BigMHC_IM [9] (seven-network ensemble, presentation-pretrained then
transfer-learned); MixMHCpred 3.0 [11] (binding only, included as a control). PRIME 2.0 was
obtained but its bundled Windows binary did not execute; it is excluded and documented.
Where a table reports the DeepImmuno architecture retrained by us, it is labelled as such
and is a different set of weights from the released checkpoint.

*Baselines.* An allele prior predicting each allele's training-set positive rate, using no
peptide information; gradient boosting on the AAindex/PCA encoding.

*This study.* A bidirectional cross-attention network (65,987 parameters): learned residue
and position embeddings, self-attention within the 10-position peptide and the 46-residue
HLA pseudo-sequence, then cross-attention in both directions, a dense trunk and a sigmoid
output. Pretrained on the binding corpus (held-out binding AUC 0.958) and fine-tuned on
immunogenicity.

*Training schemes.* Pointwise cross-entropy; dropping the dominant study; study-size
reweighting; training on mixed-outcome studies only; and a study-stratified pairwise ranking
objective over (immunogenic, non-immunogenic) pairs sharing a study and an allele. The CNN
was trained from fresh initialisation, never from the published checkpoint, with identical
pretraining, folds, class weighting and seeds.

### 7.5 Evaluation

Pooled AUC is reported with two stricter quantities: within-allele AUC, computed inside each
allele; and within-study and within-study-and-allele AUC, computed inside each source
publication. Groups require ≥10 pairs and both outcome classes. Confidence intervals for
within-study quantities bootstrap over studies rather than rows. Models are scored from
study-grouped out-of-fold predictions. Published training sets were intersected with our
evaluation pairs by exact peptide and by exact peptide–HLA pair.


## Figures

**Figure 1. How the benchmark is distributed across its source studies.** Pairs
contributed per study against study rank, both axes logarithmic (solid, left axis), with
the cumulative share of all pairs (dashed, right axis). The largest study contributes
11,486 pairs, 37% of the benchmark, and reports no positive outcome; the ten largest
supply 54% of the data and the hundred largest 81%. Studies reporting a single outcome
class account for 57% of all pairs.

**Figure 2. What the labels can support, and what a leakage-free model achieves.**
Both panels are AUC on one scale, so the two reference bands run through both and are
labelled once: the light band is the measured reliability of the labels (0.636–
0.681), the dark band the 0.85–0.90 range reported in the literature. The dashed
line is chance. **(a)** AUC of one study's outcome predicting another's on the same
pair, with 95% confidence intervals, under three robustness checks: the rule used to
aggregate repeat records, how often the pair was retested, and the assay family and HLA
locus. Filled markers are the headline estimate (0.636) and the conservative bound
adopted in the text (0.681), which are the edges of the light band. No variant
approaches the dark one. **(b)** Within-study, within-allele AUC for six training
schemes, with intervals bootstrapped over studies. Every interval clears chance; the
best reaches 84% of the conservative bound from panel (a) and leaves 0.28 AUC between
itself and the reported range.

## Data and code availability

All analyses are reproducible from `docs/PIPELINE.md`, and
`notebooks/Reproduce_Manuscript.ipynb` reproduces every number and figure. Code is organised
as `src/data` (retrieval, cleaning, splits), `src/models` (architectures, training) and
`src/analysis` (all reported analyses). The curated benchmark, both split schemes and all
model predictions are released with the repository, together with the outputs of every
audited third-party predictor on our pairs, so the comparisons in §3.5 and §3.6 can be
reproduced without installing those tools. DeepImmuno's released checkpoint is scored
directly; `python -m src.models.deepimmuno` prints its held-out AUC in one command.

Archived at Zenodo: DOI TO BE INSERTED ON DEPOSIT. Repository:
`https://github.com/Terano24/Deep-Learning-Research-Project`. Code is released under the
MIT licence; redistributed third-party data retains its own terms, listed in `LICENSE`.

External resources: IEDB Query API; UniProt UP000005640; TRAP [3]; the harmonised datasets
of Müller et al. [4]; the public releases of DeepImmuno [2], BigMHC [9] and MixMHCpred [11].

## Author contributions

Both authors contributed to study design and manuscript preparation. I Nyoman Agung Ananda
Terano is the corresponding author and handles submission and correspondence.

## Competing interests

The authors declare no competing interests.

## Funding

This research received no specific grant from any funding agency.


## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used Claude (Anthropic) to draft and edit
the manuscript text. The research question, the study design, the choice of analyses and
the interpretation of the results are the authors' own. The authors reviewed and edited all
AI-assisted content and take full responsibility for the content of the published
article.

## References

1. Janetzki S, Panageas KS, Ben-Porat L, Boyer J, Britten CM, et al. Results and
   harmonization guidelines from two large-scale international Elispot proficiency panels
   conducted by the Cancer Vaccine Consortium (CVC/SVI). *Cancer Immunol Immunother*
   2008;57(3):303–315. doi:10.1007/s00262-007-0380-6
2. Li G, Iyer B, Prasath VBS, Ni Y, Salomonis N. DeepImmuno: deep learning-empowered
   prediction and generation of immunogenic peptides for T-cell immunity. *Brief Bioinform*
   2021;22(6):bbab160. doi:10.1093/bib/bbab160
3. Lee CH, Huh J, Buckley PR, Jang M, Pinho MP, et al. A robust deep learning workflow to
   predict CD8⁺ T-cell epitopes. *Genome Med* 2023;15:70. doi:10.1186/s13073-023-01225-z
4. Müller M, Huber F, Arnaud M, Kraemer AI, Ricart Altimiras E, et al. Machine learning
   methods and harmonized datasets improve immunogenic neoantigen prediction. *Immunity*
   2023;56(11):2650–2663.e6. doi:10.1016/j.immuni.2023.09.002
5. Wells DK, van Buuren MM, Dang KK, Hubbard-Lucey VM, Sheehan KCF, et al. Key parameters
   of tumor epitope immunogenicity revealed through a consortium approach improve neoantigen
   prediction. *Cell* 2020;183(3):818–834.e13. doi:10.1016/j.cell.2020.09.015
6. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based
   science. *Patterns* 2023;4(9):100804. doi:10.1016/j.patter.2023.100804
7. Richie R, Grover S, Tsui FR. Inter-annotator agreement is not the ceiling of machine
   learning performance: evidence from a comprehensive set of simulations. *Proc BioNLP
   Workshop, ACL* 2022:275–284.
8. Calis JJA, Maybeno M, Greenbaum JA, Weiskopf D, De Silva AD, Sette A, Peters B, Keşmir C.
   Properties of MHC class I presented peptides that enhance immunogenicity. *PLoS Comput
   Biol* 2013;9(10):e1003266.
9. Albert BA, Yang Y, Shao XM, Singh D, Smith KN, Anagnostou V, Karchin R. Deep neural
   networks predict MHC-I epitope presentation and transfer learn neoepitope immunogenicity.
   *Nat Mach Intell* 2023;5:861–872.
10. Gfeller D, Schmidt J, Croce G, Guillaume P, Bobisse S, et al. Improved predictions of
    antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0. *Cell Syst*
    2023;14(1):72–83.
11. Tadros DM, et al. Predicting MHC-I ligands across alleles and species: how far can
    we go? *Genome Med* 2025. (MixMHCpred 3.0 -- the version run here. This is the
    citation the tool's own output header requests; full author list and volume are
    still to be confirmed against the publisher record.)

*Reference details were verified against CrossRef and PubMed records where available.
Entries 8–11 should be re-checked against the publisher record before journal submission;
entry 11 in particular is taken from the MixMHCpred 3.0 output header and its author list
and volume are unconfirmed.*
