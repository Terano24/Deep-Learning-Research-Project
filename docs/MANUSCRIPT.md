# Label Reliability and Study-Level Leakage Bound Measurable Performance in T-Cell Epitope Immunogenicity Prediction

**I Nyoman Agung Ananda Terano**¹  **Mirrabell Frederica Hadiwijono**²

## Abstract

Predictors of CD8⁺ T-cell epitope immunogenicity report ROC-AUC values of 0.85 and above
under random cross-validation on Immune Epitope Database (IEDB) data, and are used to
prioritise vaccine candidates. We ask what those labels can support, and how benchmark construction shapes the
result. Among 65,732 human MHC-I T-cell assay records, 2,914 peptide–HLA pairs were assayed
independently by two or more publications; one laboratory's outcome predicts another's with
72.2% agreement, κ of 0.273, and an AUC-equivalent of 0.636 (95% CI 0.617–0.657), or
0.681 for pairs assayed by exactly two studies. Because a repeat experiment observes the
biology directly, reported performance exceeds what these labels can measure. Peptide
splitting, the field's standard precaution, lowers AUC by only 0.014 (0.826 against 0.840 for
random splitting), whereas holding out source publications gives 0.607; grouping
near-identical or overlapping peptides changes it by under 0.02, so sequence similarity is
not the cause. One all-negative study supplies 60% of that gap. An allele-identity
baseline that never sees the peptide reaches 0.752, above the 0.606, 0.647 and 0.559 of three
published predictors on the same pairs. DeepImmuno and BigMHC_IM had been trained on
37.5% and 27.1% of those pairs and fall by 0.23 and 0.26 AUC on pairs they had not seen.
Counting untested candidates as negatives rather than assayed negatives raises per-patient
AUC from ≈0.55 to ≈0.97. Under leakage-free evaluation our best model reaches 0.573
(0.544–0.603), 84% of the conservative bound, regardless of architecture. We release the
benchmark, both split schemes, all predictions, and a reporting protocol.

**Keywords:** T-cell epitope · immunogenicity prediction · data leakage · benchmark design
· label reliability · reproducibility · IEDB


^¹Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia. ²University of Toronto, Toronto, Canada. Correspondence to: I Nyoman Agung Ananda Terano (teranoace@gmail.com).

## Highlights

- IEDB immunogenicity labels reproduce between laboratories at only AUC 0.636
- Peptide-level splitting removes almost no leakage; the unit is the publication
- A peptide-blind allele baseline outscores three published predictors
- Two audited predictors were trained on 27-38% of the evaluation pairs
- Leakage-free evaluation gives AUC 0.573, 84% of what the labels permit

## 1. Introduction

Predictors of which peptide–MHC complexes trigger a CD8⁺ T-cell response are used to
design neoantigen vaccines and to choose targets for adoptive cell therapy. Personalised
neoantigen vaccines have reached randomised trials in melanoma and pancreatic cancer
(Ott et al., 2017; Sahin et al., 2017; Rojas et al., 2023; Weber et al., 2024), and which
candidates are synthesised rests on predictions of this kind. Experimental screening is slow and costly, so predictors trained on IEDB have multiplied,
and recent ones report ROC-AUC values of 0.85 and above under random cross-validation
(**Table 1**). Neoantigen pipelines lacked
reference data for comparison until a consortium experimentally tested 608 predicted
epitopes (Wells et al., 2020). Two established facts argue against taking
those numbers at face value.

Reported performance tracks the evaluation protocol more closely than it tracks the method
(**Table 1**). Under random cross-validation on pooled data, reported figures reach 0.85 to
0.94. Where an independently constructed test set is used, the same class of model scores
between 0.56 and 0.68. The first sequence-based model of immunogenicity reported 0.65, and
0.61 once immunogenic and non-immunogenic peptides were matched for binding affinity
(Calis et al., 2013). Lee et al. (2023) attribute the higher recent figures to dataset
composition, noting that "high reported performance from the latest models incorporating
peptide-HLA pairs might be driven by HLA bias from an imbalanced dataset". The reported
numbers therefore do not form a progression: they separate by how the data were split. This
paper measures why, and what the labels themselves permit.

**Table 1.** Reported performance of sequence-based CD8⁺ T-cell immunogenicity predictors,
grouped by the evaluation protocol that produced it. All values are ROC-AUC as reported by
the cited work; the verbatim source sentence for each is recorded in
`data/external/reported_performance.csv`. Our own estimates are shown for comparison.

| Predictor | Reported | Evaluation protocol |
|---|---|---|
| *Random cross-validation on pooled data* | | |
| TRAP, self-antigen model (Lee et al., 2023) | 0.943 | 10-fold cross-validation |
| DeepImmuno (Li et al., 2021) | 0.85 | 10-fold cross-validation on its IEDB training data |
| TRAP, pathogenic model (Lee et al., 2023) | 0.764 | 10-fold cross-validation |
| Calis et al. (2013) | 0.65 | cross-validation; 0.61 with binding affinity matched |
| *Held-out or independently constructed test set* | | |
| NeoaPred (Jiang et al., 2024) | 0.81 | held-out test set; adds pHLA surface and structural features |
| INeo-Epp (Wang et al., 2020) | 0.77 | external validation set (0.81 at its internal optimum) |
| T-SCAPE-IM (Kim et al., 2025) | 0.678 | newly constructed test set |
| Modular protein language model (O'Brien et al., 2024) | 0.619 | pMHC immunogenicity benchmark |
| BigMHC_IM, scored by Kim et al. (2025) | 0.569 | independent evaluation |
| MHCnuggets 2.4, scored by Kim et al. (2025) | 0.557 | independent evaluation |
| *This study* | | |
| Measured reliability of the labels (§3.2) | 0.636–0.681 | repeat experiments on the same pair |
| Leakage-free estimate (§3.8) | 0.573 | within study and allele, study-grouped |



First, peptide and allele do not determine the outcome. A response also depends on the
individual's T-cell receptor repertoire, prior antigen exposure, the rest of their HLA
genotype and the tissue context (Lu et al., 2026). Two people with the same restricting allele can respond
differently to the same peptide, so a model given only a peptide and an allele has a ceiling
below 1.0 whatever its architecture.

Second, laboratories measure the response inconsistently. International proficiency panels run by the Cancer Vaccine Consortium found that
laboratories differ substantially on the same material. In two ELISpot panels (36
laboratories in 2005, 29 in 2006) performance was scored by the ability to detect all
predefined responses correctly, and it improved significantly in the second panel once
protocol lessons were applied (Janetzki et al., 2008). A 27-laboratory HLA-peptide multimer panel found
significant differences across laboratories in both assay performance and the frequencies of
specific T cells reported (Britten et al., 2009).

Harmonisation guidelines aim to reduce this variation in prospective, coordinated studies
(Janetzki et al., 2008; Britten et al., 2009), but they cannot be applied retroactively to a database that aggregates decades of heterogeneous
protocols. To our knowledge, what that heterogeneity does to machine-learning benchmarks
built on IEDB has not been measured. Predictors deal with label disagreement by convention. PredIG, for example, calls a
peptide–HLA pair positive if any experiment found a responder and discards the conflicting
negative instances (Farriol-Duran et al., 2025); this is the any-positive rule whose effect on reliability we
measure in §3.3.

Related problems have been reported. Lee et al. documented HLA bias in immunogenicity
predictors and showed that cross-species splits lower performance (Lee et al., 2023). A benchmark of 50 TCR–epitope predictors found that all of them struggle to generalize to
unseen epitopes, and cautioned that adding external TCRs as negatives would "introduce
batch-like confounders, causing models to learn dataset-specific artifacts rather than true
TCR–epitope binding signals" (Lu et al., 2026). Tools also treat the separation of training from test
data inconsistently: T-SCAPE excludes fragments with fewer than two mismatches, and notes
that NetMHCpan removes only exact peptide matches and that BigMHC allows overlapping peptide
sequences when the allele differs (Kim et al., 2025). Kapoor and Narayanan reviewed 17 scientific fields and found
leakage affecting 294 papers; in one case study, complex models claimed to beat logistic
regression stopped doing so once leakage was corrected (Kapoor and Narayanan, 2023).

We are not aware of a published measurement of what the labels themselves can support, or
of which construction choices open the gap between that limit and reported performance.
This paper does both. Our
contributions are:

1. A measured reliability bound for IEDB immunogenicity labels, with sensitivity analyses
   over aggregation rule, retesting frequency, assay family and HLA locus.
2. A demonstration that peptide-level splitting does little to prevent leakage, with controls for sequence similarity, model class, assay family and the single largest study, replicated in direction on an independent dataset, and identification of source publication as the grouping unit that determines the result.
3. A contamination audit of two widely used published predictors against our evaluation set.
4. A leakage-free performance estimate, an architecture comparison showing the choice makes
   no measurable difference, and a worked example of a feature that standard validation
   endorses and grouped validation rejects.

## 2. Scope and definitions

**Scope.** We examine benchmarks for CD8⁺ T-cell epitope immunogenicity built from IEDB
assay records, where a model receives a peptide and an MHC-I allele and predicts whether a
T-cell response was observed. We do not examine MHC binding or ligand presentation
prediction, whose labels come from binding or mass-spectrometry measurements rather than
from a T-cell assay, and where pan-allele predictors such as NetMHCpan-4.1 and MHCflurry 2.0
report far higher figures on their own task (Reynisson et al., 2020; O'Donnell et al., 2020). We also leave out patient-specific neoantigen prioritisation pipelines, which use tumour sequencing and RNA
expression in addition to peptide sequence (Müller et al., 2023); §3.7 shows why conflating the two inflates apparent
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
mutual agreement (Richie et al., 2022). We return to this in §4.

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
the same pair (§7.3, **Table 2**).

**Table 2.** Agreement between independent publications assaying the same peptide–HLA pair
(n = 2,914). The AUC is the score one study's outcome achieves when used to predict
another's; because both labels are binary it equals balanced accuracy. Intervals bootstrap
over pairs (§7.3).

| Quantity | Value (95% CI) |
|---|---|
| Agreement | 0.722 (0.706–0.738) |
| Cohen's κ | **0.273** (0.235–0.312) |
| **AUC** | **0.636** (0.617–0.657) |

A κ of 0.273 (Cohen, 1960) denotes fair agreement only on the conventional scale
(Landis and Koch, 1977). The agreement between two experiments on the same
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
by locus, HLA-A 0.594, HLA-B 0.775. Every slice falls far below reported performance. Multimer and tetramer assays are the least reproducible family here, which fits the
significant between-laboratory differences found in a multimer proficiency panel (Britten et al., 2009),
although the two measurements are not directly comparable.

### 3.4 Peptide-level splitting does not prevent leakage

Holding out data by peptide is the standard precaution in this field. We compared grouping
units using one model class throughout, so that only the split changes (**Table 3**).

**Table 3.** Pooled AUC by grouping unit, on IEDB and on an independent dataset (TRAP).
One model class is used throughout, so only the held-out unit changes. TRAP records
antigen and organism rather than source publication, so the organism is its widest
available unit.

| Grouping unit | IEDB (this study) | TRAP (independent) (Lee et al., 2023) |
|---|---|---|
| Random | 0.840 | 0.857 |
| **Peptide** | **0.826** | **0.862** |
| Antigen | — | 0.817 |
| Source (study / organism) | **0.607** | **0.552** |

On the independent dataset, peptide grouping scores *above* random splitting. On IEDB it lowers AUC by only 0.014 (95% CI 0.009–0.024, resampling studies). In both datasets, widening the grouping unit to the data source collapses performance. Peptide grouping leaves most of the structure that inflates these results untouched.

Four further checks ask whether this is an artefact of how "peptide" was defined, of the model, of one study, or of mixing assays. Each cell of **Table 4** is pooled AUC from five-fold grouped cross-validation on IEDB
unless stated.

**Table 4.** Pooled AUC under similarity-based grouping rules and across model classes.
The allele prior uses no peptide information. Sequence-similarity rules move AUC by under
0.02 relative to exact-peptide grouping, whereas holding out source publications moves it
by more than 0.2.

| Grouping unit | Gradient boosting | Within-allele | Logistic regression | Allele prior (no peptide) |
|---|---|---|---|---|
| Random | 0.840 | 0.768 | 0.742 | 0.755 |
| **Exact peptide** | **0.826** | **0.740** | **0.739** | **0.756** |
| Peptides within 1 mismatch | 0.816 | 0.718 | 0.736 | 0.754 |
| Peptides within 2 mismatches | 0.818 | 0.723 | 0.736 | 0.753 |
| Peptides sharing a 6-residue stretch | 0.818 | 0.724 | 0.735 | 0.754 |
| **Source publication** | **0.607** | **0.435** | **0.548** | **0.465** |

**Sequence similarity is not the mechanism.** Holding out peptides within one or two mismatches of each other, or sharing a six-residue stretch (which groups overlapping windows of one source protein), lowers AUC by 0.008 to 0.010 more than exact-peptide grouping (six-residue rule: 0.008, 95% CI 0.005–0.017). Holding out source publications lowers it by a further 0.210. The model is not recognising near-copies of training peptides; it is recognising the study.

**It is not specific to gradient boosting.** Logistic regression gives the same ordering (0.739 under peptide grouping, 0.548 under source grouping). The allele prior, which sees no peptide, reaches 0.756 under peptide grouping and 0.465 under source grouping. Allele frequency predicts the outcome only because studies differ in which alleles they test and what they find; when studies are held out, the same signal points the wrong way.

**One study accounts for most of the gap, but not all of it.** The single all-negative study (§3.1) inflates the random and peptide-grouped scores. Without it (19,788 pairs remain), random and peptide grouping score 0.765 and 0.759 and source grouping scores 0.673. The random-to-source gap falls from 0.232 (95% CI 0.082–0.296) to 0.092 (0.066–0.118), and the peptide-to-source gap from 0.218 (0.066–0.283) to 0.085 (0.061–0.111). About 60% of the pooled gap therefore comes from this one study. The remainder is smaller but its interval excludes zero, so the effect does not rest on one study alone. It does mean the size of the gap is a property of this benchmark's composition and should not be carried over to others.

**Mixing assay types does not explain it.** Among pairs whose records all come from one assay family, source grouping still collapses performance, from 0.875 to 0.652 for ELISpot (16,888 pairs, 285 studies), from 0.810 to 0.649 for multimer and tetramer assays (6,273 pairs, 174 studies), and from 0.726 to 0.452 for intracellular cytokine staining (1,343 pairs, 100 studies, so less precise).

**A forward-in-time test agrees but is imprecise.** Studies were ordered by PubMed ID as a proxy for publication date. Models trained on the earlier 80% of dated pairs (14,484 pairs, PubMed ID up to 36,211,805) and were tested on the 3,611 pairs of 218 later studies, of which 65% are positive. Gradient boosting reaches 0.563 (95% CI 0.442–0.684, resampling studies), logistic regression 0.536 (0.412–0.671) and the allele prior 0.457 (0.381–0.555). These sit with the study-grouped estimates and the gradient boosting interval excludes the 0.82–0.84 of random and peptide splits, but the intervals also include chance, because 50% of the test pairs come from ten studies (the largest holds 274 pairs). Only 18,095 of 31,274 pairs carry a PubMed ID, so the check covers 58% of the benchmark.

### 3.5 A baseline that cannot see the peptide outscores published predictors

On the held-out test split (4,641 pairs, 1,470 positive), an allele-prior baseline reaches
0.752. It predicts each allele's training-set positive rate and uses no peptide information
whatsoever. On the same pairs BigMHC_IM reaches 0.647, DeepImmuno 0.606 and MixMHCpred
0.559: every published predictor scores below the baseline that never sees the peptide
(**Table 5**).

**Table 5.** Performance on the held-out test split (4,641 pairs, 1,470 positive). The two
DeepImmuno rows are different models: the released checkpoint, and the same architecture
retrained on our training split. Within-allele AUC is computed inside each allele, so the
allele prior scores 0.500 by construction.

| Predictor | Pooled AUC | Within-allele AUC |
|---|---|---|
| This study | 0.832 | 0.769 |
| Gradient boosting on engineered features | 0.810 | 0.721 |
| DeepImmuno architecture, retrained here | 0.792 | 0.661 |
| **Allele prior (no peptide information)** | **0.752** | 0.500 |
| BigMHC_IM (Albert et al., 2023) | 0.647 | 0.609 |
| DeepImmuno (Li et al., 2021) | 0.606 | 0.633 |
| MixMHCpred 3.0 (binding only) (Tadros et al., 2025) | 0.559 | 0.570 |

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
overlap, and performance falls sharply outside it (**Table 6**).

**Table 6.** Overlap between our evaluation pairs and the released training sets of two
published predictors, with performance inside and outside the overlap. Overlap is counted
by shared peptide. MixMHCpred's training set is not public and could not be audited.

| Predictor | Evaluation pairs it had seen | Seen | Unseen |
|---|---|---|---|
| DeepImmuno (Li et al., 2021) | 37.5% (shared peptide) | 0.746 | **0.512** |
| BigMHC_IM (Albert et al., 2023) | 27.1% (shared peptide) | 0.807 | **0.549** |

On the 778 lung-cancer-associated pairs whose sequence appears in neither training set,
every published predictor's interval includes chance: DeepImmuno 0.528 (0.476–0.583),
BigMHC 0.502 (0.438–0.561), MixMHCpred 0.446 (0.383–0.502), allele prior 0.465
(0.415–0.514).

### 3.7 The definition of a negative changes the task by ≈0.4 AUC

Using the per-patient supplementary table of the harmonised multi-cancer resource of Müller et al. (2023), which covers 99 patients (the original report also includes an in-house cohort,
for 131 in all), the negative-set definition alone moves per-patient AUC by about 0.4
(**Table 7**).

**Table 7.** Effect of the negative-set definition on per-patient AUC. The same 178
immunogenic peptides are ranked in both rows; only the set they are ranked against differs.

| What counts as a negative | Peptides compared | Per-patient AUC |
|---|---|---|
| Every candidate neo-peptide, screened or not | 1,780,481 (76% never assayed) | ≈ 0.97 |
| **Assayed negatives only** | **421,017 assayed (23.6% of candidates)** | **≈ 0.55** |

Of the 421,017 screened peptides, 178 were immunogenic. Ranking those 178 against *all*
candidates gives per-patient AUC ≈ 0.97; discriminating them from assayed negatives gives
≈ 0.55. Both are published as immunogenicity prediction. Most "negatives" in the former are
peptides nobody screened, overwhelmingly non-binders that any filter rejects.

### 3.8 Leakage-free performance approaches the measured ceiling, and architecture makes no measurable difference

We scored every model from study-grouped out-of-fold predictions, within study and allele,
across 205 groups, bootstrapping over studies (**Figure 2**, panel b; **Table 8**).

**Table 8.** Within-study and within-allele AUC for six training schemes, scored from
study-grouped out-of-fold predictions across 205 groups. Intervals bootstrap over studies.
All schemes share pretraining, folds, class weighting and seeds, so only the objective
differs.

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
determinants of TCR recognition (Calis et al., 2013).


## 4. Discussion

Reported AUCs of 0.85 and above (**Table 1**) are higher than the measured reliability of the labels (0.636, or
0.681 under the conservative bound) and higher than any model we could build under
leakage-free evaluation (0.573). We did not reproduce the published figures on their own
test sets, so what follows compares evaluation schemes on our data. We attribute the gap to
three things: peptide-level splitting, which lowers AUC by only 0.014 relative to random
splitting; training-set overlap, which we found in both audited predictors; and negatives
defined as untested candidates, which changes the task.

AUC is measured against noisy labels. A model that learns only transferable biology should
therefore score near the label reliability, and one that scores well above it is probably
also fitting the noise. Our results place that noise at the level of the study.

**Alternative explanations.** Four readings of §3.4 would weaken the argument, and the checks there address each. If the peptide split were merely too lenient about similarity, a homology-aware split would recover the study-grouped number; it does not, since sequence clusters move AUC by under 0.02. If the effect were specific to gradient boosting, logistic regression would differ; it follows the same ordering. If assay differences drove it, it would vanish within an assay family; it does not. If one study drove it, removing that study would remove the effect; it removes about 60% of the pooled gap and leaves 0.09. A forward-in-time test, the nearest this benchmark comes to deployment, gives 0.563 for gradient boosting, close to the leakage-free estimate, though the interval (0.442–0.684) is wide because half of the test pairs come from ten studies.

**Agreement as a bound.** Agreement between annotators does not cap model performance in
general. If annotators err independently, a well-specified model can beat their mutual
agreement (Richie et al., 2022). Our figures therefore describe performance against these labels, not how
accurate prediction could be in principle. Random label noise would make a high measured
AUC unattainable. High AUCs are attainable here, which indicates that the noise is
systematic and learnable, as §3.4–3.6 show.

**Relation to prior work.** Kapoor and Narayanan (2023) documented this pattern across 17
fields, and ours is an instance in computational immunology. Whalen et al. (2022) catalogue
the same failure modes in genomics, naming distributional differences between training and
test data and unaccounted-for confounders as the two that most often inflate reported
performance; source publication is both at once here. In their case study, complex
models claimed to beat logistic regression and ceased to do so once leakage was corrected;
in ours, published predictors are outscored by a baseline that cannot see the peptide. Our
setting also lets us measure the reliability of the labels directly instead of assuming it,
because IEDB records which publication produced each measurement and some pairs were
measured more than once. We extend Lee et al. (2023), who identified HLA bias and cross-species degradation, by
identifying the source publication as the operative grouping unit and by quantifying the
label ceiling. Recent tools have begun to hold out homologous sequences (Kim et al., 2025). Our results suggest that this is not sufficient: sequence clusters moved AUC by under 0.02,
whereas holding out publications moved it by more than 0.2.

**What bounds performance here.** At 0.573 against a conservative bound of 0.681, we are at
84% of what the labels permit, leaving ≈0.11 AUC of headroom. We do not think architecture
can recover it: the convolutional and attention models do not differ significantly,
and neoantigen methods that rank better draw on information that sequence-only benchmarks
lack, such as HLA presentation hotspots, binding promiscuity and the oncogenic role of the
mutated gene (Müller et al., 2023). Methods that add TCR sequence without patient context still struggle to generalize to unseen
epitopes (Lu et al., 2026), so the limit is not specific to peptide-only models.

Raising the ceiling requires harmonised assays of the kind whose adoption improved
performance between two successive proficiency panels (Janetzki et al., 2008). Broadening coverage requires more distinct epitopes, currently limited
by peptide-library production, and patient-specific prediction requires patient-specific
measurement.

### 4.1 Recommendations

These extend, for this task, the general reporting items of the DOME recommendations for
supervised learning in biology (Walsh et al., 2021), which ask for the data partitioning
scheme to be stated but do not prescribe a grouping unit.

1. Split by source publication, not only by peptide, and state the grouping unit used.
   Grouping by sequence similarity is not a substitute.
2. Report within-study and within-allele AUC alongside any pooled figure.
3. Include a peptide-blind allele-prior baseline. If it is competitive, the benchmark is
   measuring dataset structure.
4. State the negative-set definition explicitly; untested candidates and assayed negatives
   are different tasks.
5. Audit training-set overlap before reporting comparative performance.
6. Bootstrap over studies, not rows; peptides within a study are not independent.
7. Report whether the result holds within an assay family and without the largest single study.


## 5. Limitations

- Label reliability is measured on IEDB alone. The independent dataset available to us
  lacks publication identifiers, so §3.4 but not §3.2 could be replicated externally.
- Repeat-tested pairs are a selected subset. We address this by stratification and adopt the
  most favourable bound, but the estimate remains conditional on that subset.
- We did not reproduce the published values of Table 1 on the datasets and test sets of the
  papers that report them. The comparison is between evaluation schemes on our data, and
  between predictors on our pairs, not an audit of specific published numbers.
- Study-grouped evaluation could not be replicated on an independent dataset. No public
  immunogenicity dataset we found records the source publication of each label, and TRAP
  supports antigen and organism grouping only.
- About 60% of the pooled random-to-study gap depends on one all-negative study (§3.4). The
  gap would differ in a benchmark with another composition.
- The forward-in-time check covers 58% of pairs, those with a PubMed ID, uses ID order as a
  proxy for publication date, and its test pairs are concentrated in a few studies (50% come
  from ten), so its intervals are wide.
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
0.85 and above under random cross-validation, and 0.56 to 0.68 on independently constructed
test sets (**Table 1**). We find three features of benchmark construction that inflate apparent performance:
peptide-level splitting that removes almost none of the leakage (a 0.014 drop in AUC against
random splitting), training-set contamination of 27–38%, and negative-set definitions that
alter the task being measured. Most of the leakage is study structure rather than sequence
similarity, and about 60% of its size in this benchmark comes from one all-negative study. A leakage-free model reaches 0.573, attaining 84% of what the labels permit,
and, trained identically, the convolutional model scores 0.013 above the attention model
(95% CI -0.017 to +0.043), which is not significant.
Progress in this area is currently limited by measurement reproducibility and epitope
coverage, and model architecture is not the constraint.


## 7. Methods

### 7.1 Data retrieval and cleaning

T-cell assay records were retrieved from the IEDB (Vita et al., 2019) Query API restricted to MHC class I,
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

The checks in §3.4 add three similarity-based grouping rules, each used with five grouped
folds. Peptides of equal length within one or two mismatches are merged by single linkage.
Candidate pairs come from a pigeonhole index (two peptides within d mismatches must agree
exactly on one of d+1 segments) and are then checked exactly. Peptides sharing any exact
six-residue substring are also merged; a five-residue rule was rejected because it joined
14.5% of all pairs into one cluster. Peptides of different lengths are never merged. For the
forward-in-time test each study is dated by its smallest PubMed ID, and the training set is
every pair from studies up to the 80th percentile of dated pairs. A pair counts as belonging
to an assay family only if all of its records come from that family, the family being the
last item of the IEDB assay name.

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

*Published predictors.* DeepImmuno (Li et al., 2021) (two-branch Conv2D), scored through its released
`cnn_model_331_3_7` checkpoint, which we verified byte-identical to the copy in this
repository; BigMHC_IM (Albert et al., 2023) (seven-network ensemble, presentation-pretrained then
transfer-learned); MixMHCpred 3.0 (Tadros et al., 2025), binding only, included as a control. PRIME 2.0 (Gfeller et al., 2023) was obtained but its bundled Windows binary did not execute; it is excluded
and documented.
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

We report ROC-AUC throughout, for comparability with the figures of **Table 1**. ROC-AUC and
PR-AUC can rank methods differently when classes are imbalanced (Saito and Rehmsmeier,
2015); PR-AUC for every predictor is included in the released result files rather than
tabulated here. Pooled AUC is reported with two stricter quantities: within-allele AUC, computed inside each
allele; and within-study and within-study-and-allele AUC, computed inside each source
publication. Groups require ≥10 pairs and both outcome classes. Confidence intervals for
within-study quantities bootstrap over studies rather than rows. Models are scored from study-grouped out-of-fold predictions. Differences between grouping rules, and the forward-in-time intervals, resample whole studies with replacement (1,000 replicates), because pairs within a study are not independent. Differences are computed from unrounded values, so they can differ by 0.001 from the difference of two rounded figures in a table. Published training sets were intersected with our
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
0.681), the dark band the 0.85–0.94 range reported under random cross-validation (**Table 1**). The dashed
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

Archived at Zenodo: `10.5281/zenodo.23719454` for all versions, and
`10.5281/zenodo.23719455` for v1.0.0, the release that produced every number reported
here. Repository:
`https://github.com/Terano24/Deep-Learning-Research-Project`. Code is released under the
MIT licence; redistributed third-party data retains its own terms, listed in `LICENSE`.

External resources: IEDB Query API; UniProt UP000005640; TRAP (Lee et al., 2023); the harmonised datasets
of Müller et al. (2023); the public releases of DeepImmuno (Li et al., 2021), BigMHC (Albert et al., 2023) and MixMHCpred (Tadros et al., 2025).

## CRediT authorship contribution statement

**I Nyoman Agung Ananda Terano:** Conceptualization, Methodology, Software, Formal
analysis, Data curation, Investigation, Visualization, Writing – original draft, Writing –
review & editing. **Mirrabell Frederica Hadiwijono:** Conceptualization, Methodology,
Investigation, Validation, Writing – original draft, Writing – review & editing.

Both authors contributed to study design and manuscript preparation, read and approved the
final manuscript, and accept responsibility for its content. I Nyoman Agung Ananda Terano
is the corresponding author and handles submission and correspondence.

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

Albert, B.A., Yang, Y., Shao, X.M., Singh, D., Smith, K.N., Anagnostou, V., Karchin, R., 2023.
Deep neural networks predict class I major histocompatibility complex epitope presentation and
transfer learn neoepitope immunogenicity. Nat. Mach. Intell. 5 (8), 861–872.
https://doi.org/10.1038/s42256-023-00694-6

Britten, C.M., Janetzki, S., Ben-Porat, L., Clay, T.M., Kalos, M., Maecker, H., Odunsi, K.,
Pride, M., Old, L., Hoos, A., Romero, P., 2009. Harmonization guidelines for HLA-peptide multimer
assays derived from results of a large scale international proficiency panel of the Cancer Vaccine
Consortium. Cancer Immunol. Immunother. 58 (10), 1701–1713.
https://doi.org/10.1007/s00262-009-0681-z

Calis, J.J.A., Maybeno, M., Greenbaum, J.A., Weiskopf, D., De Silva, A.D., Sette, A., Peters, B.,
Keşmir, C., 2013. Properties of MHC class I presented peptides that enhance immunogenicity.
PLoS Comput. Biol. 9 (10), e1003266. https://doi.org/10.1371/journal.pcbi.1003266

Cohen, J., 1960. A coefficient of agreement for nominal scales. Educ. Psychol. Meas. 20 (1),
37–46. https://doi.org/10.1177/001316446002000104

Farriol-Duran, R., Domínguez-Dalmases, C., Cañellas-Solé, A., Vazquez, M., Porta-Pardo, E.,
Guallar, V., 2025. PredIG: an interpretable predictor of T-cell epitope immunogenicity.
Genome Med. 17, 140. https://doi.org/10.1186/s13073-025-01569-8

Gfeller, D., Schmidt, J., Croce, G., Guillaume, P., Bobisse, S., Genolet, R., Queiroz, L.,
Cesbron, J., Racle, J., Harari, A., 2023. Improved predictions of antigen presentation and TCR
recognition with MixMHCpred2.2 and PRIME2.0 reveal potent SARS-CoV-2 CD8⁺ T-cell epitopes.
Cell Syst. 14 (1), 72–83.e5. https://doi.org/10.1016/j.cels.2022.12.002

Janetzki, S., Panageas, K.S., Ben-Porat, L., Boyer, J., Britten, C.M., Clay, T.M., Kalos, M.,
Maecker, H.T., Romero, P., Yuan, J., Kast, W.M., Hoos, A., 2008. Results and harmonization
guidelines from two large-scale international Elispot proficiency panels conducted by the Cancer
Vaccine Consortium (CVC/SVI). Cancer Immunol. Immunother. 57 (3), 303–315.
https://doi.org/10.1007/s00262-007-0380-6

Jiang, D., Qi, Y., Zhang, H., Zhang, Y., Zhang, L., Wang, J., Chen, J., Zhu, W., Li, Y., Liu, J.,
2024. NeoaPred: a deep-learning framework for predicting immunogenic neoantigen based on surface
and structural features of peptide–human leukocyte antigen complexes. Bioinformatics 40 (9),
btae547. https://doi.org/10.1093/bioinformatics/btae547

Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in machine-learning-based
science. Patterns 4 (9), 100804. https://doi.org/10.1016/j.patter.2023.100804

Kim, J., Jung, N., Lee, J., Cho, N.H., Noh, J., Seok, C., 2025. T-SCAPE: T cell immunogenicity
scoring via cross-domain aided predictive engine. Sci. Adv. 11 (49).
https://doi.org/10.1126/sciadv.adz8759

Landis, J.R., Koch, G.G., 1977. The measurement of observer agreement for categorical data.
Biometrics 33 (1), 159–174. https://doi.org/10.2307/2529310

Lee, C.H., Huh, J., Buckley, P.R., Jang, M., Pinho, M.P., Fernandes, R.A., Antanaviciute, A.,
Simmons, A., Koohy, H., 2023. A robust deep learning workflow to predict CD8⁺ T-cell epitopes.
Genome Med. 15, 70. https://doi.org/10.1186/s13073-023-01225-z

Li, G., Iyer, B., Prasath, V.B.S., Ni, Y., Salomonis, N., 2021. DeepImmuno: deep
learning-empowered prediction and generation of immunogenic peptides for T-cell immunity.
Brief. Bioinform. 22 (6), bbab160. https://doi.org/10.1093/bib/bbab160

Lu, Y., Wang, Y., Xu, M., Xie, B., Yang, Y., Xu, H., Suo, S., 2026. Assessment of computational
methods in predicting TCR–epitope binding recognition. Nat. Methods 23 (1), 248–259.
https://doi.org/10.1038/s41592-025-02910-0

Müller, M., Huber, F., Arnaud, M., Kraemer, A.I., Ricart Altimiras, E., Michaux, J., Taillandier-
Coindard, M., Chiffelle, J., Murgues, B., Gehret, T., Auger, A., Stevenson, B.J., Coukos, G.,
Harari, A., Bassani-Sternberg, M., 2023. Machine learning methods and harmonized datasets improve
immunogenic neoantigen prediction. Immunity 56 (11), 2650–2663.e6.
https://doi.org/10.1016/j.immuni.2023.09.002

O'Brien, H., Salm, M., Morton, L.T., Szukszto, M., O'Farrell, F., Boulton, C., King, L.,
Bhuva, M., Reading, J.L., Litchfield, K., 2024. A modular protein language modelling approach to
immunogenicity prediction. PLoS Comput. Biol. 20 (11), e1012511.
https://doi.org/10.1371/journal.pcbi.1012511

O'Donnell, T.J., Rubinsteyn, A., Laserson, U., 2020. MHCflurry 2.0: improved pan-allele
prediction of MHC class I-presented peptides by incorporating antigen processing.
Cell Syst. 11 (1), 42–48.e7. https://doi.org/10.1016/j.cels.2020.06.010

Ott, P.A., Hu, Z., Keskin, D.B., Shukla, S.A., Sun, J., Bozym, D.J., Zhang, W., Luoma, A.,
Giobbie-Hurder, A., Peter, L., et al., 2017. An immunogenic personal neoantigen vaccine for
patients with melanoma. Nature 547 (7662), 217–221. https://doi.org/10.1038/nature22991

Reynisson, B., Alvarez, B., Paul, S., Peters, B., Nielsen, M., 2020. NetMHCpan-4.1 and
NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif
deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Res. 48 (W1),
W449–W454. https://doi.org/10.1093/nar/gkaa379

Richie, R., Grover, S., Tsui, F.R., 2022. Inter-annotator agreement is not the ceiling of machine
learning performance: evidence from a comprehensive set of simulations. In: Proceedings of the
21st Workshop on Biomedical Language Processing (BioNLP), ACL, pp. 275–284.
https://doi.org/10.18653/v1/2022.bionlp-1.26

Rojas, L.A., Sethna, Z., Soares, K.C., Olcese, C., Pang, N., Patterson, E., Lihm, J.,
Ceglia, N., Guasp, P., Chu, A., et al., 2023. Personalized RNA neoantigen vaccines stimulate
T cells in pancreatic cancer. Nature 618 (7963), 144–150.
https://doi.org/10.1038/s41586-023-06063-y

Sahin, U., Derhovanessian, E., Miller, M., Kloke, B.P., Simon, P., Löwer, M., Bukur, V.,
Tadmor, A.D., Luxemburger, U., Schrörs, B., et al., 2017. Personalized RNA mutanome vaccines
mobilize poly-specific therapeutic immunity against cancer. Nature 547 (7662), 222–226.
https://doi.org/10.1038/nature23003

Saito, T., Rehmsmeier, M., 2015. The precision-recall plot is more informative than the ROC
plot when evaluating binary classifiers on imbalanced datasets. PLoS One 10 (3), e0118432.
https://doi.org/10.1371/journal.pone.0118432

Tadros, D.M., Racle, J., Gfeller, D., 2025. Predicting MHC-I ligands across alleles and species:
how far can we go? Genome Med. 17 (1). https://doi.org/10.1186/s13073-025-01450-8

Vita, R., Mahajan, S., Overton, J.A., Dhanda, S.K., Martini, S., Cantrell, J.R.,
Wheeler, D.K., Sette, A., Peters, B., 2019. The Immune Epitope Database (IEDB): 2018 update.
Nucleic Acids Res. 47 (D1), D339–D343. https://doi.org/10.1093/nar/gky1006

Walsh, I., Fishman, D., Garcia-Gasulla, D., Titma, T., Pollastri, G., ELIXIR Machine
Learning Focus Group, et al., 2021. DOME: recommendations for supervised machine learning
validation in biology. Nat. Methods 18 (10), 1122–1127.
https://doi.org/10.1038/s41592-021-01205-4

Wang, G., Wan, H., Jian, X., Li, Y., Ouyang, J., Tan, X., Zhao, Y., Lin, Y., Xie, L., 2020.
INeo-Epp: a novel T-cell HLA class-I immunogenicity or neoantigenic epitope prediction method
based on sequence-related amino acid features. BioMed Res. Int. 2020, 5798356.
https://doi.org/10.1155/2020/5798356

Weber, J.S., Carlino, M.S., Khattak, A., Meniawy, T., Ansstas, G., Taylor, M.H., Kim, K.B.,
McKean, M., Long, G.V., Sullivan, R.J., et al., 2024. Individualised neoantigen therapy
mRNA-4157 (V940) plus pembrolizumab versus pembrolizumab monotherapy in resected melanoma
(KEYNOTE-942): a randomised, phase 2b study. Lancet 403 (10427), 632–644.
https://doi.org/10.1016/S0140-6736(23)02268-7

Wells, D.K., van Buuren, M.M., Dang, K.K., Hubbard-Lucey, V.M., Sheehan, K.C.F., Campbell, K.M.,
Lamb, A., Ward, J.P., Sidney, J., Blazquez, A.B., et al., 2020. Key parameters of tumor epitope
immunogenicity revealed through a consortium approach improve neoantigen prediction. Cell 183 (3),
818–834.e13. https://doi.org/10.1016/j.cell.2020.09.015

Whalen, S., Schreiber, J., Noble, W.S., Pollard, K.S., 2022. Navigating the pitfalls of
applying machine learning in genomics. Nat. Rev. Genet. 23 (3), 169–181.
https://doi.org/10.1038/s41576-021-00434-9
