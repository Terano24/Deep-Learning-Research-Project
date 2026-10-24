# The short version

The plan was to finish the OPSI model and publish it. While it was being rebuilt properly,
it became clear that the thing being improved doesn't work as well as anyone thought — not
just ours, but every published tool in this area. So the paper changed from *"we built a
lung cancer predictor"* to *"here's why nobody's predictor works as well as they report."*

That sounds like bad news. It isn't. The second paper is more defensible, more useful, and
genuinely publishable. The first one wasn't.

---

## Problem 1: the weights aren't ours

This is the uncomfortable one, and it's better heard here than from a reviewer.

The model weights in `models/CNN_WEIGHT_OPSI/` are **byte-for-byte identical** to the
published weights of DeepImmuno, a 2021 tool from the University of Cincinnati. An MD5
hash confirms it — the same file, exactly:

```
5cb6f9b8122c2ed25d1bc4973f4f14c0   DeepImmuno's published weights
5cb6f9b8122c2ed25d1bc4973f4f14c0   models/CNN_WEIGHT_OPSI
```

The architecture matches too — line for line, same layers, same kernel sizes.
`arsitekturCNN()` is their `seperateCNN()` with a different function name.

**To be fair about this:** using a published open-source model is completely normal and
legitimate. DeepImmuno is public precisely so that people can use it. The problem is only
that the OPSI paper described it as a model built and trained by us, and reported its
performance as ours. For a high-school competition entry that's an understandable framing
mistake. For a journal submission it would be a serious problem — these things get checked,
DeepImmuno is well known in this niche, and the consequence would attach to both names.

So the OPSI paper can't be revised and resubmitted. It has to be treated as superseded.

## Problem 2: lung cancer specifically doesn't have enough data

Once real IEDB data was in place, the lung-cancer subset turned out to be:

- 1,109 peptide–HLA pairs, only 208 positive
- from 67 studies, but **59% of records come from just 3 papers**
- only 9 alleles have both positive and negative examples
- 81% of lung studies report only one outcome class

Lung-specific performance came out at AUC 0.555, confidence interval 0.438–0.670. Chance
(0.5) sits inside that interval. There is no lung-cancer signal to report.

Later, using an independent 99-patient dataset, lung turned out **not** to be harder than
other cancers — it actually had the highest screening yield. So the difficulty was never
biology. It was how IEDB's lung records happen to be curated.

---

## What turned up instead

Chasing *why* the numbers were bad produced something better. Four findings:

**1. The labels themselves are unreliable.** There are 2,914 peptide–HLA pairs in IEDB that
were tested by two or more different laboratories. If lab A says "immunogenic," does lab B
agree? Agreement is 72%, Cohen's κ = 0.27 — "fair," barely above chance. Expressed as AUC,
one experiment predicts another at **0.637**.

That's the key number. A repeat experiment is the best predictor there is, because it
observes the biology directly. If *that* only reaches 0.637, then papers reporting 0.85–0.90
are reporting something the labels cannot support.

This matches what immunologists already knew about the assays: international proficiency
panels found ELISpot results varying 35-fold between laboratories and tetramer assays
100-fold. What nobody had checked is what that does to the machine-learning benchmarks
built on top of them.

**2. The standard way everyone splits data doesn't work.** Test data is held out so a model
can't cheat. Everyone in this field holds out by *peptide*. Splitting by peptide turns out
to perform identically to not splitting at all (0.826 versus 0.840 for a random split).
Splitting by **source publication** instead drops it to 0.607 — because a single
all-negative study makes up 37% of the entire dataset, and 75% of studies only ever report
one answer. The model learns to recognise which paper the data came from.

The clearest demonstration: a "model" that ignores the peptide entirely and predicts each
HLA allele's base rate scores **0.752** — beating BigMHC, a well-regarded published tool, at
0.647.

**3. Published tools were trained on our test data.** DeepImmuno had seen 37.5% of the test
peptides; BigMHC had seen 26.5% of the lung pairs. Both score much worse on data they had
not seen (BigMHC: 0.807 → 0.549).

**4. "Immunogenicity prediction" means two different tasks.** Using the Müller et al. data
you downloaded: ranking immunogenic peptides against *all* candidates — 76% of which were
never tested — gives AUC ≈ 0.97. Distinguishing tested positives from tested negatives gives
≈ 0.55. Same name, completely different difficulty.

---

## What the paper says now

> Published immunogenicity predictors report 0.85–0.90. The labels they are measured against
> only reproduce at 0.637. Here is exactly where the gap comes from, and here is what honest
> performance looks like.

The best model built here reaches **0.557** under strict evaluation — 82% of what the labels
can support. That is reported as a realistic ceiling, not as a win.

**Worth noting:** the paper explicitly states that this model is *not* an improvement. A
feature set that looked promising turned out to be detecting which study the data came
from rather than anything about peptides — and that failure is written into the paper as a
worked example of the method catching a false result. Reporting your own failures is what
makes a methods paper credible.

---

## Why this is worth publishing

- **Novel.** Repeated literature searches found no one who has measured label
  reproducibility for IEDB immunogenicity data, or who splits this data by source publication.
- **Useful.** It affects every group using these tools for vaccine design.
- **Hard to argue with.** It rests on hashes, overlap counts and confidence intervals — not
  on a claim that our model is better.
- **Precedented.** Kapoor & Narayanan did exactly this for machine learning across 17 fields
  (*Patterns*, 2023), and it is heavily cited. They built no model either.

---

## Journal target

**Computational Biology and Chemistry** (Elsevier, Scopus Q3), with IEEE TCBB (Q2) as a
backup and PLOS Computational Biology (Q1) as a stretch.

Worth discussing: open-access fees run US$2,500–4,400, but U of T's read-and-publish
agreements cover them fully for Elsevier hybrid journals and PLOS — provided the
**corresponding author** is U of T affiliated, which as a U of T student you are. That role
also means handling reviewer correspondence, so it needs talking through properly. The
manuscript is at `docs/MANUSCRIPT.pdf` — comments on the immunology framing would help most.
