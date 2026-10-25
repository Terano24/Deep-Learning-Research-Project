# Reproducing the benchmark

Every result in `results/` is produced by the commands below, run from the repository
root with the pinned environment in `requirements.txt`. All random seeds are fixed
(`SEED = 42`); the only expected variation is TensorFlow's non-deterministic CPU kernels,
which move AUCs by well under 0.01.

## 1. Data

```bash
python -m src.data.fetch_data              # IEDB T-cell assays, MHC-I, human, 9/10-mers
python -m src.data.fetch_data binding      # IEDB binding/presentation assays (pretraining)
python -m src.data.dataset                 # cleaning, labelling, and the two split schemes
```

`fetch_data` pages the IEDB Query API and writes `data/raw/iedb_tcell_mhc1.csv`
(92,721 records) and `data/raw/iedb_mhc_binding.csv` (400,000 records). The binding
fetch appends each page to disk and resumes, so an interrupted run can be restarted.

`dataset.py` collapses records to unique peptide-HLA pairs by majority vote (ties
dropped), then assigns **two independent split schemes** into the same table:

| Columns | Grouped by | Purpose |
|---|---|---|
| `split`, `fold` | peptide | the conventional evaluation |
| `study_split`, `study_fold` | `reference_id` (source publication) | the strict evaluation |

Result: `data/processed/dataset.csv`, 31,274 pairs, 31.7% positive.

The human proteome used for self-similarity features is not in the repo:

```bash
curl -L "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000005640+AND+reviewed:true&format=fasta&compressed=true" \
  -o data/external/human_proteome.fasta.gz
python -m src.data.features                # caches data/processed/peptide_features.csv
```

## 2. Models

```bash
python -m src.models.train                           # original CNN, leakage-safe retrain
python -m src.models.train --rescore                 # rebuild test_predictions.csv only
python -m src.models.improve pretrain --epochs 12    # binding pretraining (held-out AUC 0.958)
python -m src.models.improve cv --variants pretrained_lung_feat --folds 3
python -m src.models.improve final --variant pretrained_lung_feat --seeds 5
python -m src.models.rank --epochs 25 --pairs-per-group 800   # study-stratified ranking
```

`train.py` writes `results/test_predictions.csv` with one column per predictor.
`--rescore` rebuilds that file from the saved weights without retraining anything.

### The two CNN checkpoints are not interchangeable

| Path | What it is | Test AUC |
|---|---|---|
| `models/CNN_WEIGHT_OPSI` | DeepImmuno's released `cnn_model_331_3_7`, byte for byte | 0.606 |
| `models/CNN_RETRAINED` | the same architecture, trained here on our split | 0.792 |

The published checkpoint is the one that may be called DeepImmuno, and it is only ever
loaded through `src/models/deepimmuno.py`. Our copy is written to the `cnn_retrained`
column. An earlier version of this pipeline wrote ours to a column named `cnn` and the
analyses renamed that column to `deepimmuno`, reporting our model as the published one;
the column names are explicit now so that cannot recur. Check with:

```bash
python -m src.models.deepimmuno            # prints the published checkpoint's test AUC
```

## 3. Analyses

```bash
python -m src.analysis.confounds           # within-allele AUC; allele-prior confound
python -m src.analysis.leakage             # random vs peptide- vs study-grouped splits
python -m src.analysis.within_study        # within-study ranking, the operational task
python -m src.analysis.robustness --folds 5  # H2/H3/H4: mitigation schemes compared
python -m src.analysis.external_validation # TRAP: does the effect generalise?
python -m src.analysis.fair_vs_deepimmuno  # contamination-aware DeepImmuno comparison
python -m src.analysis.external_baselines  # BigMHC / MixMHCpred / DeepImmuno
python -m src.analysis.feature_ablation    # do self-similarity features help?
```

## 4. External baselines

Their outputs on our pairs are committed under `data/external/` (see the README there),
so §3.5 and §3.6 can be reproduced without installing any of these tools. The commands
below are how those files were generated.

**BigMHC** (immunogenicity; ~2.5 GB of weights, needs `torch`)

```bash
git clone --depth 1 https://github.com/KarchinLab/bigmhc
cd bigmhc/src
python predict.py -i=INPUT.csv -m=im -d=cpu -o=OUT.prd
```
Input is a CSV with an allele column (`-a`, default 0) in `HLA-A*02:01` form and a
peptide column (`-p`, default 1). Our dataset stores alleles without the colon, so
`src/analysis/external_baselines.py` converts them back.

**MixMHCpred 3.0** (binding only — included as a control)

```bash
git clone --depth 1 https://github.com/GfellerLab/MixMHCpred
./MixMHCpred -i peptides.txt -o out.txt -a A0201
```
One run per allele; allele nomenclature is `A0201`, not `HLA-A*02:01`.

**PRIME 2.1** — not used. The bundled `PRIME.exe` exits 0 without writing output on
Windows. Paths are passed to it as msys (`/c/Users/...`) rather than Windows paths; a
`cygpath` conversion in `lib/run_PRIME.pl` fixes that but does not fix the binary.
Recompiling `lib/PRIME.cc` on Linux is the likely route if this baseline is wanted.

## Known limitations

- Peptide lengths are restricted to 9 and 10, inherited from the DeepImmuno encoding.
- BigMHC's immunogenicity training set (Mendeley `10.17632/dvmz6pkzvb`) is included as
  `data/external/bigmhc_im_train.csv` and is what the 26.5% overlap figure is measured
  against. MixMHCpred's training set is not public, so it could not be audited the same
  way.
- The binding pretraining set is capped at 400,000 records.
- MixMHCpred has no model for `HLA-B*4102` (1 pair in the test split, scored as missing).
