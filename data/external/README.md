# External predictor outputs and training sets

Everything the manuscript's comparisons against published predictors depends on, in a
form that does not require re-installing those predictors. Each file was produced by
running the released tool on our pairs; the scores are theirs, not ours.

Before this directory existed, these analyses read from an absolute path into a
machine-specific temporary folder, so nobody else could reproduce §3.5 or §3.6.

| file | rows | what it is |
|---|---|---|
| `bigmhc_im_test.csv` | 4,641 | BigMHC_IM on the held-out test split |
| `bigmhc_im_lung.csv` | 1,109 | BigMHC_IM on every lung-cancer-associated pair |
| `mixmhcpred_test.csv` | 4,640 | MixMHCpred 3.0 on the test split (one pair unscored) |
| `mixmhcpred_lung.csv` | 1,109 | MixMHCpred 3.0 on the lung pairs |
| `deepimmuno_train.csv` | 8,971 | DeepImmuno's released training pairs, for the overlap audit |
| `bigmhc_im_train.csv` | 7,940 | BigMHC's immunogenicity transfer-learning set, same purpose |
| `trap_pathogenic_db.csv` | 10,097 | TRAP's curated pathogenic epitope set (Lee et al. 2023) |

## How each was generated

**BigMHC_IM** — `python bigmhc/src/predict.py -i <pairs>.csv -m im -d 0`, the seven-model
`im` ensemble (`bat512` … `bat32768`). Input columns `mhc,pep` with alleles as
`HLA-A*02:01`; the `BigMHC_IM` output column is kept and alleles are rewritten to this
project's `HLA-A*0201` form.

**MixMHCpred 3.0** — one run per allele, since the per-allele score column is the one we
want. From each `<allele>.out` we take `Score_<allele>`, not `Score_bestAllele`. Higher
is a better binder. MixMHCpred has no immunogenicity training at all and is included as
a control: if a pure binding predictor matches the immunogenicity models, that supports
the claim that most apparent signal is binding plus allele identity.

**DeepImmuno** is *not* here, because it is scored live from its released checkpoint at
`models/CNN_WEIGHT_OPSI` — see `src/models/deepimmuno.py`. Only its training set is
here, for measuring overlap with our evaluation pairs.

## Note on the two CNN checkpoints

`models/CNN_WEIGHT_OPSI` is DeepImmuno's published `cnn_model_331_3_7`, byte for byte.
`models/CNN_RETRAINED` is the same architecture trained by us. They score very
differently (0.606 against 0.792 on the held-out split), so a column holding one must
never be relabelled as the other.
