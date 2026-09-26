# Business Entity Resolution: reproduction

Offline pipeline using only the provided train/test data (no external lookups, no external data).
Models: LightGBM (MIT). Python 3.11; versions in `requirements.txt` (polars 1.44.2 is required: fold
assignment and hashed keys depend on polars' hash function).

Data layout: `BER_DATA_DIR` = the `dataset/` folder (train/, test/), whose parent holds
`utils/validate_submission.py`. `BER_WORK_DIR` = scratch folder (~15 GB).

## End to end (from `src/`)
```bash
A="--data-dir $BER_DATA_DIR --work-dir $BER_WORK_DIR --n-jobs 4"
python run.py prepare --split all $A          # parse + normalize
python run.py translit --split all $A         # cross-script name skeletons
python run.py folds $A                        # 5 S1-disjoint folds, fold 4 sealed
python run.py rank-train $A                   # learned candidate ranker (cross-fitted A/B)
python run.py rank --split train $A && python run.py rank --split test $A
python run.py features --split train --variant c $A && python run.py features --split test --variant c $A
python run.py train-b $A                      # stage 1 LightGBM + isotonic calibration
python run.py score-b --split test $A
python run.py stack-features --split train $A && python run.py stack-features --split test $A
python run.py train-c $A                      # stage 2 stacked LightGBM
python run.py stack-features --split test $A  # (test stage-2 features)
# decision layer + output files (see alt/):
python alt/run_ownership.py                   # fits the target-ownership model on dev folds
python alt/make_upload.py --work $BER_WORK_DIR --out ../output --owner --country-shift US=1.0,India=0.45
python alt/write_candidates.py $BER_WORK_DIR ../output/candidate_pairs.tsv
```
`candidate_pairs.tsv` is exactly the set of pairs scored by the matcher (top-50 ranked candidates
per S1); `matching_results.tsv` ⊆ candidates. Both are checked with
`utils/validate_submission.py --check-ids`.

Memory: `blocking._addr_keys` builds keys in 1M-row slices so the ranker fits in 15 GB RAM.
