# Hand-off: apply the validated steps to the 0.982 model (V13-V15 + transformer)

All three steps work on any v10-lineage pipeline. None uses external data or test labels.

1. **France self-training of stage 2.** Validated leave-one-country-out: US labels only, India
   unseen, +0.0039 on India's held-out fold. On v10 test, French number agreement rose
   0.9896 → 0.9969.
   - Take your test stage-2 features and your calibrated stage-2 p for France pairs.
   - Pseudo-positives: p ≥ 0.98 and `num_match == 1`. Pseudo-negatives: p ≤ 0.02, sampled like
     your train negatives.
   - Add a feature `dom` (0 for train rows, 1 for France rows). Refit stage 2 once on train dev
     rows plus the pseudo rows, weight 1.0. More rounds hurt in the LOCO test.
   - Rescore France pairs with dom=1. Decide with single_owner + exact expected-F0.5 at shift 0.6.
   - Code: `alt/src/france_selftrain.py`; swap rows with `alt/src/swap_country.py`.
2. **Separate strictness per country, measured at test density (stress-v2):**
   - US 1.0: +0.0012 vs 0.6.
   - India 0.45: +0.0002.
   - Code: `alt/src/make_upload.py --country-shift US=1.0,India=0.45`.
3. **Target-ownership layer:** +0.0003 on fold 4. Code: `alt/src/ownership.py`, `run_ownership.py`.

What didn't help, so don't spend time on it:
- count rescaling for test density (US loses only 0.0035 at test density);
- rescuing blank-address records by exact name (48% precise);
- a char-3gram name channel (recovers 31% of US and 9% of India misses, +23 pairs/S1, ceiling about +0.002).
