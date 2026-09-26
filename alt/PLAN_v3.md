# Implementation plan: v10 → maximum leaderboard score by 27 Sep morning (rule-safe)

## Context
v10: held-out fold 4 = 0.9761, leaderboard = 0.961. Top 50 ≥ 0.9867 (#1 0.99068). Now: 26 Sep 09:34 UTC
(~15:00 IST), about 16 h left. This container: 4 CPU, 15 GB RAM, no GPU. The teammates' machine runs
V13–V15 plus the U2 transformer. We reviewed 4 external LLM critiques plus the teammates' notes; this plan
keeps what the evidence supports. Rules: only the provided train/test data, test used label-free only,
LightGBM (MIT) and a transformer that must be MIT/Apache and ≤ 8B, no ID/row-order artifacts.

## Evidence (measured this session)
| Finding | Number | Source |
|---|---|---|
| Local → leaderboard offset, same for v5/v7/v10 | 0.015 | uploads |
| Recall stops growing: recall@40/50/60 | 0.97505 / 0.97527 / 0.97539; oracle@50 0.9914 | HF `report_cand_b.json` |
| Retrieval misses at K=50 | blank-address targets 17.4% (US), 9.5% (India); website names ~7%; ordinal streets 3.3% vs 2.1% | HF `cand_b` sample |
| **Density path into the models** | Ranker gain: `n_union` 19.5%, plus block sizes 3.6%. Stage-1 gain: `rk_p` 72.7% + `t_best_gap` 11.1%. Stage 2 is driven by `p1`/`c_gap`. | HF model files |
| Test pool vs train | US S1 ×0.50, targets ×0.62; India ×0.92/×1.14; France 260k/1.43M | raw data |
| Leakage shortcuts | none: Spearman(row order) ≈ 0.001, Spearman(IDs) ≈ 0.0001 | raw data |
| Chains / ties | one name, many branches; the blank-address "LLC" record belongs to the **LP** branch | raw data |

## Verdict on the external suggestions
| Suggestion | Decision |
|---|---|
| Parity check first; check feature distributions before the long stress run; clip the scale; re-sweep the shift | **Adopt** (steps 1, 3, 4) |
| Density/count shift is the offset | **Adopt as main hypothesis.** It now has a mechanism (`n_union` → `rk_p` → p1). Test it with a counterfactual (step 3) rather than the full 5 h stress rerun. |
| Per-target ownership softmax with a NULL owner | **Adopt** (step 2). It replaces `single_owner`'s independence assumption. |
| Reverse retrieval (target → S1) and extra channels for blank/website/ordinal records | **Adopt** as recall channels (step 6), then hand to the teammates' retrain |
| Cross-encoder / U2 | **Adopt, on the teammates' machine** (no GPU here). Its score becomes a stage-1 feature. |
| Dense bi-encoder retrieval; full new V16 architecture | **Reject for this deadline.** Training on 10M records with no GPU here, and no time to reach parity. |
| Singleton classifier / dynamic thresholds / hard negatives | Already in v10: exact expected-F0.5 per S1 includes the empty set; candidates are hard negatives |
| Exploit ID or row-order artifacts; look up forums | **Reject.** Leakage/rules. Checked anyway: no signal. |
| "2024 winner scored 0.865" | Irrelevant (different task) |
| "Skip France" | **Partly.** A France-only rerun is cheap because blocking and counts are per country; ship it only behind label-free gates. |

## Steps (this container; code under `alt/src/`, results to `alt/experiments.csv`, commit to `claude/amazing-bohr-64t0ac`)

**0. Fetch (~30 min).** `alt/src/fetch_v10.py`:
- downloads from `hf.co/Galectic/v10` into `work_v10/`: `models/`, `train/{s1,tgt,s1_sk,tgt_sk,gt,folds}.parquet`, `scores_b`, `scores_c`, `cand_b/*`;
- plus `feat_c/*`, filtered to fold 4 while streaming.
- Keep `work_v10/` out of git.

**1. Parity and loss table (~1 h).** `alt/src/parity.py`:
- Use v10's own code (`sys.path` → `v10/.../src`): `matcher_b.apply_policy` with `meta["policy"]` from
  `models/matcher_c/meta.json` on `scores_c`, and `score.per_entity`.
- Fold 4 must give **0.9761** (0.9766 without the shift).
- Loss by country × {not retrieved, retrieved but rejected, tie, false match}, plus the join-only share.

**2. Decision layer (~1.5 h).** `alt/src/ownership.py`:
- For targets with ≥2 claimants, set P(owner = i) = softmax over {claimants' logit(p), NULL}.
- The NULL logit is a function of the claimant count and the country's hidden-owner rate.
- Isotonic calibration on dev folds 0–3 only.
- Replaces `decide.single_owner` inside `apply_policy`; `dta_select` is unchanged.
- Re-sweep the shift on dev.
- Gate: fold-4 macro F0.5 ≥ parity + 0.0005, per country.
- Works on any model's `(s1, t, p)`, so it can be applied to v10 **and** to the teammates' V1x scores.

**3. Density counterfactual (~2.5 h).** `alt/src/density_cf.py`, reusing `stress.injection_stack`,
`ranker.build_index` / `union_chunk` / `_predict`, `stack._stack_part` / `_target_context` and
`stress._decide`:
- For a 60k fold-4 US sample, rebuild the ranker union.
- Shift the density features to test scale: `n_union` and block sizes `e_bs`/`addr_bs`/`cc_bs`/`na_bs`
  by log(0.62); `s_name_rep` by log(0.5); `s_name_tdf` and `nw_*` by log(0.62).
- Rescore ranker → stage 1 → stage 2 → decide, and report the F0.5 drop.
- Also check that `count_scale` recovers the drop.
- Gate for the fix: drop ≥ 0.003 **and** ≥ 70% recovered.

**4. Test inference with v10 in this container (~4 h compute, runs during steps 2–3).**
- `v10/.../run.py prepare/translit/rank/features --variant c/score-b/stack-features/submit-c
  --split test --work-dir work_test` with the HF models.
- Parity: output equals `v10/matching_results.tsv`.
- Log label-free test vs train-fold-4 distributions of `n_union`, block sizes, top-1 `rk_p` and
  final p per country.

**5. Patched test run (~2–3 h from the rank step).**
- `count_scale` per country in v10's `config.py`, applied in `ranker.union_chunk` (`n_union`),
  `_with_block_size` (sizes), `record_stats` (`s_name_rep`/`s_name_tdf`), `features.name_frequencies`
  and `stack.token_df`.
- Clip to [0.5, 3]; France uses the train-US scale, clipped.
- Then apply the step-2 decision layer and validate with
  `validate_submission.py --check-ids` → PASS, 1,732,544 rows.

**6. Recall channels and France (CPU time left over; hand-off).**
- `alt/src/reverse_retrieval.py`: character 3-gram TF-IDF target→S1 k-nearest-neighbours
  (`sparse_dot_topn`) for blank-address and website targets, plus an ordinal-street address key.
  Report the fold-4 recall/oracle gain against `cand_b`. Needs retraining, so it goes to the
  teammates' V1x.
- France patch in `normalize.py` (`cie`/`ets`/`fils`/`groupe`/`(france)`/`bis`/`ter`, department↔region
  alias). Validate with a France-only test run (~1 h; exact because everything is per country).
  Label-free gates: number agreement ≥ 99%, matches per S1 within ±5%.

## Uploads (2)
- **#1 (~01:00 UTC, before the deadline):** base = teammates' best V1x if ready, else v10; plus the
  step-2 decision layer; plus `count_scale` only if step 3 passed.
- **#2:** #1 plus the France patch (France rows swapped, US/India byte-identical), or the teammates'
  merge with U2 and the recall channels.
- A safe fallback file is kept ready at all times.

## Expected
- Step 2: +0.001–0.003.
- Step 5: up to +0.010 if the density hypothesis holds (step 3 tells us before any upload).
- Teammates' U2 + V13 recall: +0.005–0.008.
- Realistic 0.975–0.985. Top 50 (≥ 0.987) needs the offset to be mostly density and all levers to land.

## Verification
- Parity: fold 4 = 0.9761 exactly (step 1); the test rerun equals v10's file (step 4).
- Every change: fold-4 Δ per country, plus the density counterfactual; each is logged.
- Before any upload: validator `--check-ids` PASS, row count, matches ⊆ candidates, and byte-identical
  checks for partial swaps.
