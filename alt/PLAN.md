# Challenger plan v2: what to change and why (revised with measurements)

Status 26 Sep: dataset downloaded, v10 code and output read, four label-free measurements run
(scripts in `alt/diagnostics/`). They change the original plan in several places.

## 1. Measurements (test = no labels, only v10's uploaded output)

| # | Measurement | Train | Test | Meaning |
|---|---|---|---|---|
| M1 | S2+S3 records per S1 | US 4.67, India 4.68 | US 5.76, India 5.82, France 5.53 | Test has +1.1 records per S1 |
| M2 | S1 share by country | US 60%, India 40% | India 46.8%, US 38.3%, France 15.0% | France weight is exactly 0.15 |
| M3 | Same-name targets per S1 (exact name key) | US 2.08, India 1.92 | US 2.38, India 2.13, France 2.38 | Only +0.3 of the +1.1 extra records are same-name copies |
| M4 | Targets whose name matches no S1 | US 52%, India 63% | US 56%, India 67%, France 53% | The extra records are mostly **orphans** (their S1 was removed), not denser decoys |
| M5 | v10 test: predicted matches per S1 / empty rate | truth 3.46 / 5.6% | US 3.29/6.0%, India 3.23/6.2%, France 3.27/5.6% | v10 on France predicts normal counts: no recall collapse |
| M6 | House number shared by matched pairs (when both have one) | truth US 92.8%, India 97.2% | v10 US 92.7%, India 97.3%, **France 99.0%** | v10's France matches look as clean as US/India |

Implications:

1. **Test ≈ train with ~19% of S1s hidden** (4.67 / 5.76 = 0.81). This is exactly what v10's
   `stress.py injection_stack` simulates (hides 20%/30%), and it chose `decision_shift = 0.6`.
   So plan steps **E1 and E4 are already done by v10**; they only add value split by country.
2. **The "France is the gap" hypothesis is weak.** Under the plan's model LB = 0.85·(L−s) + 0.15·F
   with s ≈ 0.002 (v10's own stress drop), France would have to be ≈ 0.887. But v10's France
   predictions have normal counts (M5) and 99% house-number agreement (M6), and moving France
   strictness from 0.08 to 4.0 barely moved LB (0.961 → 0.959). A France F of 0.887 would need
   ~0.3 wrong matches per S1, and those would show up in M6. They don't.
3. So the 0.015 gap is **not localised yet**. Remaining candidates: (a) France errors that keep the
   house number (same-number decoys, possible with the French generator: "Vent Parents (France)"
   vs "vent parents (france) sarl" both at N°6 R. Gustave Delory), (b) a larger orphan penalty than
   the stress test shows, (c) US/India drift on test.

## 2. Issues with the original plan

| Issue | Evidence | Fix |
|---|---|---|
| Rewriting a separate pipeline to reach parity is the riskiest step and burns the whole budget | v10 is 3.5k lines and needed 10 versions to reach 0.976; parity (±0.005) is required before any Alt component is measurable | **Fork v10** under `alt/`, add components as flags, compare against v10 on identical folds |
| Folds differ | `alt/src/common.py` uses its own multiplicative hash; v10 uses stratified `polars hash(seed=2026)` | Use v10's `splits.make_folds` with polars 1.44.2 (installed) |
| E1 and E4 duplicate v10 | `stress.py`, README: "~41–48% ownerless", `decision_shift = 0.6` | Run v10's stress **per country** only |
| FR legal forms already in v10 | `normalize.LEGAL` has sarl/sas/sasu/sa/eurl/sci/snc/ei, dotted and parenthesized | FR work should target what is missing: `cie`, `ets`, `& fils/frères`, `groupe` (≠ `group`), `(france)`, `bis/ter`, `None` addresses, department↔region |
| E3 dictionary vs skeleton | v10 already recovers Indic candidates via skeletons (recall 0.909 at K=40 on Indic targets) | Keep, but measure on v10's error buckets first |
| Gate +0.015 locally is unreachable and not the right gate | headroom 0.024 | Gate on **the thing that explains the LB gap**, once it is found |
| The LB target 0.99 | needs error 0.039 → 0.010 on test (−75%) in ~1 day | Not realistic. Honest target: 0.965–0.975 |

## 3. The single most informative action: localise the gap with one upload

Upload v10 with **France rows emptied** (US/India byte-identical). France singletons score 1, all
other France S1s score 0, so

    F_France(v10) = (LB_v10 − LB_emptyFR) / 0.1498 + 0.056

- If F_France ≈ 0.88–0.90 → France is the gap: spend everything on the FR module (section 4, P1).
- If F_France ≈ 0.96–0.97 → France is fine: the gap is test-wide (orphans / drift): spend on P2/P3.

This costs 1 of 2 uploads but replaces a day of guessing. (Emptying US instead works the same way.)

## 4. Pipelines / components, ranked by expected LB gain per hour

| ID | Component | Targets | Expected LB Δ | Cost | Validation |
|---|---|---|---|---|---|
| P0 | Reproduce v10 here (fork), per-country + error-bucket report on fold 4 | parity | 0 | 3–5 h compute | sealed 0.976 ± 0.002 |
| P1 | **FR normalisation fixes** in v10 fork: `cie/ets/fils/frères/groupe/(france)` as legal/generic words, `bis/ter` numbers, literal `None`, department↔region alias mined label-free from co-occurrence of city → {dept, region} | France, only if M-upload says F_France < 0.93 | +0.005 … +0.010 | 2 h + rerun test | LOCO (train US → India) can't test FR vocab; use label-free France proxies (M5/M6 + city-consistency) |
| P2 | **Per-country vocab DF** for `vocab_features`: France name vocab is tiny (e.g. Vent/Entre/Amicale × Ecole/Club/Groupe/Holding), so v9's "new word DF" signal flips meaning; normalise DF by country vocabulary size | France decoy detection | +0.002 … +0.006 | 1 h | LOCO on India (train US) |
| P3 | **Orphan-aware stress re-tune**: rerun `stress --variant c --pool dev` at hide 19% per country, choose shift per country | test-wide | +0.001 … +0.003 | 1 h | sealed-fold stress |
| P4 | Ensemble v10 stage-2 with a second stage-2 trained on different seeds/feature subset (bagging) | all | +0.001 … +0.002 | 2 h | fold 4, both copies |
| P5 | Indic token dictionary mined from train pairs, added as a blocking key + feature | India Indic (18% of India pairs) | +0.001 … +0.003 | 3 h | fold 4 India cross-script bucket |
| — | Dense / cross-encoder / LLM judge | — | — | infeasible here (no GPU, 10M records, 1 day) | — |

Realistic ceiling with P0–P5: LB ≈ 0.966–0.975 if France is the gap, ≈ 0.964–0.968 if not.
0.99 would require a different class of signal (e.g. a learned cross-encoder), which is not
achievable before the deadline on this hardware.

## 5. Order of work

1. P0: reproduce v10 in this container (prepare → folds → rank → features c → train-b → stack L2),
   sealed-fold report per country and per error bucket. Keep every run in `alt/experiments.csv`.
2. Decide with the team on the France-empty diagnostic upload (section 3).
3. P1/P2 if France is the gap; otherwise P3 → P4 → P5.
4. Before any upload: `validate_submission.py --check-ids` PASS, 1,732,544 rows, matches ⊆ candidates,
   and (for partial swaps) untouched rows byte-identical to the base file.
