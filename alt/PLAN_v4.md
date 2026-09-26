# Plan to 0.99: v20 = v10 skeleton + new retrieval channel + France self-training + per-country decisions

## Context
- Friends' best official score is 0.982; v10 scores 0.961 (fold 4: 0.9761). Leaderboard top is 0.9907, and the whole top 50 is ≥ 0.9867.
- **Deadline: 27 Sep 21:00 IST = 15:30 UTC** (aicompetition.dev / Unstop). From 26 Sep ~18:00 UTC that leaves ~21 h.
- This container: 4 CPU, 15 GB RAM, no GPU. It reproduces v10 exactly (fold 4 0.97614; test file identical row for row).
- Research done:
  - About 30 public GitHub repos. None publishes a method above ~0.98. The best claim is a simple 4-channel blocking + 14-feature LightGBM ("F0.5 > 0.98", AvinashMalladi).
  - Reddit could not be fetched from this environment (blocked). The linked threads are about submission limits and general competition state.
  - Research: cross-encoders are the strongest matchers (arXiv 2607.24688); self-training/pseudo-labelling is the standard tool for an unlabelled target domain (GPL 2112.07577); "JEV" = dense + BM25 retrieval fused with Reciprocal Rank Fusion (RRF).

## Where the points are (measured in this session)
| Gap | Evidence | LB value |
|---|---|---|
| **France ≈ 0.90** vs US 0.9755 / India 0.9674 at test density | stress-v2 on both countries + leaderboard 0.961 | ≈ 0.011 |
| **True pairs never retrieved** (largest local loss: US 0.010, India 0.0135) | parity loss table; blank-address 17% missed, websites 7% | ≈ 0.008 |
| Found but rejected / false matches | loss table | ≈ 0.005 |
| Ties (chain names, blank address) | chain inspection | ≈ 0.005, not fixable |
| Strictness per country (US 1.0, India 0.45) | stress-v2 | +0.0005 |
| Ownership layer | fold 4 | +0.0003 |

0.99 needs France ≥ 0.97 **and** retrieval losses more than halved **and** the matcher improved. No single lever is enough.

## Work plan (this container; the friends' machine runs the cross-encoder in parallel)

**Measured before planning (20k fold-4 S1 sample, 3.06% of true pairs missed by v10):**
- A char-3gram name channel (top-30, cos ≥ 0.3) recovers **31% of US misses, 9% of India misses**.
- The remaining US misses are 27% blank-address and 16% website names; India's are mostly Indian-script names.
- The channel adds **23 new pairs per S1**, so the LB ceiling gain is ≈ +0.002. W1 is therefore **second priority**. Run it only after W2 passes, and on the friends' machine if possible (their cross-encoder can score the extra pairs).

**W2 is the main lever (France).** Its validation proxy is **LOCO**:
- Train stage 2 on US-only dev rows plus **India pseudo-labels** (India labels hidden), then score India fold 4 with real labels.
- Compare against US-only training without pseudo-labels.
- If self-training does not raise India fold-4 in LOCO, it is not applied to France.
- Cost: 2 stage-2 fits here (~1 h each). Needs `train/feat_d/*` from HF (5.4 GB).

**W1. New retrieval channel "c3" (JEV-style fusion), then retrain v10p end to end.** Files: `alt/v10p/src/ranker.py`, `config.py`, a new `alt/v10p/src/knn_channel.py`.
- Per country, char 3-gram TF-IDF on `concat`: S1→target top-30 with cosine ≥ 0.3 (`sparse_dot_topn`).
- Also target→S1 top-5 for blank-address and domain targets (reverse retrieval).
- Pairs join the ranker union with new features `c3_cos`, `c3_rank`, `c3_rev`. Missing values = 0 for pairs from other channels.
- Ordinal-street fix: "66th ln" stays a street key.
- Retrain in order: rank-train, rank (train + test), features c, train-b, score-b test, stack features, train-c, test decisions.
- Budget: ~11 h here (score-b is the slow step: about 2.3 h test + 2.5 h train). Use `BER_RK_CHUNK` and run one job at a time.
- Gate: fold-4 recall/oracle and fold-4 macro above 0.9761.

**W2. France self-training (transductive, test labels never used).**
- From the test run: take France pairs with p ≥ 0.98 whose house number agrees with the S1 as positives, and pairs with p ≤ 0.02 as negatives.
- Add them to stage-2 training (`stack.train_and_tune` sample) with weight 0.5, and add a country indicator feature, so stage 2 learns French legal and word patterns.
- Also the French normalisation fixes: `cie`→co, `groupe`→group (the `normalize.LEGAL` map), and `bis`/`ter` as number suffixes.
- Gate: France label-free proxies (number agreement, matches per S1, contested claims) must not get worse; US/India on fold 4 unchanged.
- The final check is the leaderboard: France-only swap upload.

**W3. Decisions.** Per-country shift (US 1.0, India 0.45, France 0.6) plus the ownership layer (`alt/src/ownership.py`) via `alt/src/make_upload.py` (already built and validated).

**W4. Hand-off to the friends' pipeline (their 0.982 model).**
- Give them the c3 channel code, the French self-training recipe and the decision layer, so they can apply them to their cross-encoder model.
- If they share test pair scores, W3 is applied within 30 minutes.

## Honest score math
- Friends' 0.982 → 0.99 needs +0.008.
- Levers available:
  - France from ≈0.90 to ≈0.96: +0.009 (only if self-training passes LOCO);
  - decisions: +0.001;
  - c3 channel: +0.002.
- The sum reaches ≈0.99 only if the France lever works almost fully. I found no public or research method that guarantees it.
- Most likely outcome: 0.985–0.988.

## Timeline (UTC)
- 18:30–21:30: W2 LOCO test (download feat_d, 2 stage-2 fits, India fold-4 comparison).
- 21:30–01:00: if LOCO passes, stage 2 with France pseudo-labels + country feature, France test rescoring, France-only file.
- 01:00–02:00: W3 per-country decisions + validation.
- 02:00–13:00: W1 c3 channel if time allows, else hand it to the friends.
- Uploads before 15:30. Preferred base: the friends' 0.982 model with W2/W3 applied (needs their test pair scores and their stage-2 code).

## Verification
- Fold-4 macro and recall/oracle per country.
- The stress-v2 US/India run gives expected leaderboard scores for US/India.
- France label-free proxies.
- `validate_submission.py --check-ids` PASS, 1,732,544 rows, matches ⊆ candidates.
