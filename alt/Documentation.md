# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [team]
**Team Members:** [members]
**Submission Date:** 27 Sep 2026

## 1. Executive Summary
A multi-pass blocking union is cut by a learned ranker to 50 candidates per Source-1 entity. A
cross-fitted two-stage LightGBM matcher follows (stage 1 on pair similarity; stage 2 on target
competition, list context and decoy-detection features), then an exact expected-F0.5 set decision
under a one-owner constraint. On top of that: a target-ownership model, a separate decision
strictness per country (tuned on a test-density simulation), and self-training of stage 2 on the
unseen country (France) using only the model's own confident test predictions.

## 2. Methodology
### 2.1 Problem analysis
- Names are shared by chains (many S1 branches with one name); the address decides the branch.
- Decoys copy an S1's name, swap its legal form and shift its house number.
- Blank-address records are almost always owned.
- About 18% of Indian true pairs have target names in Indian scripts.
- Test differs from train in three ways:
  - it has about 19% of its owners hidden (5.8 records per S1 vs 4.7);
  - test US has half the S1s of train US;
  - it adds France, which has no training labels.
### 2.2 Strategy
Approach: blocking + learned ranker + stacked GBDT + decision theory. Core ideas: decoy-aware
stacking, exact expected-F0.5 decoding, test-density stress simulation, unsupervised country
adaptation.

## 3. Candidate Generation (Blocking)
- Passes, all within a country:
  - exact order-insensitive name key;
  - rare name tokens (IDF);
  - (house number, street-word prefix);
  - rare address words;
  - concatenated-name / website prefix;
  - cross-script consonant skeletons (IAST transliteration).
- A LightGBM ranker (blocking-graph features + token Jaccards) keeps the top 50 per S1.
- Candidate pairs: 81.5M on test (47/S1). Union recall ≈ 0.975 on the held-out fold (oracle F0.5 0.991).

## 4. Matching Model
- **Stage 1:** 44 pair features (token/set overlaps, rapidfuzz ratios, Jaro-Winkler, legal-form
  agree/conflict, house-number equal/near/suffix, name frequency, ranker score, skeleton similarity);
  2 cross-fitted LightGBM models with isotonic calibration.
- **Stage 2:** 64 features. Target competition, list context, sibling similarity, legal-swap and
  number-offset decoy features, consensus, word-frequency and per-source features.
- **Decision:**
  - a target-ownership LightGBM re-scores claims shared by several S1s;
  - exact expected-F0.5 prefix selection (Poisson-binomial);
  - odds shift per country (US 1.0, India 0.45, France 0.6), chosen on a test-density simulation.
- **France (unseen):** stage 2 refit with pseudo-labels from our own confident test predictions
  (p ≥ 0.98 with agreeing house number / p ≤ 0.02) and a domain flag. Validated leave-one-country-out
  (US labels only, India unseen): +0.0039 F0.5 on India's held-out fold.

## 5. Results & Error Analysis
- Held-out fold (S1-disjoint) macro F0.5: 0.9768 (US 0.9802, India 0.9719).
- Loss is dominated by true pairs never retrieved (blank-address and website-only names), then by
  blank-address ties between chain branches (not decidable), then by rejected noisy copies.
- False matches are mostly same-street decoys with the house number kept.

## 6. Conclusion
Precision-first stacking with decoy features and exact F0.5 decoding carries most of the score.
Measuring the test conditions (density, hidden owners, unseen country) mattered more than further
local tuning.

## Appendix A. Code
See `code/business_entity_resolution/README.md`. `src/` holds the pipeline; `src/alt/` holds the
decision layer, the stress simulation, the France self-training and the output builders.
