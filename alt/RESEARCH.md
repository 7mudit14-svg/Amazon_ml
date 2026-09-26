# Research notes: public solutions and relevant ER research (26 Sep)

## Public solutions for this challenge (GitHub/Kaggle)
- Most public repos use TF-IDF char n-gram blocking (top 15-20) + LightGBM/HistGBM on 15-30 string
  features + a threshold sweep. Reported validation scores: 0.912 (ankitpaul6201), none on LB.
- Akash-bardia/amazon-ml-challenge-2026 reports 0.9761, the same number as our v10.
- No public write-up of a >0.98 leaderboard method was found. The top 50 (>= 0.9867) have not
  published.

## Research that applies
- "Beyond Scale and Generation: Understanding LM-based Entity Matching" (arXiv 2607.24688, 2026):
  cross-encoders are consistently the strongest matchers. Generative matchers help mainly under
  distribution shift (our France case). Larger models rely more on shortcuts, so bigger is not
  automatically better.
- Self-training / pseudo-labeling is the standard unsupervised domain-adaptation tool for an
  unlabeled target domain (GPL arXiv 2112.07577; source-free UDA arXiv 2504.11992).
- "JEV" (Joint Embedding Vectors) = dense embedding + BM25 fusion with Reciprocal Rank Fusion
  (a retrieval technique). The equivalent here is adding a dense or char-n-gram retrieval channel,
  fused with the lexical union.

## What this means for us (measured in this session)
| Gap | Size (LB) | Lever |
|---|---|---|
| France ~0.90 vs ~0.97 US/India | ~0.011 | pseudo-label self-training on France; cross-encoder |
| Blocking misses (17% of blank-address, 7% of website targets) | ~0.004 | char-3gram/dense channel + RRF fusion |
| Per-country strictness (US 1.0, India 0.45 at test density) | +0.0005-0.0012 | decision only |
| Ownership layer | +0.0003 | decision only |
| Ties (chain names, blank address) | ~0.005 | not fixable |
