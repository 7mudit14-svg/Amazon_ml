# Submission B (best validated file)

Contents: v10 pipeline, then:
- the target-ownership decision layer;
- a separate strictness per country (US 1.0, India 0.45, France 0.6);
- self-training of stage 2 on France (pseudo-labels from our own confident test predictions; no external data, no test labels).

- `matching_results.tsv.gz`: leaderboard file. Run `gunzip matching_results.tsv.gz` and upload the `.tsv`.
- `team_submission.zip.part00..NN`: round-2 package (output/, code/, Documentation_template.md), split for GitHub.
  Rebuild it with `cat team_submission.zip.part* > team_submission.zip` (Windows: `copy /b team_submission.zip.part00+team_submission.zip.part01+... team_submission.zip`),
  then check it against `SHA256SUMS`.

Validation: `utils/validate_submission.py --check-ids` PASS; 1,732,544 rows; matches ⊆ candidates.
