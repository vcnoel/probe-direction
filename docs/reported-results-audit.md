# Reported-results audit

## Verdict

The registered source checkout at `4bfd36c39494b5425276de0b893aea969503a49b` contains no tracked result CSV, array, checkpoint, figure, prediction file, or results manifest. Every historical empirical number in the README, research notes, repo-local skills, and former manuscript is therefore **reported-unverified**. None is permitted as an asserted finding in the rebuilt README or paper.

The machine-readable source of truth is [`paper/claim_ledger.json`](../paper/claim_ledger.json). The accepted statuses are `validated`, `reported-unverified`, `falsified`, `conjecture`, and `planned`. At this revision, no empirical claim is `validated`.

## Claim inventory

| Claim family | Former sources | Required evidence | Status | Mathematical validity | Permitted wording |
|---|---|---|---|---|---|
| Spectral-probe separability | README, manuscript | Group-disjoint predictions, split manifest, config and checksummed summary | reported-unverified | Basis-dependent features from the legacy random-walk path are invalid | Historical only; no result is asserted |
| Length and position controls | README, manuscript, experiment design | Rendered prompts, tokenizer revision, span hashes and negative controls | reported-unverified | A fixed wrapper can remain a template shortcut | The future protocol tests all shortcuts |
| Selectivity | README, manuscript, theory notes | Out-of-fold predictions for both contrasts and full-pipeline permutations | reported-unverified | Signed raw-AUC subtraction is orientation-sensitive | Use a paired orientation-invariant interaction |
| Fiedler stability | README | Symmetric eigenpairs, eigengap audit and permutations | reported-unverified | Invalid on the non-symmetric random-walk basis | Untested after correction |
| Cross-cue transfer | README, manuscript | Fully disjoint rows, source tasks, template families and negatives | reported-unverified | Legacy overlap does not identify cue-general recognition | Planned disjoint transfer only |
| Cross-model transfer | Manuscript, law notes | Family-held-out manifest and aligned or dimension-agnostic features | reported-unverified | Width mismatch does not make transfer impossible | Planned confirmation analysis |
| Scale-dependent crystallization | README, manuscript, law and theory notes | Multi-family hierarchical estimate with uncertainty | conjecture | Evidence was insufficient for law language | Candidate pattern with alternatives |
| Frequency-ordering mechanism | README, manuscript, theory notes | A corrected retest would require symmetric fixed-reference decompositions | falsified | The algebra never entailed the empirical ordering | The proposed mechanism was unsupported |
| Verbalized recognition | README, manuscript | Independent stochastic recognition labels and endpoint validation | reported-unverified | Prompt classification measures cue sensitivity, not belief | Recognition remains a separate variable |
| Fixed-graph DC proposition | Theory notes, manuscript | Proof assumptions and synthetic invariant test | planned | Valid only for a fixed combinatorial graph | A scoped proposition, not an empirical result |

## Critical mathematical correction

The former primary cell used the random-walk operator `I - D^-1 W` as if it supplied a Euclidean-orthonormal Fourier basis. On an irregular undirected graph this operator is non-symmetric. Passing it to a symmetric eigensolver does not recover its eigenvectors and does not justify Parseval energy accounting, spectral entropy, high-frequency energy ratio, spectral-mass Gini, or Fiedler-vector claims. The corrected package uses the symmetric normalized Laplacian by default and rejects basis analysis of a non-symmetric operator.

## Publication rule

A quantitative empirical statement may enter an asserted README or manuscript section only when its claim identifier appears in a results manifest that conforms to `paper/results_manifest.schema.json`, all referenced files exist, every checksum matches, and the claim uses the metric and split recorded in that manifest. Historical prose is not an artifact.
