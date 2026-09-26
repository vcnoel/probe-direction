# Theory index

This index distinguishes proved operator identities from empirical hypotheses. The proofs and qualifications are in [mathematical notes](mathematical-notes.md); planned tests are in the [future experiment protocol](future-experiment-protocol.md).

## Established mathematical facts

1. **Fixed combinatorial DC invariance.** For fixed symmetric W, combinatorial Dirichlet energy is invariant to adding the same hidden vector at every token.
2. **Degree-weighted normalized null mode.** The symmetric-normalized Laplacian annihilates `D^{1/2}1`, not generally the ordinary constant vector.
3. **Lifted Frobenius accounting.** Ordinary mean energy contributes a token-count factor when lifted into token space; normalized-null accounting instead uses total degree.
4. **Orthonormal spectral accounting.** A real symmetric operator supports Euclidean Parseval and frequency-energy identities.
5. **Categorical random-walk rejection.** Random-walk right eigenvectors are not generally a Euclidean-orthonormal basis, so basis-dependent diagnostics cannot use them as if they were.

These are algebraic statements. They do not require model inference and do not imply an empirical effect.

## Empirical hypotheses, not consequences of the algebra

- fixed wrappers produce low-rank nuisance shifts;
- nuisance shifts occupy lower graph frequencies than evaluation-specific shifts;
- graph structure adds information beyond residual/value summaries;
- an E-sensitive representation predicts independently measured R;
- R mediates an effect of E on P;
- recognition–propensity coupling changes with post-training lineage;
- selective effects scale with model size or transfer across architectures.

Each requires alternatives, disjoint data, controls, and a confirmation test. A failed hypothesis does not invalidate the operator identities; a true identity does not rescue a failed mechanism.

## Comparison-mode index

- **Graph-only:** compare W, L, or graph spectra; make graph-structure claims only.
- **Fixed-reference-signal:** freeze one side to isolate graph or signal change relative to a declared reference.
- **Joint graph-and-signal:** evaluate each signal on its own graph; attribute the result only to the coupled pipeline unless controls separate the sources.

## Claim vocabulary

Use **definition** for E/R/P and protocol constructs; **theorem** only for proved mathematical statements with assumptions; **candidate observation** for an unvalidated measurable pattern; **validated observation** only with a manifest-backed confirmation result; **conjecture** for a proposed explanation; and **causal claim** only for the specific intervention estimand supported by controls. Avoid “belief,” “awareness,” “mechanism,” and “law” when the endpoint establishes only cue decodability.
