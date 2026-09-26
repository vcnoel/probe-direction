# Research intuitions

These are accepted design intuitions, not empirical findings. Each intuition earns its place by naming a simpler alternative and a test that can falsify it.

## 1. Fixed wrappers can create low-rank nuisance structure

A template repeated around many tasks can add a shared shift to task-token activations. Because the wrapper is fixed while content varies, that shift may be low rank across examples. A probe can decode the wrapper without representing evaluation, recognition, or behavioral intent.

**Simplest alternative:** the shared direction is a generic consequence of token position, length, punctuation, or chat templating.

**Falsification test:** cross independent template families and match token position and task span. Fit all dimensionality reduction inside grouped training folds, then test on held-out templates and source tasks. Compare the target contrast with non-evaluative wrapper contrasts. The evaluation-specific account fails if the same low-rank direction or selectivity appears for the nuisance controls.

## 2. Prompt-only activations measure cue sensitivity, not stochastic recognition

A single forward pass over a framed prompt can show that activations vary with E. It cannot establish that the model recognized the situation, because R needs an independently defined recognition label or endpoint. It also cannot estimate stochastic variation in R without repeated observations under controlled sampling.

**Simplest alternative:** activations encode lexical or positional properties of the cue while the model’s recognition judgment is unchanged or undefined.

**Falsification test:** collect repeated, blinded recognition judgments from a preregistered gold or validated-judge endpoint, with held-out cue templates. Test whether an E-derived representation predicts R beyond prompt-surface controls. If it predicts E but not held-out R, call it cue sensitivity only.

## 3. Attention-graph summaries need value and output controls

Attention weights define routing coefficients, not the information transmitted. A graph statistic can change while value vectors or output projections cancel the change; conversely, transmitted content can change while a coarse attention graph remains stable.

**Simplest alternative:** any apparent graph effect is explained by residual/value statistics, or by head aggregation that discards signed output effects.

**Falsification test:** compare graph-only features with fixed-reference-signal and joint graph-and-signal modes; add value-weighted and attention-output baselines where the runtime exposes them. Patch or ablate candidate edges while holding appropriate signals fixed. Reject a graph-routing interpretation if graph summaries add no held-out information and interventions on the corresponding transmitted values do not move the endpoint.

## 4. Recognition and behavioral propensity may be weakly coupled and training dependent

R and P are distinct variables. A model can classify an evaluation cue while behaving identically, or its behavior can change through a route not captured by the recognition endpoint. Their coupling may depend on post-training objectives, system prompts, task domain, and decoding policy.

**Simplest alternative:** E affects R and P through separate pathways, or an observed R–P association is induced by task difficulty or template family.

**Falsification test:** estimate R and P on disjoint held-out tasks with repeated samples, then intervene on a representation selected using training data. Require endpoint-specific, dose-aware changes relative to random, placebo, and fluency controls. The coupling account fails if R moves without P, P moves without R, or the association disappears under grouped controls. Even a successful intervention is local to the tested training lineage and endpoint.

## 5. Every conjecture must travel with alternatives

A preferred mechanism is not an explanation until it outperforms explicit competitors. Candidate accounts include lexical cue decoding, absolute position, wrapper length, tokenization, source-task difficulty, model-family identity, residual norm, generic formality, attention routing, and value/output transformation.

**Simplest alternative:** the least structured nuisance model that predicts the same observable.

**Falsification test:** register the preferred conjecture, its alternatives, discriminating contrasts, and stop rule before confirmation data are opened. Run complete-pipeline permutations and negative controls. Retire or narrow the conjecture when an alternative matches its out-of-group prediction or when the discriminating test misses its sign-of-life gate.

## Working discipline

Use “E is assigned,” “R is measured,” and “P is estimated.” Do not write that a model “believes” it is evaluated unless a validated operational definition of R supports that wording. Decodability is correlational; mediation or controlled intervention is needed for a causal claim, and neither establishes intent or deception.
