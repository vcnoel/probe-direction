# Candidate observations and conjectures

No empirical law is established here. No validated empirical artifacts are bundled. The items below are candidates to test under the [future experiment protocol](future-experiment-protocol.md), and all law language is prohibited until the hierarchical evidence gate is met.

## Candidate observation: wrapper-associated low rank

Fixed environment wrappers may induce a shared, low-rank shift across task-token representations.

**Alternatives:** absolute position, token count, template family, tokenizer boundary effects, residual norm, or generic formality.

**Required evidence:** grouped nested estimates on held-out source tasks and template families, matched nuisance contrasts, complete-pipeline permutations, and replication across task and model families. A low effective rank alone does not identify evaluation recognition.

## Candidate observation: selective cue representation

A representation family may discriminate a target E contrast more than a matched non-target wrapper contrast.

**Alternatives:** unequal contrast difficulty, leakage through fitting, condition valence, or a nuisance omitted from the negative control.

**Required evidence:** orientation-invariant paired selectivity with both component performances reported, disjoint confirmation groups, and a permutation procedure that repeats feature selection and fitting. This supports selective cue sensitivity, not R, P, belief, or intent.

## Conjecture: recognition is separable from cue sensitivity

Some internal representation may predict independently measured R beyond surface features of E.

**Alternatives:** the endpoint restates prompt wording; judge leakage; task difficulty; or R is not stable under repeated sampling.

**Falsifier:** held-out templates and tasks retain E decodability but lose prediction of a gold or validated-judge R endpoint after surface controls.

## Conjecture: recognition–propensity coupling depends on training lineage

The relationship between R and P may vary with post-training data, objectives, system prompts, and decoding policy rather than obeying a universal coupling.

**Alternatives:** apparent heterogeneity is sampling noise or endpoint non-equivalence; apparent coupling is confounded by task family.

**Falsifier:** preregistered hierarchical estimates show stable coupling across disjoint lineages, tasks, and endpoints, with low heterogeneity and successful leave-one-family-out checks. Until then, report family-specific estimates.

## Conjecture: graph routing contributes beyond residual summaries

Attention-derived graph structure may add held-out information beyond ordinary DC/AC and residual-moment controls.

**Alternatives:** values and output projections carry the effect; graph summaries merely correlate with residual norm or template geometry.

**Falsifier:** graph-only and fixed-reference-signal features add no selective information, or targeted value/output interventions fail while generic residual interventions reproduce the endpoint change.

## Candidate scaling relationship

Selective E, R, or P effects may change with model scale.

**Alternatives:** model family, architecture, tokenizer, data mixture, post-training lineage, or endpoint compatibility.

**Required evidence:** multiple sizes nested within multiple independent families, multiple task families, a hierarchical scale estimate, family heterogeneity, and leave-one-family-out sensitivity. A monotone list of checkpoints is not a scaling law.

## Promotion rules

A candidate may be promoted to a validated observation only when its claim-ledger entry points to a complete checksummed result manifest and the frozen confirmation test passes. “Conjecture” remains appropriate for a mechanism consistent with an observation but not causally distinguished from its alternatives. “Law” requires replicated quantitative regularity over prespecified domains and remains defeasible; it must never be inferred from a single family or exploratory sweep.
