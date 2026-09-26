# Pre-registered analysis rules

Fixed on 2026-08-02, after the four Qwen2.5 checkpoints and before any new concept (T3) or any
new model family or generation (T4) is analysed. The four-Qwen data is the exploratory set and is
exhausted; everything below is applied unchanged to the confirmation sets, and any departure is
reported as a departure.

The reason for writing this down is specific. The interior-layer candidate set was identified by
sweeping four sets and reporting the one with the highest generalizability coefficient. That is
selection on the reported metric, which is the same error this paper criticises in best-of-N
probe selection, and it cannot be undone by a good post-hoc control. The control below is
supportive, but the rule only becomes a finding if it holds where it was not chosen.

## Statistic

**S1.** The primary statistic is `max |AUROC - 0.5|` over the candidate layer set. Peak AUROC is
not reported separately: folding each layer to `max(a, 1-a)` makes it identically this quantity
plus one half.

**S2.** The depth statistic is the OLS slope of the `|AUROC - 0.5|` curve against relative depth.
An argmax is reported only where the maximum is interior and has a stability interval; it is
never reported at a layer boundary.

## Candidate layer set

**L1. NOT VALIDATED at this design; NOT refuted.** The rule is: candidate set is the interior
band, relative depth 0.25 to 0.75. The confirmation test on the three models it was never fit to
cannot resolve the ordering, so no recommendation is made, and the reason is power rather than
failure.

On the held-out models interior gives E rho^2 = 0.073 against 0.081 for layers 1..L-1, an observed
difference of -0.008 whose bootstrap interval over templates and item blocks is
[-0.283, +0.244] -- spanning zero roughly thirty-fold. Those models also span log10 0.43 in
parameters against Qwen's 1.15, and between-model variance scales with the square of the log-span,
so a 7-fold smaller sigma^2_m is expected there before any candidate set enters; the observed
ratio is 5.0. Propagating that through the Qwen fit predicts E rho^2 of 0.09 to 0.13 for a rule
performing exactly as well as it does on Qwen, against 0.073 observed. Two of three subsets favour
interior (0.421 vs 0.273 on Qwen; 0.190 vs 0.123 on all seven); the third is the narrowest in
range and the least powered.

Consequence: all three candidate sets are reported side by side and none is recommended, because
the test that would license a recommendation has not been passed. This is a statement about the
design's power, and it is an argument for a wider parameter range rather than against the rule.

An earlier version of this file said the rule was WITHDRAWN and refuted by its own confirmation
test. That was wrong -- a point ordering between two coefficients estimated from three models was
treated as a result. The claim is corrected here rather than deleted.

**L2.** Justification and its limit. On the exploratory set the interior band gives
E rho^2 = 0.413 against 0.153 for layers 1..L-1 and 0.018 for all of `hidden_states`. A
cardinality explanation was tested by drawing 100 random subsets matched per model to the
interior band's size: their median E rho^2 is 0.113 and only 2% reach 0.413, so the advantage is
not attributable to having fewer candidates. This does not license the rule, it only removes one
competing explanation; the rule is licensed by holding on the confirmation sets.

**L3.** Full-range and trunk results continue to be reported alongside, because published work
uses them and the comparison is part of the finding.

## Pooling and preprocessing

**P1.** Activations are mean-pooled over tokens after dropping BOS and every token whose decoded
form strips to empty, matching the reference implementation.

**P2.** Newlines are LF. Prompts are never round-tripped through CSV.

**P3.** No truncation below the model's context length.

**P4.** Any reported number carries the convention that produced it. Numbers from different
conventions are never compared silently.

## Design and estimation

**D1.** Model is nested within family, not crossed with it. Components are sigma^2_family and
sigma^2_model(family), and the k-requirement is evaluated separately for within-family and
cross-family comparisons, which have different denominators.

**D2.** Components are estimated by REML rather than by ANOVA moment estimators. The moment
estimators are unstable near zero with few levels, require truncating negative estimates, and
cannot handle the unbalance that arrives with Gemma, which has no system role and therefore
cannot be crossed with the system-versus-user-only implementation level.

**D3.** Every component that REML returns at or near the boundary is flagged as unresolved rather
than reported as absent.

**D4.** The primary evidence that templates are not interchangeable measuring devices is the
ratio sigma^2_mt / sigma^2_mh, which needs no reference distribution. Kendall's W and its
equivalence test are reported as illustration, not as the primary.

**D5.** Claims are stated as claims about recoverability. A stable ranking may exist and be
swamped by template-induced measurement error; the evidence bears on whether it can be recovered
from a single-template design, not on whether it exists.

## Retraction discipline

**R1.** Before withdrawing a claim, compute whether the test that killed it could have detected
the effect. A point ordering, a non-significant difference, or a boundary estimate is not evidence
against a claim unless the design had the power and the range to resolve it. Five retractions in
this project were later found to be aimed at underpowered tests or at the wrong estimand, so the
check is now mandatory and its result is reported alongside any withdrawal.

**R2.** State what a failed test licenses. "Not validated" permits withholding a recommendation;
it does not permit asserting the opposite.

## Surviving prescriptions

**V1.** Report the slope of the discriminability curve against relative depth. Report an argmax
only where the maximum is interior and carries a stability interval, never at a layer boundary.

Grounded in reproducibility rather than variance, and therefore untouched by the retraction of the
sigma^2_m argument (R3) and by the layer-set question, which concerns a different facet. Across the
four models where the reference implementation could be reproduced, the slope reproduced on all
four with a maximum deviation of 0.0055, while the peak failed on one of four at 0.0191 because
two near-tied layers swapped order. This is the one recommendation in the project that has
survived every correction.

## Standing checks

**S1. Estimand match on model population.** Before comparing any of our Qwen-only measurements to
a published figure, check which model set that figure was fitted on. Four separate claims in this
project attached a Qwen-only measurement to a headline whose fit excluded Qwen. The published
scaling headline is the ten-model no-Qwen fit; nobody published a scaling law on the four Qwen
checkpoints we measure, so no result of ours can confirm or contradict one there.

**S2. Bracket claims are per-quantity.** A set of nulls that brackets on one statistic need not
bracket on another. Our three nulls bracket correctly on floor MEANS in three of four models but
not on the floor SLOPE, where the autocorrelation-matched null exceeds both others. State which
quantity a bracket is claimed for.

**S3. Report standardised excess.** Any comparison against a null is reported as
(statistic - null mean) / null SD, not as a raw difference, because null SD varies across models
and a difference conceals that. This is also the form a reader can apply without rerunning
anything.

## Nulls

**N1.** A null is an equivalence statement against a stated smallest effect of interest. Where no
such interval is available the finding is reported as unresolved at this design size, never as an
absence of effect.

## T3: second concept (refusal), pre-registered before the run

Fixed 2026-08-02, before any refusal activation is extracted.

**T3.1 Wrapper set is held fixed across concepts.** The same 18 framings are used, with the
contrast changed from eval-versus-deploy to harmful-versus-harmless instruction content. No
refusal-specific wrappers are authored, because that would confound concept with wrapper set and
destroy the comparison the experiment exists to make.

**T3.2 The two concepts do not load the wrapper facet identically, and refusal's version is the
cleaner one.** For evaluation awareness the wrapper IS the contrast -- a template is an
(eval, deploy) pair -- so wrapper variance and treatment are entangled by construction. For
refusal the contrast is content and the wrapper is context, held fixed across both arms, so only
the wrapper varies. This asymmetry is stated up front rather than left for a reader to find: the
refusal estimate isolates whether a direction depends on surrounding context, without the
manipulation moving underneath it.

**T3.3 Matched k.** Refusal has 18 wrapper levels where eval-awareness has 36 pairs. E rho^2 at
k = 1 is comparable across concepts because k = 1 is the estimand rather than the sample, but any
E rho^2 at k = all must be compared at a matched k. The cross-concept contrast is reported at
k = 1 and at a common k.

**T3.4 Item counts.** Directions are built from 48 harmful_train and 48 harmless_train
instructions per wrapper. Scoring uses 200 harmful_test and 200 harmless_test items, which both
avoids the 39-item val split and matches the 400-item count of the evaluation-awareness arm. The
item facet is 4 blocks of 100 as before.

**T3.5 Implementation facet gains a level.** The reference refusal pipeline selects jointly over
layer AND post-instruction token position, a larger selection surface than the
evaluation-awareness pipeline's layer-only search, and position is undocumented in the same way.
Position enters as an implementation level, which makes the cross-concept comparison a test of
whether the same facet structure appears under a larger selection surface rather than a repetition.

**T3.6 Causal endpoint.** The authors' own refusal_score -- log-odds of the refusal token at the
final position, one forward pass, no generation -- evaluated under directional ablation, per
wrapper. A generation subset with substring matching validates it and the correlation between the
two is reported.

**T3.7 Ablation transfer matrix.** A direction built under wrapper W is ablated and scored on
harmful items presented under wrapper W'. The diagonal is same-wrapper efficacy; the off-diagonal
is causal generalisation across wrappers, and their ratio is the causal analogue of
leave-one-framing-out. The diagonal is computed for every model; the full 18x18 for one or two.

**T3.8 Behavioural evaluation-awareness endpoint, at no marginal cost.** Six wrappers are eval
framings and six are deployment framings, so refusal rate on identical harmful instructions under
eval versus deployment wrappers is a behavioural evaluation-awareness measurement, computed from
forward passes already being run. A null result is reported as an equivalence statement against a
stated smallest effect of interest, per N1, and not as an absence.

**T3.9 What each outcome licenses.** Written down now so the interpretation is not chosen
afterwards.

- Refusal E rho^2 high and evaluation-awareness low: reliability is a per-concept empirical
  property, and concepts can be ranked by how well posed they are.
- Both low: the finding is about the instrument rather than the concept, and the paper's scope
  extends past this subfield.
- Decoding reliability low but causal reliability high: decoding statistics and causal relevance
  come apart, which bounds what a probe AUROC licenses.
- Decoding high but causal low: the direction is recoverable but not the mechanism, which bounds
  causal claims built on selected directions.

## Pipeline freeze, scoped

Declared 2026-08-02 and then immediately tested by a correction that moved a headline coefficient,
so the scope has to be stated rather than implied.

**F1. The freeze covers DATA COLLECTION.** Models, pooling conventions, prompt rendering, item
sets, and the extraction code. None of these has moved since the declaration and none will without
a new version number.

**F2. Model specification was NOT frozen by it, and is now closed for a stateable reason.** The
template facet turned out to be factorial, 6 eval x 6 deployment, and treating it as 36
exchangeable levels overstated the generalizability coefficient at full k. That is now corrected,
and the design is fully decomposed: template's 35 degrees of freedom partition exactly as
eval (5) + deployment (5) + their interaction (25), with no further nesting available to discover.
The two fits are related by identities that hold to machine precision, so the earlier and later
numbers are not rival specifications -- the earlier one was biased by a computable amount.

**F3. Anything found after F2 goes in a robustness section** and does not move a headline. A
bounded process is the point; an open-ended one would make the final numbers as suspect as the
intermediate ones.

## T3.10 The eval-arm asymmetry: pre-registered mechanism test

Fixed before any refusal statistic is computed.

The arms-crossed decomposition gives model x eval at 16.5% of variance against model x deployment
at 2.2%, a ratio of 7.5. The direction geometry gives eval-specific 41% against deployment-specific
31%, a ratio of 1.3. Something amplifies the geometric asymmetry roughly sixfold in the variance,
and we do not currently know what.

The candidate mechanism is coupling between the direction and the scoring set rather than a
property of the framings. The held-out positive class is testing-flavoured content, so it is
semantically nearer the evaluative framings; variation in the eval arm then moves the projection of
the positive class specifically, while variation in the deployment arm moves both classes together
and largely cancels in an AUROC.

The refusal arm tests this at no additional cost, because there the contrast is instruction content
and neither wrapper arm is coupled to the positive class. Computing sigma^2_(model x wrapper)
separately over the six evaluative wrappers and the six deployment wrappers:

- ratio near 7.5 persists: the asymmetry is a property of the framings, and the prescription is to
  sample the evaluative arm harder.
- ratio collapses toward 1.3: the asymmetry is direction-to-scoring-set coupling, and the
  prescription is to sample harder on whichever arm resembles the positive class of the scoring set.

**T3.10a Amended 2026-08-02, before the refusal statistic was computed.** Two things were wrong
with the version above.

First, the effect is not specific to model interaction. The eval arm's variance exceeds the
deployment arm's by a similar factor in every term the arm enters -- arm main effect 6.2, model by
arm 7.4, arm by item 6.9 -- so "model-reordering variance" was the wrong description and whatever
amplifies acts on the arm effect wholesale. In standard-deviation terms the target is about 2.4x
against a raw direction-space norm ratio of 1.14, so the amplification to explain is roughly 2.1x,
not sevenfold.

Second, the magnitude is not resolved. A ratio of two components on five degrees of freedom each is
approximately the true ratio times F(5,5), and F(0.975,5,5) = 7.15, giving an interval of about
[1, 53]. The better-powered statement uses the seven per-model estimates on a log scale: geometric
mean 4.2, 95% CI [1.2, 14.6]. Those estimate the combined
(sigma^2_a + sigma^2_ma)/(sigma^2_b + sigma^2_mb), because within one model the arm main effect and
the model-by-arm term are not separable, and they share the arm main effects so the interval is
optimistic.

Three candidate mechanisms have now been measured and none accounts for it: mean proximity to the
scoring set's discriminative axis (median cosine-variance ratio 0.99 against 5.6 required, and the
two models showing it are not the models carrying the asymmetry), and class concentration (median
alignment ratio with the positive class's principal axis 1.12 against 2.1 required, with the
uniform 1.5x scatter asymmetry being a property of the classes rather than of the arms). It is
reported as a robust effect with no identified cause.

**T3.10b A THIRD READING THE REFUSAL ARM CANNOT DISTINGUISH.** The original wording read
persistence across concepts as showing "the asymmetry is a property of the framings", which is
ambiguous between evaluative framings as a class and these six particular strings. Because the
refusal arm reuses the same eighteen texts, an asymmetry arising merely because our six evaluative
wrappers are more heterogeneous than our six deployment wrappers -- one author, one sitting -- will
persist and look like confirmation. The refusal arm discriminates coupling from non-coupling; it
cannot discriminate authoring. Every claim is therefore scoped to "in this framing set", and what
would settle it is a second, independently authored set, which is named as future work rather than
implied by the present design.

Neither actionable version is stated before this test resolves.
