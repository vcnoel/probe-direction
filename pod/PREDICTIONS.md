# What we expect from the pod, written before the pod exists

Committed before any ladder beyond `Qwen3-0.6B` has been extracted. It exists because this project
has already produced one threshold chosen after the statistic was visible (Appendix L, and the
corrections record's entry for it). The cheapest defence against a second is to write the decision
rules down while the answer is still unknown.

These are stated predictions, not a pre-registration: the design was fixed after seeing the Qwen2.5
results, and only the pod ladders are unseen.

---

## 1. The Qwen3 thinking switch

`enable_thinking=False` injects `<think>\n\n</think>\n\n` inside the pooled region; the default does
not. Both are defensible renderings of one prompt, so the switch is a level of the implementation
facet and both are extracted for every model.

**The one rung already measured is the least informative one.** At `Qwen3-0.6B` the 36 wrapper
statistics correlate at +0.982 between renderings, mean absolute shift 0.004. A 0.6B model is the
one least able to do anything with a thinking mode, so a flat result there is what the null and the
effect both predict. Nothing about the switch should be claimed from it.

**Prediction.** The gap between renderings grows with scale, because the larger models are the ones
whose behaviour the mode actually changes. Stated as a direction, not a size: we do not predict a
magnitude, and any magnitude quoted afterwards is a description, not a confirmation.

**Statistic, fixed now.** Same as everywhere else in the paper: folded maximum over all layers of
|AUROC − 0.5|, `lf` scoring array, directions from the all-token framing arrays. The comparison is,
per model, the correlation across the 36 wrappers between renderings and the mean |difference|. The
per-model series is reported whatever it shows.

**What each outcome means, decided now.**

- **Gap grows with scale.** The thinking switch is a scale-dependent implementation facet and joins
  pooling and layer set in Section 7. Before the word "grows" appears in the paper, the trend must
  pass the same test the slope claim passed: refit on disjoint item halves, correlated across
  halves. A monotone series on six points is not evidence on its own — that is the mistake the
  wrapper analysis exists to document, and it applies to us.
- **Flat across the whole ladder.** This is a *result*, and a useful one, not a failed experiment.
  It says not every undeclared choice moves the statistic, which is the direct answer to the
  strongest objection to this paper: *you have only shown that prompts matter.* If a rendering
  difference sitting literally inside the pooled region leaves the statistic alone while the wrapper
  moves its sign, then the facets we identify are specific rather than generic, and the paper can
  say which choices need declaring and which demonstrably do not. Section 7 gains a sentence saying
  so, and the abstract's claim about latitude becomes narrower and better supported.
- **Non-monotonic.** Reported as the series, with no story attached.

The one thing not permitted is deciding after the fact which of these three the result was closest
to. **Flat** means the correlation stays above +0.95 and mean |difference| below 0.01 at every rung;
anything else is not flat.

Where that band came from, since it matters: it is calibrated on the one rung already extracted,
which produced +0.982 and 0.004. It is not a threshold derived from a prior, and a band drawn around
an observed value will always contain that value. Declaring it before the other five rungs exist is
what makes it usable — it can be failed by every model that has not been run — but it is a weaker
instrument than a bar set from theory, and the paper should describe it that way rather than as a
criterion the 0.6B result passed.

## 2. If the authors reply about the transposed axes

§ 10 reports that two released per-layer score files store the layer and item axes transposed
relative to the others, that scoring them as stored reproduces the published figure while scoring
them corrected does not, and that we have contacted the authors. A reply changes that section, and
which way it changes is not ours to choose after the fact. So:

- **They confirm the stored orientation is intended.** The published figure stands and our
  reproduction is correct as printed. Section 10 keeps the finding as a documentation gap rather than
  an error, and the sentence about the sign depending on which is intended is withdrawn, because it
  would no longer be undetermined. This is the outcome that costs us a claim, and it is the one we
  should expect to have to write.
- **They confirm the axes are transposed in error.** The corrected scoring is the right one and the
  published sign changes. We report that, credit the correction to them, and it strengthens nothing
  we argue: our case rests on our own wrappers precisely so that it does not depend on this.
- **No reply before the deadline.** Section 10 stands as written, since it already states both
  scorings and declines to pick one.

In every branch the reanalysis conclusion is unchanged, because Section 10 is an argument for why we
argue the sign from our own wrappers rather than from released arrays. If a reply arrives during
rebuttal, it goes in the rebuttal under the same three rules, decided here.

## 2. The sixth Qwen3 rung, written before it is looked at

The committed Qwen3 replication is FIVE rungs, 0.6B-14B. Qwen3-32B was still transferring when it
was analysed, so nothing below has been observed. Written now so the sixth point cannot be read
after the fact.

The mechanism claimed for the drop from 6 of 6 (Qwen2.5, four rungs) to 3 of 6 (Qwen3, five rungs)
is that more points estimate each wrapper's correlation better, so fewer sign flips arise by
accident. If that mechanism is right, a sixth rung extending the span from 1.29 to 1.64 decades
predicts:

- **the both-signs framing count falls, at or below 3 of 6** on the nothink rendering, and at or
  below 4 of 6 on think;
- **the range of r narrows**, because the extreme correlations at five points are the ones most
  helped by an additional, well-separated point;
- the positive/negative split of the 36 wrappers moves toward positive, since most negative r values
  at five rungs sit close to zero.

A count that RISES, or a range that widens, falsifies the better-estimation mechanism and means the
flip count is tracking something about which models are on the ladder rather than how many.

Either way the claim the paper rests on is untouched: what matters is that the range spans both
signs, not how many evaluation framings individually do.

## 3. What OLMo-2 is for, and what would stop it

**Qwen3 does not relieve the limitation the abstract states.** That limitation is *family*, not
range. Qwen3 buys six scales from 0.6B to 32B within one lab and one tokenizer lineage, which
sharpens the scaling axis and leaves "the coefficients are the Qwen2.5 ladder's" exactly where it
was. A reviewer will not count Qwen3 as a second vendor, and they will be right.

OLMo-2 is the run that changes the sentence. So the stopping rule is not "stop if Qwen3 came out
clean":

- **OLMo-2 runs unless Qwen3 fails**, not unless Qwen3 succeeds. A clean Qwen3 ladder is a reason to
  continue, since it means the pipeline works and the remaining cost buys the independence claim.
- **If the budget forces a cut, cut `Qwen3-32B` first.** It is the single most expensive rung
  (66 GB of weights, batch 7) and the scale axis is already carried by 0.6B–14B. Dropping the top of
  one ladder costs a data point; dropping the second ladder costs the paper's family limitation.
- **Qwen3 failing** means the extraction does not reproduce the statistic in a plausible range, the
  prompt hashes do not verify, or a rung OOMs in a way that changes the batch size mid-ladder. It
  does not mean "the result was uninteresting."

**Declared in advance, because it is already known:** `OLMo-2-0325-32B-Instruct` does not prepend
`bos_token` while the 1B, 7B and 13B models do — `render_prompts.py` records this per template group
and prints it. Consequences, fixed now rather than after seeing the numbers:

- The `lf` scoring array, which is the paper's declared convention, drops BOS anyway. The primary
  statistic is unaffected and the OLMo-2 ladder is comparable end to end under it.
- The **all-token** arm is *not* comparable across this ladder: at 32B the mean is taken over a
  token set the smaller models do not have. Any all-token quantity on OLMo-2 is reported per model
  or not at all, and never averaged across the ladder. This is a convention difference of exactly
  the kind Section 7 measures, so it will be declared in the results file, not mentioned in prose.

---

## 4. The Qwen3.5 ladder, written before any of it is extracted

Written 2026-09-25, before any Qwen3.5 checkpoint has been run. The five rungs are `Qwen3.5-0.8B`,
`-2B`, `-4B`, `-9B` and `-27B`, pinned by revision in `configs/ladders.json`. They are the current
generation of the family the primary design was built on, so the ladder tests whether the wrapper
result is a property of Qwen2.5 or of the estimator.

**Declared in advance, because it is already known.** All five checkpoints are vision language
models whose text decoder is loaded alone; parameter counts are text parameters only. The layers are
three linear attention blocks to one full attention block; the folded maximum over layers does not
care, but any per layer statement must say which layers it counts. The template's thinking default
changes with size (off at 0.8B and 2B, on from 4B), so both renderings are fixed explicitly with
`enable_thinking` and neither reproduces the Qwen2.5 prompt shape.

**Statistic, fixed now.** The same as Sections 1 to 3: folded maximum over all layers of
|AUROC − 0.5|, `lf` scoring array, directions from the BOS dropped arrays, analysed by
`scripts/pod_ladders.py` with the code the other ladders use.

**Predictions.**

- **Replication.** Across the 36 wrappers the slope of the statistic on log parameters takes both
  signs, as on every ladder measured so far. Failure means one sign on every wrapper, and would be
  reported as the ladder where the wrapper does not decide the sign.
- **Components.** The model's share of total variance stays below the wrapper by model share, and
  wrapper by model exceeds item by model. Magnitudes are not predicted, since they have not
  transferred across ladders.
- **Reliability.** E rho^2 at the published k stays below the conventional 0.80.
- **Thinking switch.** Reported per rung under the band of Section 1, with no direction predicted,
  because the default flips inside this ladder and a trend across the flip is not interpretable.

Whatever the ladder shows is reported, in the main text, beside the other ladders.
