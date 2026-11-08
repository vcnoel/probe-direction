# Post-submission checks (descriptive, not registered)

Written 2026-11-08, after an adversarial audit of the post-submission analyses. **None of these
checks was registered before it was computed.** They have no decision rules; they read stored
activations or stored results and report numbers. None changes a reported number of the
submitted paper. Registered analyses are in `docs/controlled-transfer-registration.md`,
`docs/interior-band-registration.md` and `docs/paraphrase-registration.md`.

| Check | Motivation | Script | Output |
|---|---|---|---|
| Band labels | `matched_null_band.json` was computed under the registered interior band, but its conventions said `layers: ALL` and its per-ladder config said "folded max over all layers". The cell scorers now record the band (relative depth 0.25 to 0.75, index / (number of hidden states - 1), hidden state 0 counted) in their config, and `matched_null.py` records it in its conventions. The band file was regenerated from the stored band cells without recomputing any cell. Before that, the band cells were checked against a band-restricted maximum computed from the stored projections (agreement to 7.5e-5; they differ from the all-layer cells by up to 0.22). Numbers are unchanged; only labels changed. | `scripts/ladder_audit_cells.py`, `scripts/matched_null_cells.py`, `scripts/matched_null.py` | `results/post_submission/matched_null_band.json` |
| Embedding-layer orientation | Amendment 1's orientation (and P7d) could be carried by shared vocabulary rather than by a representation the network builds. Hidden state 0 is the mean token embedding, so orientation there is a lexical floor. Reported at hidden state 0 and over the band, with the crossing-partition baseline split into evaluation-minus-deployment and same-class directions, and chance 0.5. Covers every model of the Qwen3, Qwen3.5, OLMo-2 and Qwen2.5 ladders (non-thinking rendering) and the paraphrase models. | `scripts/orientation_layers.py` | `results/post_submission/orientation_layers.json` |
| Signed statistic | The paper's statistic folds each layer's AUROC and takes the maximum over all hidden states, so a direction pointing the wrong way scores like one pointing the right way. Reports, per model and wrapper, the folded statistic and the mean of (AUROC - 0.5) over the band, then per wrapper the scaling r, sign counts and spanning counts under both. | `scripts/signed_statistic.py` | `results/post_submission/signed_statistic.json` |
| Sign-flip null geometry | A paired sign-flip control keeps any component shared by the per-item differences, so its directions may not be content-free. Reports the \|cos\| between sign-flip draws and the real direction, and between independent draws, for the released construction (the released script's own draws). Alongside: the permutation and AR(1) means, and the ladder cells' sign-flip \|cos\|. | `scripts/signflip_cosine.py` | `results/post_submission/signflip_cosine.json` |
| Model-by-wrapper share against nulls | A direct read of the model x wrapper share (and model share) of the real directions and its percentile in each content-free null, per ladder, for all layers and for the band. | `scripts/mw_share_nulls.py` | `results/post_submission/mw_share_nulls.json` |
| Read layer | Where the paper's maximum is attained. Reports the share of wrappers whose maximum sits at hidden state 0 exactly, in the first and last 10 percent of depth, and the median relative depth, per model, for both renderings. | `scripts/read_layer.py` | `results/post_submission/read_layer.json` |

Inputs are the stored arrays under `$EAS_ACTIVATION_ROOT` and the stored post-submission results;
none of these checks runs a model.
