# A Probe Direction Is a Property of Its Prompt: code and results

Anonymous repository for the ICLR 2027 submission *A Probe Direction Is a Property of Its Prompt*.
It holds the analysis code, the model and wrapper configurations, and every result file the
manuscript's numbers, tables and figures are generated from.

## Layout

```
src/eval_awareness_spectral/   library the analyses import (manifests, splits, statistics, probes)
scripts/                       one script per result; scripts/pipeline.py declares the dependency
                               graph and generates the Makefile
results/*.json                 every result the manuscript reads (activation arrays are not included)
configs/                       model ladders pinned by revision, release dates, claim ledger, schema
data/                          wrapper framings, task pools, and a manifest (hashes, counts, template
                               ids) of the rendered prompt sets
pod/                           extraction on a large GPU for the Qwen3, OLMo-2 and Qwen3.5 ladders;
                               pod/PREDICTIONS.md holds the predictions recorded for the added
                               ladders before their checkpoints were run
paper_tools/                   the generators that turn results/*.json into the manuscript's macros,
                               tables and figures
docs/                          pre-registered analysis rules, generation contract, mathematical notes
tests/                         CPU-only unit tests
```

## Install

Python 3.11. The result files record an AST hash of the script that produced them, and every
generator refuses a file whose producer has changed since it was written. `ast.dump` output differs
between Python minor versions, so under 3.12 or later every check reports a stale producer even
though nothing changed.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[analysis,dev]"
pytest                                   # CPU-only unit tests
```

`paper_tools/make_figures.py` draws with matplotlib's pgf backend, so it needs a LaTeX installation
with `pdflatex` on the path.

## Reproducing every number and figure (CPU, minutes)

Every number in the paper is a LaTeX macro generated from a file in `results/`. From the stored
results:

```bash
python paper_tools/fill_numbers.py            # main macros         -> paper_tools/out/generated_numbers.tex
python paper_tools/make_ladders.py --strict   # ladder macros/table -> paper_tools/out/ladder_numbers.tex, ladders_table.tex
python paper_tools/make_subsets.py --strict   # subset reliability  -> paper_tools/out/subset_numbers.tex
python paper_tools/make_figures.py            # the six body figures -> paper_tools/out/figures/fig*.pdf
```

Set `EAS_OUT=<dir>` to write elsewhere. Each generator checks the conventions block and producer
hash of every input and refuses a stale or incomplete file. `fill_numbers.py` also rewrites
`results/macro_provenance.json` and `results/macro_conventions.json`, which record the content hash
of every input it read. `fill_numbers.py --verify --tex-dir <manuscript dir>` additionally checks
that every result macro the manuscript uses is defined.

The CPU analyses that sit between the ladder extractions and the generators can be rerun from the
stored per-ladder summaries:

```bash
python scripts/ladder_curves.py               # per-wrapper statistic on every ladder
python scripts/wrapper_subsets.py             # disjoint wrapper-subset reliability
```

## Regenerating the results themselves

This needs activation extraction on one GPU (a 16 GB card for the checkpoints up to 4B; an
80 GB-class card for the larger rungs) and inputs we do not redistribute:

```bash
export EAS_REANALYSIS_ROOT=/path/to/released-artifacts   # the two reanalysed studies' released files
export EAS_SAD_ARCHIVE=/path/to/sad_archive.zip          # the SAD stages_oversight split
export EAS_SAD_PASSWORD=...                              # obtained from the SAD project

# 1. Render the prompts locally. The rendered files embed SAD question text and are never committed;
#    scripts/render_prompts.py --verify checks them against data/rendered/manifest.json.
python scripts/render_prompts.py

# 2. Extract pooled activations.
python scripts/extract_model.py --all                   # smaller ladders, models in configs/gstudy_models.json
python scripts/gstudy_extract.py --model qwen2.5-3b     # wrapper x item crossing, per model
python scripts/extract_sad_filtered.py --model qwen2.5-0.5b
python scripts/extract_refusal.py --all                 # second concept (refusal splits, see below)
POD_LADDER=qwen3 bash pod/run.sh                        # larger ladders: qwen3 | olmo2 | qwen3.5
python scripts/ingest_pod.py --src <extracted dir>

# 3. Analyses, in dependency order, then the ladder summaries.
python scripts/pipeline.py --run                        # every results file of the main analysis
python scripts/pod_ladders.py                           # results/pod_*.json
python scripts/ladder_curves.py && python scripts/wrapper_subsets.py
python scripts/backfill_conventions.py --write          # required after rerunning any analysis
```

`python scripts/pipeline.py --list` prints the dependency graph; `make all` runs the same graph.

## Not included, and why

| Item | Reason |
|---|---|
| SAD benchmark question text, and the rendered prompt files `data/rendered/prompts_rendered_*.json` that embed it | Licence-restricted. The SAD authors ask that item text never appear in plain text anywhere scrapable. `data/rendered/manifest.json` carries hashes, counts and template ids only, so a locally rendered set (step 1 above) can be verified against it. |
| Released direction vectors, contrastive sets and notebooks of the two reanalysed studies | They belong to their authors; fetch them from the original releases and point `EAS_REANALYSIS_ROOT` at them. |
| Activation arrays (`results/**/*.npz`, `*.npy`, `*.pt`, `*.csv`, about 39 GB) | Large and regenerable by the extraction scripts. The JSON summaries the manuscript reads are included. |
| Harmful/harmless instruction splits used for the second concept | Public; obtain them from the original release. |
| The manuscript sources | Submitted separately. `paper_tools/` writes the macro, table and figure files the manuscript includes. |

## Data licences

`data/framings.json`, `data/framings_multi.json` and `data/tasks.jsonl` were written for this
work and are released under the repository licence. `data/tasks_bench.jsonl` (60 tasks) also
contains 15 items each from GSM8K (MIT licence), the AI2 Reasoning Challenge (CC BY-SA 4.0) and
MMLU (MIT licence), reproduced under those licences; the `bench_*` category of each row names its
source.

## Licence

AGPL-3.0-or-later (`LICENSE`).
