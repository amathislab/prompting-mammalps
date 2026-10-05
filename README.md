# Prompting MammAlps: Text-to-video retrieval on the MammAlps-S2 dataset
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-4b44ce)](https://eccv.ecva.net/virtual/2026/poster/5474)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-Prompting--MammAlps-FFD21E)](https://huggingface.co/datasets/EPFL-ECEO/Prompting-MammAlps)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the **official code to evaluate your method on the prompting-mammalps benchmark**.

We also include:
- the agent parsing framework used in the proposed baseline (`models/llm`)
- a tutorial notebook to visualize labels on the raw videos (`tutorials/visualize_predictions.ipynb`)
- intermediary result files to reproduce our baseline performance (`baseline`)
- SALMA architecture, training and inference scripts (`models/salma`)

To keep a simple repository, we did not include the code to run SOTA comparisons and the ablation experiments.

## Installation

```bash
conda create -n prompting-mammalps python=3.12
conda activate prompting-mammalps
pip install uv
uv pip install -e .                   # base: evaluation and visualization
uv pip install -e ".[smolagents]"     # + parsing agent framework
uv pip install -e ".[salma]"          # + SALMA training and inference
```

## Dataset

Download the dataset from the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="amathislab/Prompting-MammAlps",
    repo_type="dataset",
    local_dir="./data",
)
```

This gives :
- `./data/videos/`
- `./data/annotations/` (annotated animal tracks per video), 
- `./data/metadata/` (`label_mapping.json`, `query_categories.json`, `queries_and_videos.json`). 

Scripts below default to `./data`. Override with the relevant `--data-root` flag if you downloaded it elsewhere.

## Evaluation

**Predicted TVR file format**:  
To evaluate your method on prompting-mammalps, you need to prepare a `.json` file with the text queries as keys and the retrieved list of corresponding candidate (test) file ids as value. This format follows the ground-truth associations available [here](https://huggingface.co/datasets/amathislab/Prompting-MammAlps/blob/main/metadata/queries_and_videos_test.json) or the [baseline predictions](./baseline/Qwen3-8B-salma.json).

For example:  
```json
{
  "An animal engaged in any activity other than foraging.": [
    "S3_C3_E501_V0277",
    "S3_C3_E523_V0324",
    ...],
  "An animal that is neither a red deer nor a roe deer.": [
    "S1_C3_E61_V0029",
    "S1_C5_F168_V0221",
    ...],
  ...
}
```

**Evaluation script**:  
`evaluation/evaluate.py` reports macro-averaged F1, broken down per query and per ecology/vision
category from `metadata/query_categories.json`.

```bash
python evaluation/evaluate.py retrieval \
  --data-root ./data \
  --predictions baseline/generated_functions_retrieval.json \
  --output baseline_results.csv
```

## Repository structure

```
baseline/       Intermediary predictions from SALMA and parsing agents used to derive baseline performance
models/
  salma/        SALMA training and inference (see models/salma/README.md)
  llm/          Parsing agent framework
evaluation/
  metrics.py    Shared metric implementations (F1)
  evaluate.py   Retrieval evaluation script (see below)
tutorials/
  visualize_predictions.ipynb   Visualize ground truth or model predictions
```

## SALMA

SALMA detects, tracks and describes animals in each video. Its per-video prediction JSONs, which use the same schema as `data/annotations/`, are the input of the parsing agent below.

```bash
uv pip install -e ".[salma]"   # from the repo root
cd models/salma
bash scripts/train.sh          # two-stage curriculum training
bash scripts/inference.sh      # two-view inference -> output/.../multi_view/<video_id>.json
```

See [models/salma/README.md](models/salma/README.md) for data, weights and options.

## LLM-based parsing agent

The agent turns each benchmark query into a short Python function that checks a video's annotation JSON against it, using [smolagents](https://github.com/huggingface/smolagents).


```bash
uv pip install -e ".[smolagents]"   # from the repo root
cd models/llm
bash scripts/run_llm_agent.sh   # edit --llm / paths as needed, or run.py directly
```

  * Some LLM models require setting up a huggingface authentification key.
  * We ran the models with their recommended parameters that lead to non-deterministic behaviors. Repeating this stage of the pipeline yields slightly different results.

The output from `models/llm/run.py` is a `.json` file that has the text queries as keys and the generated `check_file` functions as values; apply them to a folder of prediction/annotation JSONs with `evaluate_llm_functions.py` to get a
`{query: [video_id, ...]}` retrieval result, which is the input to `evaluate.py retrieval` above.

Generated functions can then be applied against the candidate videos JSON file representation. For example, to obtain the set of retrieved videos when using the ground-truth annotations (oracle):  
```bash
python evaluate_llm_functions.py \
  -IQF ../../baseline/Qwen3-8B_generated_functions.json \
  -IJ ./data/annotations/test \
  -OJ ./baseline/Qwen3-8B_oracle.json
```

## Tutorial

`tutorials/visualize_predictions.ipynb` downloads a small slice of the dataset and renders bounding boxes + attributes (species, action, activity,
demographics) on a video, from either the ground-truth annotations or a SALMA prediction JSON (same schema).

## Citation

```bibtext
@article{gabeff2026prompting,
  title={Prompting-MammAlps: Fine-Grained Text-to-Video Retrieval for Camera-Trap Data},
  author={Gabeff, Valentin and Maquignaz, Baptiste and Shan, Jennifer and Mamooler, Sepideh and Sumbul, Gencer and Costelloe, Blair and Tuia, Devis and Mathis, Alexander},
  journal={arXiv preprint arXiv:2607.09876},
  year={2026}
}
```

## Code attributions

| Author | Attributions |
| ---    | -------------         |
| Valentin Gabeff | benchmark elaboration, evaluation, agent-based parsing framework, salma, tutorial |
| Sepideh Mamooler | agent-based parsing framework |
| Baptiste Maquignaz | salma |
| Jennifer Shan | benchmark elaboration, similarity-based evaluation |

