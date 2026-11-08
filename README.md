<div align="center">

<h1>Counterfactual Reasoning for<br>Robust Visual Question Answering</h1>

<p>
  <strong>KES 2026</strong> ·
  <em>Procedia Computer Science</em> · Elsevier
</p>

<p>
  <a href="https://doi.org/10.1016/j.procs.2026.09.191">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Elsevier-334155?style=flat&logo=doi&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2609.16567">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.16567-B31B1B?style=flat&logo=arxiv&logoColor=white">
  </a>
  <a href="https://github.com/duongtruongbinh/robust-vqa-counterfactual">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-24292F?style=flat&logo=github&logoColor=white">
  </a>
</p>
<sub>Official PyTorch implementation</sub>

</div>

## Overview

Visual Question Answering (VQA) models often rely on language priors instead of visual evidence, limiting out-of-distribution (OOD) generalization. We combine **counterfactual sample synthesis** with a **three-stage training curriculum** and three complementary objectives:

- **Batch-Contrastive (BC):** Discriminates factual and counterfactual representations using in-batch negatives.
- **Answer-Contrastive (AC):** Regularizes the answer prediction space.
- **Gradient-Discrepancy (GD):** Encourages sensitivity to causally relevant visual evidence.

## Results

| Benchmark                  |   Accuracy |
| :------------------------- | ---------: |
| **VQA-CP v2** (OOD test)   | **61.64%** |
| **VQA v2** (ID validation) | **62.80%** |

**OOD–ID accuracy gap:** 1.16 %. Results are reported in the [paper](https://doi.org/10.1016/j.procs.2026.09.191).

## Setup

**Requirements:** Linux, NVIDIA GPU, Conda.

```bash
git clone https://github.com/duongtruongbinh/robust-vqa-counterfactual.git
cd robust-vqa-counterfactual

conda create -n robust-vqa python=3.7 -y
conda activate robust-vqa
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
pip install h5py==2.10.0 Click==7.0 numpy==1.16.5 tqdm==4.35.0 spacy==2.2.4 wandb
python -m spacy download en_core_web_lg
```

### Data preparation

1. Download the VQA/VQA-CP annotations and GloVe embeddings, then preprocess them:

   ```bash
   bash tools/download.sh
   bash tools/process.sh
   ```

2. Place the **pre-extracted bottom-up image features** in `data/rcnn_feature/`: one `{image_id}.pth` file per image, containing `image_feature` with shape `(36, 2048)`. See [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa).

3. Obtain the remaining assets from the [CSS-VQA](https://github.com/yanxinzju/CSS-VQA) resources:
   - Hint scores: `data/train_cpv2_hintscore.json`, `data/test_cpv2_hintscore.json` (and corresponding VQA v2 files).
   - Answer embeddings: `data/answer_embs_ac.npy`, `data/answer_embs_ac_v2.npy`.
   - Question-type metadata and masks in `util/` (`qid2type_*.json`, `*_type_mask.json`, `*_notype_mask.json`).

## Training & Evaluation

Use the provided experiment script, or configure arguments in `main.py`.

```bash
bash run.sh
```

`run.sh` specifies the training configuration; outputs and checkpoints are saved under `logs/<experiment_name>/`.

## Evaluation

Evaluate a saved checkpoint on the selected dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --dataset cpv2 \
  --model_state logs/exp_name/model.pth
```

For the complete set of options, see [`main.py`](main.py) and [`eval.py`](eval.py).

## Citation

If you find this work useful, please cite:

```bibtex
@article{duong2026counterfactual,
  title     = {Counterfactual Reasoning for Robust Visual Question Answering},
  author    = {Duong, Truong-Binh and Tran, Thanh-Ngan and Nguyen, Ngoc-Thao and Le, Bac},
  journal   = {Procedia Computer Science},
  volume    = {292},
  pages     = {1162--1171},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.procs.2026.09.191},
}
```

## Acknowledgments

This implementation builds on [CSS-VQA](https://github.com/yanxinzju/CSS-VQA) and [CL-VQA](https://github.com/jokieleung/CL-VQA). We thank the authors for releasing their research code.

## License

This repository contains adapted third-party research code. Please review the upstream repositories' terms and the respective dataset licenses before reuse or redistribution.
