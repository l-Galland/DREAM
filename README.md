# DREAM: Tailored Conversations Beyond LLMs

This repository accompanies the paper:

**Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager**  
Lucie Galland, Catherine Pelachaud, Florian Pecune  
*arXiv preprint arXiv:2506.19652 (2025)*  
[View on arXiv](https://arxiv.org/abs/2506.19652)

---

## Overview

DREAM (Dialogue REinforcement Agent Manager) is a framework that advances dialogue management beyond traditional large language models by incorporating reinforcement learning. This repository contains the training scripts, model implementations, and ablation setups described in the paper.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/DREAM.git
cd DREAM
pip install -r requirements.txt
```
## Usage

To train the main dialogue manager:

```bash
python trainer.py
```

## Ablation Experiments

To run the ablation versions described in the paper:

- **Without Hierarchical Reinforcement Learning**:

  ```bash
  python trainer_no_hrl.py
  ```

  - **Without Meta Learning**:

  ```bash
  python trainer_no_meta.py
  ```

  ## Citation

If you use this repository in your work, please cite the following:

```bibtex
@article{galland2025tailored,
  title={Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager},
  author={Galland, Lucie and Pelachaud, Catherine and Pecune, Florian},
  journal={arXiv preprint arXiv:2506.19652},
  year={2025}
}
  ```
