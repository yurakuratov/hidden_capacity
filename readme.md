# Exploring Language Models Embeddings Space Capacity

This repository contains code for compressing texts into language models input vectors and notebooks used to make plots for the paper:

Kuratov, Y., Arkhipov, M., Bulatov, A., Burtsev, M., "Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity"


<p align="center">
  <img src="./notebooks/imgs/compression_schema.png" width="45%" />
  <img src="./notebooks/imgs/results_brief.png" width="45%" />
</p>

Left: **Compressing text into a `[mem]` vector.** The pre-trained LLM is frozen, and we only finetune one or multiple `[mem]` vectors to decode the sequence of tokens $[t_1, t_2, \ldots, t_N]$. `[mem]` vectors are trained for each text separately.

Right: **How many tokens fit into a single input vector?** We estimate maximum number of tokens that can be decoded from a single input vector across various language models

## Python scripts
- `train.py` - implements training loop for text compression into a vector.
- `model.py` - implementation of wrapper that adds trainable input vectors referred as `[mem]` to any model from HF, it is based on Recurrent Memory Transformer (RMT) [implementation](https://github.com/booydar/recurrent-memory-transformer).

## Scripts
- `scripts/run.*.sh` - bash scripts for different models, they include running experiments on PG-19, fanfics, and random texts with single or multiple trainable input `[mem]` vectors.

## Visualizations
- `notebooks/` - Folder with notebooks used for visualizations and collecting results.
- `notebooks/ablation_analyze_results.ipynb` - Table 1, Figure 3, Figure 6. Analysis of compression, capacity in tokens, capacity in terms of entropy.
- `notebooks/plot_length_model_brief.ipynb` - Figure 1, text compression results on PG-19.
- `notebooks/plot_length_vs_n_mem_tokens.ipynb` - Figure 4, scaling compression and number of trainable `[mem]` vectors.
- `notebooks/plot_model_theor_capacity_vs_actual.ipynb` - Figure 5, Theoretical capacity vs empirical.

## Citation
```
TBA
```