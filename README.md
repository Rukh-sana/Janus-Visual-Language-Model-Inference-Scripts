# Janus-Visual-Language-Model-Inference-Scripts
This repository contains inference scripts for running a multi-modal visual language model using Deepseek Janus Pro. It supports both GPU (CUDA-enabled) and CPU-only environments. The model takes an input image and answers a custom visual question.


# Janus Visual Language Model Inference Scripts

This repository provides Python scripts to perform inference using the **Deepseek Janus Pro** visual language model (VLM). The model processes an image along with a text prompt and generates descriptive or analytical responses.

## 🧠 Model Overview

The scripts use `janus.models.MultiModalityCausalLM` and `VLChatProcessor` to run multimodal inference. The input is an image and a textual query, and the output is a generated textual response.

## 🧰 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Deepseek Janus modules
- PIL or similar image loader

## 🖥️ Scripts

### `test.py` (GPU version)

Use this script if your system supports **CUDA** and **GPU acceleration**.

```bash
python test.py
