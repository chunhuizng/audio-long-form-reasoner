# Model Priors Shape Experience: Reinforcement Learning for Complex Audio Reasoning


## Features

1. The **first** implementation to support the **Qwen2.5-Omni Thinker** as a base model for **complex end-to-end audio reasoning** tasks.  
2. The **first** to support **vLLM** as a rollout speed-up framework to improve GPU utilization for mainstream audio LLMs, including the Qwen2 and Qwen2.5-Omni series.

## Introduction

This is a training framework for **audio-based logical question answering (AQA)** tasks requiring **very complex reasoning** over real-world sound.

We use **GRPO with Dr_GRPO** to improve token efficiency while preserving high reasoning quality. Our system reinforces responses aligned with task-specific reward functions — a token-efficient, feedback-driven training loop. Unlike prior multimodal QA work focused on **shallow or recognition-level tasks**, our setup centers on **long-form complex reasoning from audio-text pairs**, where auditory inputs must be **understood, abstracted, and reasoned about**.

> Inspired by *The Era of Experience*, we treat reinforcement learning not merely as optimization, but as a vehicle for **experience generation**.  
> We show that choosing a **strong base model** (e.g., Qwen2.5-Omni Thinker) elicits **better model trajectories**, producing higher-quality completions and thus **better learning signals** for RL.  
> This reflects the insight that **model priors shape the experience**, and that **better experience leads to better RL reasoners**.


## Training

### Data Preparation

Download dataset:

```bash
huggingface-cli download chunhuizng/audiokk --repo-type=dataset
```

### Model Preparation

Download model:

```bash
huggingface-cli download chunhuizng/AudioOnlyThinker --repo-type=model
```

### GRPO Training

Run training:

```bash
sh sv_vllm.sh
```

### Other Notes

- Install [OpenR1](https://github.com/huggingface/open-r1)
- Then run:
  
  ```bash
  uv pip install -r requirements.txt
  ```

- ✅ **Recommended CUDA versions**: 12.2, 12.4, or 12.8  
- ✅ **Recommended hardware**: at least **4× H100s** or **8× A6000s**
