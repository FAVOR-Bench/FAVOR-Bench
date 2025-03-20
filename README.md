# FAVOR-Bench

<div align="center">

<p align="center">
    <img src="./docs/favor-bench.png" width="85%" >
</p>

<h1>A Comprehensive Benchmark for Fine-Grained Video Motion Understanding</h1>


[![arXiv](https://img.shields.io/badge/cs.CV-2503.14935-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.14935)
[![Dataset meta](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-FAVOR-yellow)](https://huggingface.co/datasets/zl2048/FAVOR) 
[![Static Badge](https://img.shields.io/badge/website-FAVOR--Bench-8A2BE2)](https://favor-bench.github.io/)


</div>

---

## 🔥 News

* **`2025.03.19`** 🌟 We released Favor-Bench, a new benchmark for fine-grained video motion understanding!

## Introduction

Multimodal Large Language Models (MLLMs) have shown remarkable capabilities in video content understanding but still struggle with fine-grained motion comprehension. To comprehensively assess the motion understanding ability of existing MLLMs, we introduce FAVOR-Bench, comprising 1,776 videos with structured manual annotations of various motions. Our benchmark includes both close-ended and open-ended tasks. For close-ended evaluation, we carefully design 8,184 multiple-choice question-answer pairs spanning six distinct sub-tasks. For open-ended evaluation, we develop both a novel cost-efficient LLM-free and a GPT-assisted caption assessment method, where the former can enhance benchmarking interpretability and reproducibility. Comprehensive experiments with 21 state-of-the-art MLLMs reveal significant limitations in their ability to comprehend and describe detailed temporal dynamics in video motions. To alleviate this limitation, we further build FAVOR-Train, a dataset consisting of 17,152 videos with fine-grained motion annotations. The results of finetuning Qwen2.5-VL on FAVOR-Train yield consistent improvements on motion-related tasks of TVBench, MotionBench and our FAVOR-Bench. Comprehensive assessment results demonstrate that the proposed FAVOR-Bench and FAVOR-Train provide valuable tools to the community for developing more powerful video understanding models.

### Evaluation Tasks

<p align="center">
    <img src="./docs/tasks.png" width="90%">
</p>
