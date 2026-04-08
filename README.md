# 🏆 LG Aimers 8th AI Hackathon – 3rd Place

## Overview

This repository contains our solution for the LG Aimers 8th AI Hackathon, where we achieved 3rd place.

* Task: Model compression of EXAONE-4.0-1.2B
* Goal: Reduce model size and improve efficiency while maintaining performance under a fully private evaluation setting

🔗 Hackathon page:
https://dacon.io/competitions/official/236689/overview/description

## Key Contributions

We propose a practical and robust compression pipeline tailored for LLM deployment under constrained environments.

1. Stable Quantization with Activation Outlier Handling
Applied W8A8 quantization
Leveraged QuantizationModifier to mitigate activation outliers
Achieved more stable and reliable quantization compared to naive approaches
2. KV Cache Compression (FP8)
Compressed KV cache into FP8 precision
Effectively reduced memory bandwidth bottleneck
Enabled faster inference and improved efficiency in long-context scenarios
3. Calibration Strategy under Private Evaluation
Since the evaluation dataset was 100% hidden, we designed a robust calibration approach:
Used QA tasks commonly adopted in LLM evaluation
Included instruction-following benchmarks (e.g., Google IFEval)
This strategy improved generalization and prevented overfitting to specific data distributions

## Main Insights
* Find the optimal combination of quantization values and recipe empirically is important
* KV cache compression is a highly effective but often overlooked optimization lever
* Exaone-4.0-1.2B has duplicated layers which can be removed without significant performance loss

## Presentation

You can find our presentation slides [here](./ppt.pdf).

## Acknowledgements

We thank the organizers of LG Aimers and DACON for providing a challenging and well-designed benchmark environment.

