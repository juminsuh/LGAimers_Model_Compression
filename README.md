# 🏆 LG Aimers 8th AI Hackathon – 3rd Place

## Overview

This repository contains our solution for the LG Aimers 8th AI Hackathon, where we achieved 3rd place.

* Task: Model compression of EXAONE-4.0-1.2B
* Goal: Reduce model size and improve efficiency while maintaining performance under a fully private evaluation setting

🔗 Hackathon page:
https://dacon.io/competitions/official/236689/overview/description

## Key Contributions

We propose a practical and robust compression pipeline tailored for LLM deployment under constrained environments.

1. We handled the activation outliers by applying `W8A8 quantization` and `QuantizationModifier` which achieved more stable and reliable quantization compared to naive approaches. (We found out the optimal approaches empirically.)
2. We compressed `KV cache into FP8 precision` which effectively reduced memory bandwidth bottleneck and enabled improved inference speed.
3. We included `two types of calibration data` which were synthesized by Gemini, which was prompted to generate instruction-following QA (i.e., IFEval from Google) and general Korean text. Since the evaluation setting was a fully private, we planned to utilize one of the most common task (raw text & instruction-following QA) for generalization. 

## Main Insights
* Find the optimal combination of quantization values and recipe empirically is important
* KV cache compression is a highly effective but often overlooked optimization lever
* Exaone-4.0-1.2B has duplicated layers which can be removed without significant performance loss

## Presentation

You can find our presentation slides [here](./ppt.pdf).

## My Contribution
* Constructed the implementation environment and distributed it to other team members.
* Estimated the optimal calibration dataset's distribution which was hidden for evaluation, testing math problems, multilingual texts (English + Korean), and instruction-following tasks.
* Analyzed the model architecture and verified why the compression of KV cache into FP8 could drastically reduce the inference speed.
* Performed multiple experiments such as applying speculative decoding to vLLM, adding special tokens to calibration set, and applying the layer-wise pruning, which makes the team focus on our final strategy. 
* Prepared the slides end-to-end with full understanding of background knowledge about quantization and the target model, providing the final oral presentation. 


## Acknowledgements

We thank the organizers of LG Aimers and DACON for providing a challenging and well-designed benchmark environment.

