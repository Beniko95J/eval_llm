---
license: apache-2.0
language:
- en
pipeline_tag: image-text-to-text
---


# Model description
`xGen-MM` is a series of the latest foundational Large Multimodal Models (LMMs) developed by Salesforce AI Research. This series advances upon the successful designs of the `BLIP` series, incorporating fundamental enhancements that ensure a more robust and superior foundation. These models have been trained at scale on high-quality image caption datasets and interleaved image-text data. 

In the v1.5 (08/2024) release, we present a series of XGen-MM models including:
- [🤗 xGen-MM-base](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-base-r-v1.5): `xgen-mm-phi3-mini-base-r-v1.5`
- [🤗 xGen-MM-instruct](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5): `xgen-mm-phi3-mini-instruct-singleimg-r-v1.5`
- [🤗 xGen-MM-instruct-interleave (our main instruct model)](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-multi-r-v1.5): `xgen-mm-phi3-mini-instruct-interleave-r-v1.5`
- [🤗 xGen-MM-instruct-dpo](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5): `xgen-mm-phi3-mini-instruct-dpo-r-v1.5`

In addition to the models, our team also released a series of datasets for multi-modal pre-training, including:
- [🍃 MINT-1T: Scaling Open-Source Multimodal Data by 10x: A Multimodal Dataset with One Trillion Tokens](https://arxiv.org/abs/2406.11271)
- [🤗 BLIP3-OCR-200M](https://huggingface.co/datasets/Salesforce/blip3-ocr-200m): a dataset with dense OCR annotations.
- [🤗 BLIP3-GROUNDING-50M](https://huggingface.co/datasets/Salesforce/blip3-grounding-50m): a dataset for enhancing the ability to ground semantic concepts in images.
- BLIP3-KALE (stay tuned): a large-scale curated high-quality caption dataset. 

For more details, check out our [tech report](https://arxiv.org/pdf/2408.08872), [fine-tuning code](https://github.com/salesforce/LAVIS/tree/xgen-mm), and project page (coming soon).

# Data
The base model is pre-trained on a mixture of data sources described above, with around 100 billion image-text tokens in total.


# Results

### Few-shot Evaluation on Base model (without instruction tuning)

| Model         | Shot | VQAv2 | TextVQA | OKVQA | COCO  | NoCaps | TextCaps |
|:--------------|:-----|:------|:--------|:------|:------|:-------|:---------|
| Flamingo-3B   | 0    | 49.2  | 30.1    | 41.2  | 73.0  | -      | -        |
|               | 4    | 53.2  | 32.7    | 43.3  | 85.0  | -      | -        |
|               | 8    | 55.4  | 32.4    | 44.6  | 90.6  | -      | -        |
| MM1-3B        | 0    | 46.2  | 29.4    | 26.1  | 73.5  | 55.6   | 63.3     |
|               | 4    | 57.9  | 45.3    | 44.6  | **112.3** | 99.7   | 84.1     |
|               | 8    | 63.6  | 44.6    | 48.4  | **114.6** | **104.7**  | 88.8     |
| xGen-MM-base  | 0    | 43.1  | 34.0    | 28.0  | 67.2  | 82.6   | 69.5     |
|               | 4    | **66.3**| **54.2**| **48.9**| 107.6 | **100.8**| **89.9**     |
|               | 8    | **66.9**| **55.3**| **50.1**| 109.8| 104.6| **94.0**|


### Showcases on In-Context Learning

Below are some qualitative examples below of the mutli-modal in-context learning capacity of our base model.

<img src="icl_examples/art.png" alt="Art" width=500>


<img src="icl_examples/animal.png" alt="Animal" width=500>


<img src="icl_examples/street.png" alt="Street" width=500>


# How to use

Please check out our [inference notebook](demo.ipynb) for example code to use our model.

# Reproducibility: 

The pretraining evaluation is implemented based on [OpenFlamingo: An open-source framework for training large multimodal models.](https://github.com/mlfoundations/open_flamingo).
Few-shot examples are randomly drawn so there will be some variance with different random seeds.

# Bias, Risks, Limitations, and Ethical Considerations
The main data sources are from the internet, including webpages, 
image stock sites, and curated datasets released by the research community. We have excluded certain data, such as LAION, due to known CSAM concerns.
The model may be subject to bias from the original data source, as well as bias from LLMs and commercial APIs. 
We strongly recommend users assess safety and fairness before applying to downstream applications. 


# License

Our code and weights are released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) license.

# Code acknowledgement
Our training code is based on [OpenFlamingo: An open-source framework for training large multimodal models.](https://github.com/mlfoundations/open_flamingo), and part of our data preprocessing code is adapted from [LLaVA](https://github.com/haotian-liu/LLaVA).
Our evaluation code is based on [VLMEvalKit: Open-source evaluation toolkit of large vision-language models (LVLMs)](https://github.com/open-compass/VLMEvalKit).

We thank the authors for their open-source implementations.


# Citation
```
@article{blip3-xgenmm,
  author    = {Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, Shrikant Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, Ran Xu},
  title     = {xGen-MM(BLIP-3): A Family of Open Large Multimodal Models},
  journal   = {arXiv preprint},
  month     = {August},
  year      = {2024},
}
```


# Troubleshoot

1. If you missed any packages, please consider the following

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch==2.24.0
pip install einops
pip install einops-exts
pip install transformers==4.41.1
```