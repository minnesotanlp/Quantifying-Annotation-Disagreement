# Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information
This repository provides datasets and code for preprocessing, training and testing models for quantifying annotation disagreement with the official Hugging Face implementation of the following paper:

> Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information <br>
> [Ruyuan Wan](https://ruyuanwan.github.io/), [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), [Dongyeop Kang](https://dykang.github.io/) <br>
> [AAAI 2023](https://aaai.org/Conferences/AAAI-23/) <br>

Our code is mainly based on HuggingFace's `transformers` libarary.

## Installation
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using Python 3.7.


## HuggingFace Integration


## Datasets
We used public datasets of subjective tasks that contain annotators’ voting records from their original raw dataset <br>

- [Social Bias Corpus](https://maartensap.com/social-bias-frames/index.html) 
- [Social Chemistry 101](https://github.com/mbforbes/social-chemistry-101)
- [Scruples-dilemmas](https://github.com/allenai/scruples)
- [Dyna-Sentiment](https://github.com/cgpotts/dynasent)
- [Wikipedia Politeness](https://convokit.cornell.edu/documentation/wiki_politeness.html)

You can load our processed version of disagreement datasets using Hugging Face's `datasets`, and you can also download the disagreement datasets in [datasets/](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/tree/main/dataset) <br>

## Models

![plot](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/blob/main/code/Quantifying_Disagreement.png =100x20)

## Citation
