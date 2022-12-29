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
We uploaded both our datasets and model checkpoints to Hugging Face's [repo](https://huggingface.co/RuyuanWan). You can directly load our data using `datasets` and load our model using `transformers`.
```python
# load our dataset
from datasets import load_dataset
dataset = load_dataset("RuyuanWan/SBIC_Disagreement")
# you can replace "SBIC_Disagreement" to "SChem_Disagreement", "Dilemmas_Disagreement", "Dynasent_Disagreement" or "Politeness_Disagreement" to change datasets

# load our model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor")
model = AutoModelForSeq2SeqLM.from_pretrained("RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor")
# you can replace "SBIC_RoBERTa_Demographic-text_Disagreement_Predictor" to other pretrained models
```

## Datasets
We used public datasets of subjective tasks that contain annotators’ voting records from their original raw dataset <br>

- [Social Bias Corpus(Sap et al. 2020)](https://maartensap.com/social-bias-frames/index.html) 
- [Social Chemistry 101(Forbes et al. 2020)](https://github.com/mbforbes/social-chemistry-101)
- [Scruples-dilemmas(Lourie, Bras, and Choi 2021)](https://github.com/allenai/scruples)
- [Dyna-Sentiment(Potts et al. 2021)](https://github.com/cgpotts/dynasent)
- [Wikipedia Politeness(Danescu-Niculescu-Mizil et al.
2013)](https://convokit.cornell.edu/documentation/wiki_politeness.html)

You can load our processed version of disagreement datasets using Hugging Face's `datasets`, and you can also download the disagreement datasets in [datasets/](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/tree/main/dataset) <br>

## Models

![plot](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/blob/main/code/Quantifying_Disagreement.png)
## Citation
