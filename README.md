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

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V-NC0DJ5q-7ePyuXhIgVumtRcRSl8-SD?usp=sharing)<br>
We also provided a simple [demo code](https://colab.research.google.com/drive/1V-NC0DJ5q-7ePyuXhIgVumtRcRSl8-SD?usp=sharing) for how to use them to predict disagreement. 

## Datasets
We used public datasets of subjective tasks that contain annotators’ voting records from their original raw dataset <br>

- [Social Bias Corpus(Sap et al. 2020)](https://maartensap.com/social-bias-frames/index.html) 
- [Social Chemistry 101(Forbes et al. 2020)](https://github.com/mbforbes/social-chemistry-101)
- [Scruples-dilemmas(Lourie, Bras, and Choi 2021)](https://github.com/allenai/scruples)
- [Dyna-Sentiment(Potts et al. 2021)](https://github.com/cgpotts/dynasent)
- [Wikipedia Politeness(Danescu-Niculescu-Mizil et al.
2013)](https://convokit.cornell.edu/documentation/wiki_politeness.html)

You can load our processed version of disagreement datasets using Hugging Face's `datasets`, and you can also download the disagreement datasets in [datasets/](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/tree/main/dataset) <br>

Here are the five datasets with disagreement labels. You can change the following data specifications in using Hugging Face's `datasets`:
- <a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/SBIC_Disagreement">"RuyuanWan/SBIC_Disagreement"</a>: SBIC dataset with disagreement labels;
- <a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/SChem_Disagreement">"RuyuanWan/SChem_Disagreement"</a>: SChem dataset with disagreement labels;
- <a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Dilemmas_Disagreement">"RuyuanWan/Dilemmas_Disagreement"</a>: Dilemmas dataset with disagreement labels;
- <a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Dynasent_Disagreement">"RuyuanWan/Dynasent_Disagreement"</a>: Dynasent dataset with disagreement labels;
- <a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Politeness_Disagreement">"RuyuanWan/Politeness_Disagreement"</a>: Politeness dataset with disagreement labels;


## Models
![plot](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/blob/main/code/Quantifying_Disagreement.png)

In our disagreement prediction experiments, we compared the effect of binary v.s. continous disagreement labels, only text input v.s. text with annotator's demographic information, and text with group-wise annotator's demographic information v.s. text with personal level annotator's demographic information. 

Here are the different models that we stored at Hugging Face. 
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Binary_Classifie"</a>: Binary diagreement classifier trained on SBIC text;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Predictor"</a>: Disagreement predictor trained on SBIC text(regression);
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor">"RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor"</a>: Disagreement predictor trained on SBIC text and individual annotator's demographic information in colon templated format;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/SChem_RoBERTa_Text_Disagreement_Binary_Classifier"</a>: Binary diagreement classifier trained on SChem text;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/SChem_RoBERTa_Text_Disagreement_Predictor"</a>: Disagreement predictor trained on SChem text(regression);
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Demographic-text_Disagreement_Predictor">"RuyuanWan/SChem_RoBERTa_Demographic-text_Disagreement_Predictor"</a>: Disagreement predictor trained on Schem text and individual annotator's demographic information in colon templated format;


## Citation
