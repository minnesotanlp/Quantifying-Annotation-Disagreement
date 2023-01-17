# Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information
This repository provides datasets and code for preprocessing, training and testing models for quantifying annotation disagreement with the official Hugging Face implementation of the following paper:

> [Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information](https://arxiv.org/abs/2301.05036) <br>
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
from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs()
model_args.regression = True
SBIC_person_demo_col_regression = ClassificationModel(
    "roberta",
    "RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor",
    num_labels=1,
    args=model_args
)
# you can replace "SBIC_RoBERTa_Demographic-text_Disagreement_Predictor" to other pretrained models

#predict
# you can replace example text to other random examples. 
text_example1 = ['Abortion should be legal']
predict1, raw_outputs1 = SBIC_person_demo_col_regression.predict(text_example1)
print(predict1)
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
- : ;
- : ;
- : ;
- : ;

<table>
    <tr>
        <th>Dataset name in Hugging Face</th>
        <th>Dataset information</th>
    </tr>
    <tr>
        <th><a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/SBIC_Disagreement">"RuyuanWan/SBIC_Disagreement"</a> </th>
        <th>SBIC dataset with disagreement labels</th>
    </tr>
    <tr>
        <th><a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/SChem_Disagreement">"RuyuanWan/SChem_Disagreement"</a></th>
        <th>SChem dataset with disagreement labels</th>
    </tr>
    <tr>
        <th><a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Dilemmas_Disagreement">"RuyuanWan/Dilemmas_Disagreement"</a></th>
        <th>Dilemmas dataset with disagreement labels</th>
    </tr>
    <tr>
        <th><a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Dynasent_Disagreement">"RuyuanWan/Dynasent_Disagreement"</a></th>
        <th>Dynasent dataset with disagreement labels</th>
    </tr>
    <tr>
        <th><a target="_blank" href="https://huggingface.co/datasets/RuyuanWan/Politeness_Disagreement">"RuyuanWan/Politeness_Disagreement"</a></th>
        <th>Politeness dataset with disagreement labels</th>
    </tr>    
</table>

## Models
In our disagreement prediction experiments, we compared:
- Binary v.s. continous disagreement labels, 
- Only text input v.s. text with annotator's demographic information,  
- Text with group-wise annotator's demographic information v.s. text with personal level annotator's demographic information. 

![plot](https://github.com/minnesotanlp/Quantifying-Annotation-Disagreement/blob/main/code/Quantifying_Disagreement.png)

Here are the different models that we stored at Hugging Face. 
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Binary_Classifie"</a>: Binary diagreement classifier trained on SBIC text;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/SBIC_RoBERTa_Text_Disagreement_Predictor"</a>: Disagreement predictor trained on SBIC text(regression);
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor">"RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor"</a>: Disagreement predictor trained on SBIC text and individual annotator's demographic information in colon templated format;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/SChem_RoBERTa_Text_Disagreement_Binary_Classifier"</a>: Binary diagreement classifier trained on SChem text;
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/SChem_RoBERTa_Text_Disagreement_Predictor"</a>: Disagreement predictor trained on SChem text(regression);
- <a target="_blank" href="https://huggingface.co/RuyuanWan/SChem_RoBERTa_Demographic-text_Disagreement_Predictor">"RuyuanWan/SChem_RoBERTa_Demographic-text_Disagreement_Predictor"</a>: Disagreement predictor trained on Schem text and individual annotator's demographic information in colon templated format;
- <a target="_blank" href="RuyuanWan/Dilemmas_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/Dilemmas_RoBERTa_Text_Disagreement_Binary_Classifier"</a>: Binary diagreement classifier trained on Dilemmas text;
- <a target="_blank"  href="https://huggingface.co/RuyuanWan/Dilemmas_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/Dilemmas_RoBERTa_Text_Disagreement_Predictor"</a>:Disagreement predictor trained on Dilemmas text(regression);
- <a target="_blank" href="RuyuanWan/Dynasent_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/Dynasent_RoBERTa_Text_Disagreement_Binary_Classifier"</a>: Binary diagreement classifier trained on Dilemmas text;
- <a target="_blank"  href="https://huggingface.co/RuyuanWan/Dynasent_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/Dynasent_RoBERTa_Text_Disagreement_Predictor"</a>:Disagreement predictor trained on Dynasent text(regression);
- <a target="_blank" href="RuyuanWan/Politeness_RoBERTa_Text_Disagreement_Binary_Classifier">"RuyuanWan/Politeness_RoBERTa_Text_Disagreement_Binary_Classifier"</a>: Binary diagreement classifier trained on Politeness text;
- <a target="_blank"  href="https://huggingface.co/RuyuanWan/Politeness_RoBERTa_Text_Disagreement_Predictor">"RuyuanWan/Politeness_RoBERTa_Text_Disagreement_Predictor"</a>:Disagreement predictor trained on Politeness text(regression);
## Citation
