# -*- coding: utf-8 -*-
"""Quantify_Annotation_Disagreement_Demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V-NC0DJ5q-7ePyuXhIgVumtRcRSl8-SD
"""

!pip install simpletransformers 
!pip datasets
!pip install lime

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lime
from lime import lime_text
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
import itertools
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

model_args = ClassificationArgs()
# metrics
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mse"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000
model_args.num_train_epochs = 15
model_args.regression = True
SBIC_person_demo_col_regression = ClassificationModel(
    "roberta",
    "RuyuanWan/SBIC_RoBERTa_Demographic-text_Disagreement_Predictor",
    num_labels=1,
    args=model_args
)

test = load_dataset("RuyuanWan/SBIC_Disagreement",split="test")
test = pd.DataFrame(test)

test['person_demo_col_txt'] = ["age: " + str(int(test['annotatorAge'].iloc[i])) + ', politics: ' + str(test['annotatorPolitics'].iloc[i]) + ', race: ' + str(test['annotatorRace'].iloc[i]) + ', gender: ' + str(test['annotatorGender'].iloc[i]) +'. ' + str(test['text'].iloc[i]) for i in range(len(test['text']))]
test_SBIC_person_demo_col_continuous = test[['person_demo_col_txt', 'disagreement_rate']]
test_SBIC_person_demo_col_continuous = test_SBIC_person_demo_col_continuous.rename(columns={'person_demo_col_txt': 'text', 'disagreement_rate': 'labels'})
SBIC_person_demo_col_continuous_model_outputs_result, SBIC_person_demo_col_continuous_model_outputs, SBIC_person_demo_col_continuous_wrong_predictions = SBIC_person_demo_col_regression.eval_model(test_SBIC_person_demo_col_continuous)
SBIC_person_demo_col_continuous_model_outputs_result

def predic_outputs(input):
  predict, raw_outputs = SBIC_person_demo_col_regression.predict(input)
  return np.tile(raw_outputs, (2,1)).T
  #return raw_outputs

class_names = [0,1]
explainer = LimeTextExplainer(class_names=class_names)

text_sample = "age: 22, politics: liberal, race: white, gender: woman. what is the most common name for a black dude king kong."
exp = explainer.explain_instance(text_sample, predic_outputs, num_features=6)
exp.show_in_notebook(text=True)

"""Simulation"""

age = ['age:18 -29, ', 'age: 30-39, ', 'age: 40-49, ','age: 50-59, ', 'age: 60 or elder, ']
race = ['race: white, ', 'race: black or african american, ','race: american indian or alaska native, ','race: asian, ','race: native hawaiian or other pacific islander, ', 'race: hispanic, ',
'race: some other race, ']
gender = ['gender: women.','gender: man.','gender: transgende.','gender: non-binary.']
simulated_demographics = []
simulated_demographics_text = []
simulated_demographics = [''.join([ c1, c2]) for c1, c2 in itertools.product(age,race)]
simulated_demographics = [''.join([ c1, c2]) for c1, c2 in itertools.product(simulated_demographics,gender)]
len(simulated_demographics)

simulated_demographics_text = [' '.join([c2, c1]) for c1, c2 in itertools.product(test_SBIC_person_demo_col_continuous['text'][:100],simulated_demographics)]
test_simulated_sample_predictions, test_simulated_sample_raw_outputs = SBIC_person_demo_col_regression.predict(simulated_demographics_text)

simulation_mean = []
simulation_var = []
for i in range(100):
  simulation_mean.append(statistics.mean(test_simulated_sample_predictions[140*i: 140*(i+1)]))
  simulation_var.append(statistics.variance(test_simulated_sample_predictions[140*i: 140*(i+1)]))

SBIC_disagreement_simulation = test_SBIC_person_demo_col_continuous.iloc[:100,]
SBIC_disagreement_simulation['simulation_var']=simulation_var
SBIC_disagreement_simulation['simulation_mean']=simulation_mean

sns.set(font_scale = 1.5)
p = sns.lmplot('simulation_var', 'simulation_mean', data=SBIC_disagreement_simulation, hue='labels', fit_reg=False).set(title = 'SBIC Simulation')
p.set_axis_labels("disagreement variance", "disagreement mean")
plt.show()
