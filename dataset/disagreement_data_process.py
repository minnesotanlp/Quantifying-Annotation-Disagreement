# -*- coding: utf-8 -*-
import pandas as pd
import re
from statistics import mode, mean
import math 
import numpy as np
from convokit import Corpus, download
"""#SBIC"""

SBIC_Train = pd.read_csv('/SBIC/SBIC.v2.trn.csv')
SBIC_Dev = pd.read_csv('/SBIC/SBIC.v2.dev.csv')
SBIC_Test = pd.read_csv('/SBIC/SBIC.v2.tst.csv')
frames = [SBIC_Train, SBIC_Dev, SBIC_Test]
SBIC_total = pd.concat(frames)

SBIC_total_Demographics = SBIC_total[['WorkerId','annotatorGender','annotatorPolitics','annotatorRace','annotatorAge']]
SBIC_Demographics = SBIC_total_Demographics.groupby('WorkerId')['annotatorGender','annotatorPolitics','annotatorRace','annotatorAge'].apply(lambda x: x.mode().iloc[0]).reset_index()
SBIC_Disagreement = SBIC_total[['post','WorkerId', 'annotatorGender', 'annotatorRace', 'annotatorAge','annotatorPolitics','offensiveYN']]
#remove id = -6837958490067487319, selected 33 times of 'man', 693 times of 'woman'.
SBIC_Disagreement = SBIC_Disagreement[SBIC_Disagreement.WorkerId != -6837958490067487319]

#preprocess text
SBIC_Disagreement['post'] = SBIC_Disagreement.post.apply(lambda x: re.sub(r'RT', '', x))
SBIC_Disagreement['post'] = SBIC_Disagreement.post.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
SBIC_Disagreement['post'] = SBIC_Disagreement.post.apply(lambda x: re.sub(r'http\S+', '', x))
# remove punctuation marks
punctuation = ',!"$%&()*+-/:;<=>?@[\\]^_`{|}~'
SBIC_Disagreement['post'] = SBIC_Disagreement['post'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
# convert text to lowercase
SBIC_Disagreement['post'] = SBIC_Disagreement['post'].str.lower()
SBIC_Disagreement = SBIC_Disagreement.rename(columns={"post": "text"})
SBIC_Disagreement = SBIC_Disagreement.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
SBIC_Disagreement = SBIC_Disagreement.replace(r'^\s*$', np.nan, regex=True)
SBIC_Disagreement = SBIC_Disagreement.dropna()

#major vote, group_disagreement, person_disagreement 
major_vote = SBIC_Disagreement.groupby(['text'], as_index = False)['offensiveYN'].agg(pd.Series.mode)
major_vote =  major_vote.rename(columns={"offensiveYN": "major_vote"})
annotators_count = SBIC_Disagreement.groupby(['text'], as_index = False).size()
annotators_count = annotators_count.rename(columns={"size": "annotators_count"})
text_label_count = SBIC_Disagreement.groupby(['text','offensiveYN'], as_index = False).size()
text_label_count = text_label_count.rename(columns={"size": "text_label_count"})
SBIC_Disagreement = pd.merge(SBIC_Disagreement, major_vote, on="text", how="left")
SBIC_Disagreement = pd.merge(SBIC_Disagreement, annotators_count, on="text", how="left")
SBIC_Disagreement = pd.merge(SBIC_Disagreement, text_label_count, on=["text",'offensiveYN'], how="left")
SBIC_Disagreement = SBIC_Disagreement[SBIC_Disagreement.annotators_count == 3]
SBIC_Disagreement['person_disagreement_rate'] = 1 - SBIC_Disagreement['text_label_count']/SBIC_Disagreement['annotators_count']
disagreement_rate = SBIC_Disagreement.groupby(['text'], as_index = False)['person_disagreement_rate'].agg(pd.Series.mode)
disagreement_rate = disagreement_rate.rename(columns={"person_disagreement_rate": "disagreement_rate"})
SBIC_Disagreement = pd.merge(SBIC_Disagreement, disagreement_rate, on="text", how="left")
SBIC_Disagreement['group_disagreement'] = [1 if SBIC_Disagreement['text_label_count'].iloc[x] /SBIC_Disagreement['annotators_count'].iloc[x]!=1 else 0 for x in range(len(SBIC_Disagreement))] 
SBIC_Disagreement['person_disagreement'] = [1 if SBIC_Disagreement['person_disagreement_rate'].iloc[x] != SBIC_Disagreement['disagreement_rate'].iloc[x]!=1 else 0 for x in range(len(SBIC_Disagreement))] 
#normalize disagreement rate
num_annotator = 3
most_controversial_rate = 1 - math.ceil(num_annotator/SBIC_Disagreement['offensiveYN'].nunique())/num_annotator
SBIC_Disagreement['normalized_disagreement_rate'] = SBIC_Disagreement['disagreement_rate'] / most_controversial_rate
SBIC_Disagreement['normalized_person_disagreement_rate'] = SBIC_Disagreement['person_disagreement_rate'] / most_controversial_rate

SBIC_disagreement = SBIC_Disagreement[['text','WorkerId', 'annotatorGender', 'annotatorRace', 'annotatorAge', 'annotatorPolitics','group_disagreement','normalized_disagreement_rate']]
SBIC_disagreement.rename(columns={'group_disagreement': 'binary_disagreement', 'normalized_disagreement_rate': 'disagreement_rate'}, inplace=True)

train_SBIC, validate_SBIC, test_SBIC = np.split(SBIC_disagreement.sample(frac=1, random_state=42),[int(.6*len(SBIC_disagreement)), int(.8*len(SBIC_disagreement))])

train_SBIC.to_csv('/SBIC_Disagreement/Train_SBIC_Disagreement.csv',index=False)
validate_SBIC.to_csv('/SBIC_Disagreement/Validate_SBIC_Disagreement.csv',index=False)
test_SBIC.to_csv('/SBIC_Disagreement/Test_SBIC_Disagreement.csv',index=False)

SBIC_Disagreement['group_demo'] = [" a " + str(int(SBIC_Disagreement['annotatorAge'].iloc[i])) + ' years old ' + str(SBIC_Disagreement['annotatorPolitics'].iloc[i]) + ' ' + str(SBIC_Disagreement['annotatorRace'].iloc[i]) + ' ' + str(SBIC_Disagreement['annotatorGender'].iloc[i])  for i in range(len(SBIC_Disagreement))]
SBIC_Disagreement['group_demo_sent'] = SBIC_Disagreement.groupby(['text'])['group_demo'].transform(lambda x:'The annotators are' + ', and'.join(x) + '. ')
SBIC_Disagreement['group_demo_sent_txt'] = SBIC_Disagreement['group_demo_sent'] + SBIC_Disagreement['text']
SBIC_Disagreement['demo_temp'] = ["age: " + str(int(SBIC_Disagreement['annotatorAge'].iloc[i])) + ', politics: ' + str(SBIC_Disagreement['annotatorPolitics'].iloc[i]) + ', race: ' + str(SBIC_Disagreement['annotatorRace'].iloc[i]) + ', gender: ' + str(SBIC_Disagreement['annotatorGender'].iloc[i]) +'. ' for i in range(len(SBIC_Disagreement))]
SBIC_Disagreement['group_demo_temp'] = SBIC_Disagreement.groupby(['text'])['demo_temp'].transform(lambda x: ''.join(x) )
SBIC_Disagreement['group_demo_temp_txt'] = SBIC_Disagreement['group_demo_temp'] + SBIC_Disagreement['text']
SBIC_Disagreement['person_demo_sent_txt'] = ["The annotator is a " + str(int(SBIC_Disagreement['annotatorAge'].iloc[i])) + ' years old ' + str(SBIC_Disagreement['annotatorPolitics'].iloc[i]) + ' ' + str(SBIC_Disagreement['annotatorRace'].iloc[i]) + ' ' + str(SBIC_Disagreement['annotatorGender'].iloc[i]) + ". " + str(SBIC_Disagreement['text'].iloc[i]) for i in range(len(SBIC_Disagreement))]
SBIC_Disagreement['person_demo_temp_txt'] = ["age: " + str(int(SBIC_Disagreement['annotatorAge'].iloc[i])) + ', politics: ' + str(SBIC_Disagreement['annotatorPolitics'].iloc[i]) + ', race: ' + str(SBIC_Disagreement['annotatorRace'].iloc[i]) + ', gender: ' + str(SBIC_Disagreement['annotatorGender'].iloc[i]) +'. ' + str(SBIC_Disagreement['text'].iloc[i]) for i in range(len(SBIC_Disagreement))]

"""#SChem"""

social_chem = pd.read_csv('/Social_Chemistry101/social-chem-101.v1.0.tsv',sep='\t').convert_dtypes()
demographic = pd.read_csv('/Social_Chemistry101/demographics_publish.csv')

social_chem_5 = social_chem[social_chem['m']==5 ]
agree_social_chem = social_chem_5[social_chem_5['m']==5][['m','rot-agree','rot','breakdown-worker-id']].dropna().drop_duplicates()

worker_demographic = demographic[['worker-id','gender','age', 'race','marital', 'economic', 'school', 'income', 'children', 'household','us', 'state', 'area', 'time-in-us']]
worker_demographic = worker_demographic.rename(columns = {'worker-id':'breakdown-worker-id'})
agree_social_chem = agree_social_chem.merge(worker_demographic, on=['breakdown-worker-id'], how='left')

agree_social_chem_count = agree_social_chem.groupby(['m','rot','rot-agree'],as_index=False).size()
agree_social_chem = agree_social_chem.merge(agree_social_chem_count, on=['m','rot','rot-agree'], how='left')

agree_social_chem = agree_social_chem.rename(columns={'rot': 'text','size': 'text_label_count'})
annotators_count = agree_social_chem.groupby(['text'], as_index = False).size()
annotators_count = annotators_count.rename(columns={"size": "annotators_count"})
agree_social_chem = pd.merge(agree_social_chem, annotators_count, on="text", how="left")
agree_social_chem = agree_social_chem[(agree_social_chem.m == 5) & (agree_social_chem.annotators_count == 5)]
agree_social_chem['group_disagreement'] = [1 if agree_social_chem['text_label_count'].iloc[x] /agree_social_chem['annotators_count'].iloc[x]!=1 else 0 for x in range(len(agree_social_chem))] 
agree_social_chem['person_disagreement_rate'] = 1 - agree_social_chem['text_label_count'] / agree_social_chem['annotators_count']
disagreement_rate = agree_social_chem.groupby(['text'], as_index = False)['person_disagreement_rate'].agg(pd.Series.mode)
disagreement_rate = disagreement_rate.rename(columns={"person_disagreement_rate": "disagreement_rate"})
agree_social_chem = pd.merge(agree_social_chem, disagreement_rate, on="text", how="left")
#normalize disagreement rate
agree_social_chem['normalized_disagreement_rate'] = [agree_social_chem['disagreement_rate'].iloc[x] / (1 - math.ceil(agree_social_chem['annotators_count'].iloc[x]/5)/agree_social_chem['annotators_count'].iloc[x]) for x in range(len(agree_social_chem))]
agree_social_chem['normalized_person_disagreement_rate'] = [agree_social_chem['person_disagreement_rate'].iloc[x] / (1 - math.ceil(agree_social_chem['annotators_count'].iloc[x]/5)/agree_social_chem['annotators_count'].iloc[x]) for x in range(len(agree_social_chem))]

schem_disagreement = agree_social_chem[['text','group_disagreement','normalized_disagreement_rate']]
schem_disagreement.rename(columns={'group_disagreement': 'binary_disagreement', 'normalized_disagreement_rate': 'disagreement_rate'}, inplace=True)
train_SChem, validate_SChem, test_SChem = np.split(schem_disagreement.sample(frac=1, random_state=42),[int(.6*len(schem_disagreement)), int(.8*len(schem_disagreement))])

train_SChem.to_csv('/SChem_Disagreement/Train_SChem_Disagreement.csv',index=False)
validate_SChem.to_csv('/SChem_Disagreement/Validate_SChem_Disagreement.csv',index=False)
test_SChem.to_csv('/SChem_Disagreement/Test_SChem_Disagreement.csv',index=False)

"""#Dilemmas"""

Train_dilemmas = pd.read_json('/dilemmas/train.scruples-dilemmas.jsonl', lines=True)
Dev_dilemmas = pd.read_json('/dilemmas/dev.scruples-dilemmas.jsonl', lines=True)
Test_dilemmas = pd.read_json('/dilemmas/test.scruples-dilemmas.jsonl', lines=True)

def read_dilemmas_json(data):
  action = data['actions'].to_list()
  action = pd.DataFrame(action)
  action = action.rename(columns ={0:'first', 1:'second'})
  first = action['first'].to_list()
  first =pd.DataFrame(first)
  first = first.rename(columns ={'id':'first-id', 'description':'first-description'})
  second = action['second'].to_list()
  second =pd.DataFrame(second)
  second = second.rename(columns ={'id':'second-id', 'description':'second-description'})
  Dilemmas = pd.concat([data,first,second],axis=1)
  Dilemmas['text'] = Dilemmas['first-description'] +'. '+ Dilemmas['second-description'] + '.'
  Dilemmas['controversial'] = list(map(int, Dilemmas['controversial']))
  Dilemmas.rename(columns={'controversial': 'binary_disagreement'}, inplace=True)
  Dilemmas['disagreement_rate'] = [Dilemmas['gold_annotations'][x][0]/5 if Dilemmas['gold_annotations'][x][0]< Dilemmas['gold_annotations'][x][1] else Dilemmas['gold_annotations'][x][1]/5 for x in range(len(Dilemmas))]
  #normalize disagreement rate
  num_annotator = 5
  most_controversial_rate = 1 - math.ceil(num_annotator/2)/num_annotator
  Dilemmas['disagreement_rate'] = Dilemmas['disagreement_rate'] / most_controversial_rate  
  output = Dilemmas[['text','binary_disagreement','disagreement_rate']]
  return output

train_dilemmas = read_dilemmas_json(Train_dilemmas)
dev_dilemmas = read_dilemmas_json(Dev_dilemmas)
test_dilemmas = read_dilemmas_json(Test_dilemmas)

train_dilemmas.to_csv('/Dilemmas_Disagreement/Train_Dilemmas_Disagreement.csv',index=False)
dev_dilemmas.to_csv('/Dilemmas_Disagreement/Dev_Dilemmas_Disagreement.csv',index=False)
test_dilemmas.to_csv('/Dilemmas_Disagreement/Test_Dilemmas_Disagreement.csv',index=False)

"""#DynaSent"""

dynasent_train = pd.read_json('/dynasent-v1.1/dynasent-v1.1-round01-yelp-train.jsonl', lines=True)
dynasent_dev = pd.read_json('/dynasent-v1.1/dynasent-v1.1-round01-yelp-dev.jsonl', lines=True)
dynasent_test = pd.read_json('/dynasent-v1.1/dynasent-v1.1-round01-yelp-test.jsonl', lines=True)

def read_dynasent_json(data):
  label_distribution = data['label_distribution'].to_list()
  label_distribution=pd.DataFrame(label_distribution)
  label_distribution['num_positive']=label_distribution['positive'].apply(lambda x: len(x))
  label_distribution['num_negative']=label_distribution['negative'].apply(lambda x: len(x))
  label_distribution['num_neutral']=label_distribution['neutral'].apply(lambda x: len(x))
  label_distribution['num_mixed']=label_distribution['mixed'].apply(lambda x: len(x))
  dynasent  = pd.concat([data,label_distribution],axis=1)
  dynasent['binary_disagreement'] = 1
  num_annotator = 5
  agreement_idx = [False if x==None else len(dynasent[x].iloc[idx])==num_annotator for idx,x in enumerate(dynasent['gold_label']) ]
  dynasent['binary_disagreement'].loc[agreement_idx] = 0
  dynasent['disagreement_rate'] = [1-2/num_annotator if x==None else 1-len(dynasent[x].iloc[idx])/num_annotator for idx,x in enumerate(dynasent['gold_label']) ]
  dynasent['disagreement_rate'] = [dynasent['disagreement_rate'].iloc[x] / (1 - 2/num_annotator) for x in range(len(dynasent))]
  return dynasent

train_dynasent = read_dynasent_json(dynasent_train)
dev_dynasent = read_dynasent_json(dynasent_dev)
test_dynasent = read_dynasent_json(dynasent_test)
dynasent_frames = [train_dynasent, dev_dynasent, test_dynasent]
dynasent_disagreement = pd.concat(dynasent_frames)
dynasent_disagreement.rename(columns={'sentence': 'text'}, inplace=True)
dynasent_disagreement = dynasent_disagreement[['text','binary_disagreement','disagreement_rate']]

train_dynasent, validate_dynasent, test_dynasent = np.split(dynasent_disagreement.sample(frac=1, random_state=42),[int(.6*len(dynasent_disagreement)), int(.8*len(dynasent_disagreement))])

train_dynasent.to_csv('/Dynasent_Disagreement/Train_Dynasent_Disagreement.csv',index=False)
validate_dynasent.to_csv('/Dynasent_Disagreement/Dev_Dynasent_Disagreement.csv',index=False)
test_dynasent.to_csv('/Dynasent_Disagreement/Test_Dynasent_Disagreement.csv',index=False)

"""#Politeness"""

corpus = Corpus(filename=download("wikipedia-politeness-corpus"))

utterances = corpus.get_utterances_dataframe()
utterances['annotation_distribution'] =[list(utterances['meta.Annotations'].iloc[x].values()) for x in range(len(utterances))]
utterances_binary = utterances[(utterances['meta.Binary'] == -1) | (utterances['meta.Binary']  == 1)]
utterances_binary['disagreement_rate'] = [ sum(i > 13 for i in x)/5 if mean(x)<=13 else sum(i <= 13 for i in x)/5 for x in utterances_binary['annotation_distribution']]
utterances_binary = utterances_binary[utterances_binary['disagreement_rate']!=0.6]
utterances_binary['disagreement_rate'] = utterances_binary['disagreement_rate']/(1-(math.ceil(5/2)/5))
utterances_binary['binary_disagreement'] = [0 if utterances_binary['disagreement_rate'].iloc[i] == 0 else 1 for i in range(len(utterances_binary))]

utterances_binary=utterances_binary[['text','binary_disagreement','disagreement_rate']]
train_polite, validate_polite, test_polite = np.split(utterances_binary.sample(frac=1, random_state=42),[int(.6*len(utterances_binary)), int(.8*len(utterances_binary))])

train_polite.to_csv('/Politeness_Disagreement/Train_Politeness_Disagreement.csv',index=False)
validate_polite.to_csv('/Politeness_Disagreement/Dev_Politeness_Disagreement.csv',index=False)
test_polite.to_csv('/Politeness_Disagreement/Test_Politeness_Disagreement.csv',index=False)
