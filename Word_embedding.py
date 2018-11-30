import numpy as np
import os 
from collections import Counter
import re
from math import exp,log
import json
from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from scipy.io import loadmat
import pandas as pd
from nltk import sent_tokenize
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')


def dictionary_maker(main_path):
	raw_vocabulary=[]
	for path in main_path:
		news_json=os.listdir(path)
		for file in news_json:	
			with open(os.path.join(path,file),'r',encoding='utf-8') as file:
				
				data=json.load(file)
				text=data['text']
				tokens_temp=word_tokenize(text)
				tokens=[word.lower() for word in tokens_temp]
				table=str.maketrans("","",string.punctuation)
				stripped=[w.translate(table) for w in tokens]
				words_temp=[word for word in stripped if word.isalpha()]
				words=[w for w in words_temp if w not in stop_words]
				words=[w for w in words if len(w)>2]
				raw_vocabulary+=words

	vocabulary=Counter(raw_vocabulary)
	vocabulary=vocabulary.most_common()
	print(vocabulary)




stop_words=stopwords.words('english')




path=os.getcwd()

#dictionary_maker(main_path)

data=pd.read_csv(os.path.join(path,"real_fake_news.csv"))
data1=data[pd.notnull(data["text"])]
labels_temp,text=data1['label'].values,data1['text'].values


labels=[int(label) for label in labels_temp]

tagged_data=[TaggedDocument(words=word_tokenize(w.lower()),tags=[str(i)]) for i,w in enumerate(text)]

max_epochs=100
vec_size=100
alpha=0.05

model=Doc2Vec(size=vec_size,alpha=alpha,min_alpha=0.00025,min_count=5,dm=1)

model.build_vocab(tagged_data)


for epoch in range(max_epochs):
	print("iteration {0}".format(epoch))
	model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
	model.alpha-=0.0002


model.save('Doc2Vec.model')
print("Model Saved")
