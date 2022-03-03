import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
from tqdm.auto import tqdm, trange
from transformers import T5ForConditionalGeneration, T5Tokenizer
import string
df=pd.read_csv('cae-t5train.csv')
raw_model = 'cointegrated/rut5-base-multitask'
model = T5ForConditionalGeneration.from_pretrained(raw_model,output_hidden_states=True)
tokenizer = T5Tokenizer.from_pretrained(raw_model)
text='Мама мыла раму мочалкой во время путешествия на Марс'
attr='civil: '
gam=attr+text
inputs = tokenizer(text, return_tensors='pt')
decinp = tokenizer(attr, return_tensors='pt')
#print(inputs)
rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand <= 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
inputs.input_ids[0, selection] = 103#Заменяем выбранные токены токенами <MASK> (id=103)
#print(inputs)
rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand <= 0.1) * (inputs.input_ids == 103)
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
inputs.input_ids[0, selection] = np.random.randint(0,30000)
conc=torch.cat((inputs.input_ids,decinp.input_ids),1)

