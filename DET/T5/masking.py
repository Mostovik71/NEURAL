#Code, which I used for mask random token in sentence with probability 0.15
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
model = T5ForConditionalGeneration.from_pretrained(raw_model)
tokenizer = T5Tokenizer.from_pretrained(raw_model)
text='Здесь может быть любой случайный текст'
inputs = tokenizer(text, return_tensors='pt')
print(inputs)
rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand <= 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
inputs.input_ids[0, selection] = 103 #Replacing selected tokens with tokens <MASK> (id=103)
print(inputs)
rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand <= 0.1) * (inputs.input_ids == 103)
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
inputs.input_ids[0, selection] = np.random.randint(0,30000)#Replacing masked tokens with probability 0.1

print(inputs)#MASKED SENTENCE
print(tokenizer.decode(inputs.input_ids[0],skip_special_tokens=True))
print(tokenizer.decode(model.generate(inputs.input_ids)[0],skip_special_tokens=True))
