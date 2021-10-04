import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification
import re
import pickle
import string
import lime
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#modelnew - best(accuracy 0.89), dataset - new.xlsx
#modelnews
'''

columns=['id','Descr1','Descr2','is_duplicate']
df.columns=columns
df.drop(['id'],axis=1,inplace=True)
df.dropna(inplace=True)
df.to_csv('duplideas.csv')
'''

#df=pd.read_excel('duplideasclean.xlsx')
#df=pd.read_excel('filmsbooksclean2.xlsx')
df=pd.read_excel('normalizedideasnumber.xlsx')
#df=pd.read_excel('normalizedideasx.xlsx')
#df=pd.read_excel('ideasx2109.xlsx')
#df=pd.read_excel('ideasxnew.xlsx')


print("Введите строку: ")
str = "\n".join(iter(input, ""))

df['Descr2'] = pd.Series()
df['Descr2'].fillna(str,inplace=True)
df["Descr2"]=df["Descr2"].apply(lambda x: x.lower())
df["Descr2"]=df["Descr2"].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation),' ', x))
df['is_duplicate']=pd.Series()

#From file

df.drop_duplicates(inplace=True)

DEVICE='cuda:0'

model = torch.load('modelwithoutstrangehidden.pth')
#model=BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased-sentence",num_labels=2, output_attentions=False,output_hidden_states=True)
model.to(DEVICE)
from tqdm import tqdm
from torch.utils.data import TensorDataset

X_test=df[['Descr1','Descr2']]
y_test=df['is_duplicate']
def convert_to_dataset_torch(data: pd.DataFrame, labels: pd.Series) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["Descr1"], row["Descr2"], max_length=512,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt', truncation=True)
        # Add the encoded sentences to the list.
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)
    input_ids.to(DEVICE, dtype=torch.long)
    token_type_ids.to(DEVICE, dtype=torch.long)
    attention_masks.to(DEVICE, dtype=torch.long)
    labels.to(DEVICE, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, token_type_ids, labels)


def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    embs=[]

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        # Unpack batch from dataloader.
        input_ids, attention_masks, token_type_ids, labels = batch

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        input_ids = input_ids.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        attention_masks = attention_masks.to(DEVICE, dtype=torch.long)
        labels = labels.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            m = (model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_masks,
                       labels=labels)).hidden_states[12][:,0,:]

            embs.append(m.detach().cpu().numpy())



    return embs
test = convert_to_dataset_torch(X_test, y_test)
test_dataloader = DataLoader(test,  sampler=SequentialSampler(test), batch_size=1)
model.eval()
embeddings = eval_batch(test_dataloader, model)
df1=pd.DataFrame([])
for i,k in enumerate(embeddings):
    #df1=df1.append(pd.concat([pd.DataFrame(k[0]).T,pd.Series(df['is_duplicate'].iloc[i])],axis=1))
     df1=df1.append(pd.DataFrame(k[0]).T)
#df1.to_excel('forlog.xlsx')
df1.reset_index(drop=True,inplace=True)
with open('logreg512.pkl', 'rb') as file:
 lr = pickle.load(file)

df1['duplicate']=pd.Series(lr.predict_proba(df1)[:,1])
df1['is_duplicate']=df['is_duplicate']
df1['Descr1']=df['Descr2']
df1['Descr2']=df['Descr1']

df2=df1[['Descr1','Descr2','duplicate']]
#df1['duplicate']=pd.Series(lr.predict_proba(df)[:,1])
#df2.dropna(inplace=True)
df2.sort_values('duplicate',ascending=False,inplace=True)
print(tabulate(df2, headers='keys'))
df2.to_excel('descrnormalized116(1).xlsx')
#logreg-21/24
#logreg512-21.5/24

