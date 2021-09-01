import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification
import re
import pickle
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
df.to_csv()
'''


df=pd.read_excel('filmsbooksclean2.xlsx')
#df['is_duplicate'].fillna(0,inplace=True)

df.dropna(inplace=True)
#df['is_duplicate']=pd.Series()
DEVICE='cuda:0'
#model = torch.load('modelGOOD.pth')
model = torch.load('modelwithoutstrangehidden.pth')
#model=BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased-sentence",num_labels=2, output_attentions=False,output_hidden_states=False)
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
df1.to_excel('forlog.xlsx')
df1.reset_index(drop=True,inplace=True)
with open('logreg.pkl', 'rb') as file:
 lr = pickle.load(file)
df1['duplicate']=pd.Series(lr.predict_proba(df1)[:,1])
#df1['duplicate']=pd.Series(lr.predict_proba(df)[:,1])

#df1.to_excel('bertwithlog.xlsx')




#print(tabulate(df1.head(20),headers='keys'))
#df1=pd.DataFrame(embeddings)



#df1.to_excel('medium.xlsx')
