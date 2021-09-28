import pandas as pd

import torch

import pickle

from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-sentence', do_lower_case=True)
from sklearn.metrics import accuracy_score

from tabulate import tabulate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
#df=pd.read_excel('newwithoutstrange.xlsx')
df=pd.read_excel()
#df.dropna(inplace=True)
#df.drop_duplicates(inplace=True)
#print(df['is_duplicate'].value_counts())
DEVICE='cuda:0'
#df.drop_duplicates(inplace=True)
#df.dropna(inplace=True)
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased-sentence",num_labels=2,output_attentions=False,output_hidden_states=True)
X_test=df[['Descr1','Descr2']]
y_test=df['is_duplicate']
max_length=512
def convert_to_dataset_torch(data: pd.DataFrame, labels: pd.Series) -> TensorDataset:
    input_ids1 = []
    attention_masks1 = []
    token_type_ids1 = []
    input_ids2 = []
    attention_masks2 = []
    token_type_ids2 = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded1 = tokenizer.encode_plus(row["Descr1"], max_length=max_length,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt', truncation=True)
        input_ids1.append(encoded1['input_ids'])
        token_type_ids1.append(encoded1["token_type_ids"])
        attention_masks1.append(encoded1['attention_mask'])



        encoded2 = tokenizer.encode_plus(row["Descr2"], max_length=max_length,
                                         pad_to_max_length=True,
                                         return_attention_mask=True, return_tensors='pt', truncation=True)
        input_ids2.append(encoded2['input_ids'])
        token_type_ids2.append(encoded2["token_type_ids"])
        attention_masks2.append(encoded2['attention_mask'])



    input_ids1 = torch.cat(input_ids1, dim=0)
    token_type_ids1 = torch.cat(token_type_ids1, dim=0)
    attention_masks1 = torch.cat(attention_masks1, dim=0)
    labels1 = torch.tensor(labels.values)
    input_ids1.to(DEVICE, dtype=torch.long)
    token_type_ids1.to(DEVICE, dtype=torch.long)
    attention_masks1.to(DEVICE, dtype=torch.long)
    labels1.to(DEVICE, dtype=torch.long)

    input_ids2 = torch.cat(input_ids2, dim=0)
    token_type_ids2 = torch.cat(token_type_ids2, dim=0)
    attention_masks2 = torch.cat(attention_masks2, dim=0)
    labels2 = torch.tensor(labels.values)
    input_ids2.to(DEVICE, dtype=torch.long)
    token_type_ids2.to(DEVICE, dtype=torch.long)
    attention_masks2.to(DEVICE, dtype=torch.long)
    labels2.to(DEVICE, dtype=torch.long)

    return TensorDataset(input_ids1, attention_masks1, token_type_ids1, labels1,input_ids2, attention_masks2, token_type_ids2, labels2)
def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    embs1, embs2 = [], []
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        # Unpack batch from dataloader.
        input_ids1, attention_masks1, token_type_ids1, labels1, input_ids2, attention_masks2, token_type_ids2, labels2 = batch
        model.cuda()
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        input_ids1 = input_ids1.to(DEVICE, dtype=torch.long)
        token_type_ids1 = token_type_ids1.to(DEVICE, dtype=torch.long)
        attention_masks1 = attention_masks1.to(DEVICE, dtype=torch.long)
        labels1 = labels1.to(DEVICE, dtype=torch.long)
        input_ids2 = input_ids2.to(DEVICE, dtype=torch.long)
        token_type_ids2 = token_type_ids2.to(DEVICE, dtype=torch.long)
        attention_masks2 = attention_masks2.to(DEVICE, dtype=torch.long)
        labels2 = labels2.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            m1 = (model(input_ids1,
                       token_type_ids=token_type_ids1,
                       attention_mask=attention_masks1,
                       labels=labels1)).hidden_states[12][:, 0, :]
            m2 = (model(input_ids2,
                        token_type_ids=token_type_ids2,
                        attention_mask=attention_masks2,
                        labels=labels2)).hidden_states[12][:, 0, :]
            stop=1
        embs1.append(m1.cpu().numpy())
        embs2.append(m2.cpu().numpy())
    return embs1,embs2
test = convert_to_dataset_torch(X_test, y_test)
test_dataloader = DataLoader(test,  sampler=SequentialSampler(test), batch_size=1)
model.eval()
m1, m2 = eval_batch(test_dataloader, model)
df1=pd.DataFrame([])
df2=pd.DataFrame([])
for i, k in zip(m1, m2):
    df1 = df1.append(pd.DataFrame(i[0]).T)
    df2 = df2.append(pd.DataFrame(k[0]).T)
data=pd.concat([df1, df2], axis=1)
data['is_duplicate']=y_test
data.to_excel('embeddings.xlsx')
