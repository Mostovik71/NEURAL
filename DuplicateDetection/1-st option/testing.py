import pandas as pd
import numpy as np
import torch

import re
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
#Идеи - дубликаты
#modelnew - best(accuracy 0.89), dataset - new.xlsx
#modelnews
'''

columns=['id','Descr1','Descr2','is_duplicate']
df.columns=columns
df.drop(['id'],axis=1,inplace=True)
df.dropna(inplace=True)
df.to_csv('duplideas.csv')
'''



df=pd.read_excel('filmsbooksclean.xlsx')
DEVICE='cuda:0'
model = torch.load('modelnew.pth')
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
    predictions, predicted_labels = [], []

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
                       labels=labels))
        total_eval_loss += m.loss

        y_pred = np.argmax(m.logits.detach().cpu().numpy(), axis=1).flatten()
        total_eval_accuracy += metric(labels.cpu(), y_pred)

        predictions.extend(m.logits.detach().tolist())
        predicted_labels.extend(y_pred.tolist())

    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels
test = convert_to_dataset_torch(X_test, y_test)
test_dataloader = DataLoader(test,  sampler=SequentialSampler(test), batch_size=5)
model.eval()
_, _,predictions,predicted_labels = eval_batch(test_dataloader, model)
print(pd.concat([pd.Series(y_test),pd.Series(predicted_labels)],axis=1))
list1=[]
list0=[]
pred=pd.DataFrame(predictions)


for i in predictions:
    a=i[0]
    i[0]=np.exp(i[0])/(np.exp(i[0])+np.exp(i[1]))
    i[1]=np.exp(i[1])/(np.exp(a)+np.exp(i[1]))

df01=pd.concat([pd.DataFrame(predictions),pred,pd.Series(predicted_labels)],axis=1)
df01.columns=['not_duplicate','duplicate','pred0','pred1','predict']

df1=pd.concat([df,df01],axis=1)
df1=df1.sort_values('duplicate',ascending=False)
print(tabulate(df1,headers='keys'))
print(df1['predict'].value_counts())

df1.to_excel('predictionsbig(modelnew.pth).xlsx')
