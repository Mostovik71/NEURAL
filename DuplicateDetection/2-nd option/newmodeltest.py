import pandas as pd
import numpy as np
import torch
from scipy.spatial import distance
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-sentence', do_lower_case=True)
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
df.to_csv()
'''

#from transformers import BertForSequenceClassification
df=pd.read_excel('filmsbooksclean2.xlsx')
DEVICE='cuda:0'
model = SentenceTransformer("sentence-transformers/LaBSE")
#model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased-sentence",num_labels=2,output_attentions=False,output_hidden_states=True)
#model = torch.load('modelnew.pth')
model.to(DEVICE)
from tqdm import tqdm
from torch.utils.data import TensorDataset
X_test=df[['idea1','idea2']]
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
def cosine_sim_vectors(vec1,vec2):
    vec1=vec1.reshape(1,-1)
    vec2=vec2.reshape(1,-1)
    return cosine_similarity(vec1,vec2)[0][0]
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
                       labels=labels1)).hidden_states[12][:,0]
            m2 = (model(input_ids2,
                        token_type_ids=token_type_ids2,
                        attention_mask=attention_masks2,
                        labels=labels2)).hidden_states[12][:,0]
            stop=1
        embs1.append(m1.cpu().numpy())
        embs2.append(m2.cpu().numpy())
    return embs1,embs2

test = convert_to_dataset_torch(X_test, y_test)
test_dataloader = DataLoader(test,  sampler=SequentialSampler(test), batch_size=4)
model.eval()
m1,m2= eval_batch(test_dataloader, model)
cosine=[]
for i,j in zip(m1,m2):
    for k,l in zip(i,j):
        print(cosine_sim_vectors(k,l))
        cosine.append(cosine_sim_vectors(k,l))

df['cosinesim']=pd.Series(cosine)
print(df[['is_duplicate','cosinesim']])
#df.to_excel('cosineshort.xlsx')

'''
print(pd.concat([pd.Series(y_test),pd.Series(predict)],axis=1))
list1=[]
list0=[]
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
'''
#df1.to_excel('predictionsdirtybig.xlsx')
