import pandas as pd
import re
import string
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import torch.cuda
questions_dataset = pd.read_excel("newnodupl.xlsx")
#questions_dataset=pd.read_csv('quora_question_pairs_rus.csv').sample(10000)
#questions_dataset=questions_dataset.sample(50000)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
questions_dataset.dropna(inplace=True)
DEVICE='cuda:0'
X_train, X_validation, y_train, y_validation = train_test_split(questions_dataset[["Descr1", "Descr2"]],
                                                    questions_dataset["is_duplicate"], test_size=0.2, random_state=41)

max_length=512
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset


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


train = convert_to_dataset_torch(X_train, y_train)
validation = convert_to_dataset_torch(X_validation, y_validation)


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


batch_size = 4




train_dataloader = DataLoader(
            train,

            batch_size = batch_size,
            num_workers = 0,
            sampler = RandomSampler(train),
            drop_last=True
        )


validation_dataloader = DataLoader(
            validation,

            batch_size = batch_size,
            num_workers = 0,
            sampler = SequentialSampler(validation),
            drop_last=True
        )
from transformers import BertForSequenceClassification




bert_model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased",num_labels=2,output_attentions=False,output_hidden_states=True)
#bert_model=torch.load('modelnew.pth')
bert_model.to(DEVICE)

from transformers import AdamW




adamw_optimizer = AdamW(bert_model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

from transformers import get_linear_schedule_with_warmup


epochs = 10


total_steps = len(train_dataloader) * epochs


scheduler = get_linear_schedule_with_warmup(adamw_optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0
    total_train_accuracy=0

    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):
        
        input_ids1, attention_masks1, token_type_ids1, labels1,input_ids2, attention_masks2, token_type_ids2, labels2 = batch

        model.zero_grad()

        input_ids1 = input_ids1.to(DEVICE)
        token_type_ids1 = token_type_ids1.to(DEVICE)
        attention_masks1 = attention_masks1.to(DEVICE)
        labels1 = labels1.long()
        labels1 = labels1.to(DEVICE)
        input_ids2 = input_ids2.to(DEVICE)
        token_type_ids2 = token_type_ids2.to(DEVICE)
        attention_masks2 = attention_masks2.to(DEVICE)

        

        logits1 = (model(input_ids=input_ids1,
                      token_type_ids=token_type_ids1,
                      attention_mask=attention_masks1
                      )).hidden_states[12][:, 0, :]




        logits2 = (model(input_ids=input_ids2,
                      token_type_ids=token_type_ids2,
                      attention_mask=attention_masks2
                      )).hidden_states[12][:, 0, :]

        labels1[labels1 == 0] = -1
        loss = CosineEmbeddingLoss()
        loss = loss(logits1, logits2, labels1)
        #print(loss)


        total_train_loss += loss.detach()
        total_train_accuracy += ((cosine_similarity(logits1.detach().cpu().numpy(), logits2.detach().cpu().numpy()) > 0.5) == (labels1.cpu().numpy() > 0)).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        optimizer.step()

        scheduler.step()

    return total_train_loss,total_train_accuracy


import numpy

from sklearn.metrics import accuracy_score



def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions, predicted_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
     
        input_ids1, attention_masks1, token_type_ids1, labels1, input_ids2, attention_masks2, token_type_ids2, labels2 = batch
        model.cuda()

      
        input_ids1 = input_ids1.to(DEVICE, dtype=torch.long)
        token_type_ids1 = token_type_ids1.to(DEVICE, dtype=torch.long)
        attention_masks1 = attention_masks1.to(DEVICE, dtype=torch.long)
        labels1 = labels1.to(DEVICE, dtype=torch.long)
        input_ids2 = input_ids2.to(DEVICE, dtype=torch.long)
        token_type_ids2 = token_type_ids2.to(DEVICE, dtype=torch.long)
        attention_masks2 = attention_masks2.to(DEVICE, dtype=torch.long)

        with torch.no_grad():
            
            m1 = (model(input_ids1,
                       token_type_ids=token_type_ids1,
                       attention_mask=attention_masks1
                       ))
            m2 = (model(input_ids2,
                        token_type_ids=token_type_ids2,
                        attention_mask=attention_masks2
                        ))
            stop=1
            logits1 = m1.hidden_states[12][:, 0, :]
            logits2 = m2.hidden_states[12][:, 0, :]
            labels1[labels1 == 0] = -1
            loss = CosineEmbeddingLoss(reduction='mean')
            loss = loss(logits1, logits2, labels1)

        total_eval_loss += loss

        total_eval_accuracy += ((cosine_similarity(logits1.cpu().numpy(), logits2.cpu().numpy()) > 0.5) == (labels1.cpu().numpy() > 0)).mean()
    stop=2
    return total_eval_accuracy, total_eval_loss


import random


seed_val = 42
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)


def train(train_dataloader, validation_dataloader, model, optimizer, epochs):
   
    training_stats = []

   
    total_t0 = time.time()

    for epoch in range(0, epochs):
       
        t0 = time.time()

        total_train_loss = 0



        model.train()
        total_train_loss, total_train_accuracy = fit_batch(train_dataloader, model, optimizer, epoch)
        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)
        t0 = time.time()


        model.eval()


        total_eval_accuracy, total_eval_loss = eval_batch(validation_dataloader, model)
        FILE = 'modelnew2.pth'
        torch.save(model, FILE)


        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)

        print(f"  Val Accuracy: {avg_val_accuracy}")
        print(f"  Train Accuracy: {avg_train_accuracy}")


        avg_val_loss = total_eval_loss / len(validation_dataloader)


        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {avg_val_loss}")
        #print(f"  Train Loss: {avg_train_loss}")


        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print(f"Total training took {format_time(time.time() - total_t0)}")
    return training_stats
training_stats = train(train_dataloader, validation_dataloader, bert_model, adamw_optimizer, epochs)

