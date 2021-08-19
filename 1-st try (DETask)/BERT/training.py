import pandas as pd
import re
import string
import torch.cuda
df = pd.read_excel("new.xlsx")

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
df.dropna(inplace=True)
DEVICE='cuda:0'
X_train, X_validation, y_train, y_validation = train_test_split(df[["Descr1", "Descr2"]],
                                                    df["is_duplicate"], test_size=0.2, random_state=42)

max_length=512
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset


def convert_to_dataset_torch(data: pd.DataFrame, labels: pd.Series) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["Descr1"], row["Descr2"], max_length=max_length,
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


train = convert_to_dataset_torch(X_train, y_train)
validation = convert_to_dataset_torch(X_validation, y_validation)


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


batch_size = 5




train_dataloader = DataLoader(
            train,  
            sampler = RandomSampler(train),
            batch_size = batch_size,
            num_workers = 0,
            drop_last=True
        )


validation_dataloader = DataLoader(
            validation, 
            sampler = SequentialSampler(validation), 
            batch_size = batch_size, 
            num_workers = 0,
            drop_last=True
        )
from transformers import BertForSequenceClassification



bert_model = BertForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", 
    num_labels=2, 
    output_attentions=False,
    output_hidden_states=False, 
)
bert_model.to(DEVICE)
from transformers import AdamW




adamw_optimizer = AdamW(bert_model.parameters(), lr = 2e-5, eps = 1e-8)
from transformers import get_linear_schedule_with_warmup
epochs = 1
total_steps = len(train_dataloader) * epochs


scheduler = get_linear_schedule_with_warmup(adamw_optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
import time
import datetime


def format_time(elapsed):
   
   
    elapsed_rounded = int(round((elapsed)))

   
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0

    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):
      
        input_ids, attention_masks, token_type_ids, labels = batch

        model.zero_grad()
        input_ids = input_ids.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        labels = labels.long()
        labels = labels.to(DEVICE)

        loss = (model(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_masks,
                      labels=labels)).loss

        total_train_loss += loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    return total_train_loss


import numpy

from sklearn.metrics import accuracy_score


def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions, predicted_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
   
        input_ids, attention_masks, token_type_ids, labels = batch
        model.cuda()

        input_ids = input_ids.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        attention_masks = attention_masks.to(DEVICE, dtype=torch.long)
        labels = labels.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
           
            m = (model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_masks,
                       labels=labels))
        total_eval_loss += m.loss

        y_pred = numpy.argmax(m.logits.detach().cpu().numpy(), axis=1).flatten()
        total_eval_accuracy += metric(labels.cpu(), y_pred)

        predictions.extend(m.logits.detach().tolist())
        predicted_labels.extend(y_pred.tolist())

    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels


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

        total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)

        avg_train_loss = total_train_loss / len(train_dataloader)

       
        training_time = format_time(time.time() - t0)

        t0 = time.time()

   
        model.eval()

        total_eval_accuracy, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
        FILE = 'modelnew.pth'
        torch.save(bert_model, FILE)
       
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print(f"  Accuracy: {avg_val_accuracy}")

      
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {avg_val_loss}")

     
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

