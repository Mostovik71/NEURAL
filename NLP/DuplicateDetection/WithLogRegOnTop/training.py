import pandas as pd
import re
import string
import torch.cuda
#questions_dataset = pd.read_excel("newnodupl.xlsx")
#questions_dataset = pd.read_excel('newwithoutstrange.xlsx')
questions_dataset = pd.read_excel('data.xlsx')


#questions_dataset=questions_dataset.sample(50000)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
questions_dataset.dropna(inplace=True)
DEVICE='cuda:0'
X_train, X_validation, y_train, y_validation = train_test_split(questions_dataset[["Descr1", "Descr2"]],
                                                    questions_dataset["is_duplicate"], test_size=0.2, random_state=42)

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
import multiprocessing

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it
# here.
batch_size = 8

core_number = multiprocessing.cpu_count()

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train,  # The training samples.
            sampler = RandomSampler(train), # Select batches randomly
            batch_size = batch_size, # Trains with this batch size.
            num_workers = 0,
            drop_last=True
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            validation, # The validation samples.
            sampler = SequentialSampler(validation), # Pull out batches sequentially.
            batch_size = batch_size, # Evaluate with this batch size.
            num_workers = 0,
            drop_last=True
        )
from transformers import BertForSequenceClassification



# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
bert_model = BertForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions=False, # Whether the model returns attentions weights.
    output_hidden_states=True, # Whether the model returns all hidden-states.
)
bert_model.to(DEVICE)
from transformers import AdamW



# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
adamw_optimizer = AdamW(bert_model.parameters(),
                  lr = 1e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(adamw_optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0

    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):
        # Unpack batch from dataloader.
        input_ids, attention_masks, token_type_ids, labels = batch

        model.zero_grad()
        input_ids = input_ids.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        labels = labels.long()
        labels = labels.to(DEVICE)
        # Perform a forward pass (evaluate the model on this training batch).
        loss = (model(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_masks,
                      labels=labels)).loss

        total_train_loss += loss
        #total_train_accuracy +=
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
        # Unpack batch from dataloader.
        input_ids, attention_masks, token_type_ids, labels = batch
        model.cuda()
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

        y_pred = numpy.argmax(m.logits.detach().cpu().numpy(), axis=1).flatten()
        total_eval_accuracy += metric(labels.cpu(), y_pred)

        predictions.extend(m.logits.detach().tolist())
        predicted_labels.extend(y_pred.tolist())

    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels


import random

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)


def train(train_dataloader, validation_dataloader, model, optimizer, epochs):
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    for epoch in range(0, epochs):
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.

        model.train()

        total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        total_eval_accuracy, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
        FILE = 'modelwithoutstrangenormalize.pth'
        torch.save(model, FILE)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Accuracy: {avg_val_accuracy}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)


        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print(f"  Train Loss: {avg_train_loss}")
        print(f"  Validation Loss: {avg_val_loss}")

        # Record all statistics from this epoch.
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

