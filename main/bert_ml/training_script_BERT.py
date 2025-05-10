'''
Update on 11 May 2025

@author: pasquale
'''
from os import name as nameos
from pathlib import Path
from transformers import BertTokenizer, BertModel
from bitsandbytes.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from BERT_arch import BERT_Arch
import torch
import pandas as pd
#begin standard code to import BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')
#end standard code to import BERT Model

#START parameters
# Define Max lenght tokens MAX 512
MAX_LENGHT = 512
# Define batch size
batch_size = 8
# Defining the hyperparameters
# Constructor of hometrained Model
model_def = BERT_Arch(model)
# Define the optimizer
optimizer = AdamW(model_def.parameters(), lr = 2e-5)
# Define Number of training epochs (default 5)
epochs = 5
#END define paramenters

def __getPathTrainingData():
    #mappa
    label_map = {'Fake': 1, 
                 'True': 0}
    if nameos == 'nt':
        path_file = str(Path.cwd()) + '\\main\\bert_ml\\model_prof.tsv'
    else:
        path_file = str(Path.cwd()) + '/model_prof.tsv'

    data = pd.read_csv(path_file, sep='\t', names=['surname', 'Target', 'topic'])
    data['label'] = data['Target'].replace(label_map)
    return data

data = __getPathTrainingData()

# Train-temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['topic'], data['label'],
                                                        random_state=2018,
                                                        test_size=0.7,
                                                        stratify=data['label'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                    random_state=2018,
                                                    test_size=0.5,
                                                    stratify=temp_labels)

# Tokenize and encode sequences in the train set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    padding = 'max_length',
    truncation=True,
    return_attention_mask=True
)
# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    padding = 'max_length',
    truncation=True,
    return_attention_mask=True
)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    padding = 'max_length',
    truncation=True,
    return_attention_mask=True
)

# Conversione in tensori con dtype corretto
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist(), dtype=torch.long)

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist(), dtype=torch.long)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist(), dtype=torch.long)

# Creazione del dataset per l'addestramento e la valutazione
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Creazione del DataLoader per il test set
test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

def print_data_distribution(train_text, val_text, test_text):
    print("Data distribution:\n")
    print(f"Training data   : {len(train_text)} examples")
    print(f"Validation data : {len(val_text)}   examples")
    print(f"Test data       : {len(test_text)}  examples\n")

print_data_distribution(train_text, val_text, test_text)

def train(loss_fn):
    model_def.set_classifier_trainable()
    model_def.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = [r for r in batch] 
        sent_id, mask, labels = batch
        #ids =0 att mask =1 label =2
        optimizer.zero_grad()
        pred = model_def(sent_id, mask)
        loss = loss_fn(pred, labels)
        loss.requires_grad_(True)
        optimizer.step()
        train_last_loss = loss.item()
        total_loss = total_loss + train_last_loss
        print(f'Training batch {step + 1} last loss: {loss.item():.4f}', flush=True)
    avg_loss = total_loss / len(train_dataloader)
    print(f"\nTraining epoch {epoch + 1} -> avg loss: ", avg_loss)
    return avg_loss

def evaluate(loss_fn):
    print("\nEvaluating...")
    model_def.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            sent_id, mask, labels = batch
            logits = model_def(sent_id, mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            print(predicted)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f'Evaluating batch {step + 1} last loss: {loss.item():.4f}', flush=True)
    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct / total
    print(f"\nValidation Accuracy: {accuracy:.4f} -> avg Loss: {avg_loss:.4f}")
    return avg_loss
  
# Train and predict
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]                   
valid_losses=[]
for epoch in range(epochs):
    # Define the loss function
    loss_fn  = torch.nn.CrossEntropyLoss()
    print(f'\nTraining {epoch + 1} di {epochs}', flush=True)
    train_loss = train(loss_fn)
    valid_loss = evaluate(loss_fn)
    print(f'\nBest Valid Loss {best_valid_loss} Actual loss {valid_loss}')
    if valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model_def.state_dict(), 'unclepask_propaganda.pt')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
