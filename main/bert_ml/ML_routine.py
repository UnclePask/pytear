#begin standard code to import
from transformers import pipeline
from thinc.compat import torch
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")
[
   {
      "sequence":"[CLS] Hello I'm a male model. [SEP]",
      "score":0.22748498618602753,
      "token":2581,
      "token_str":"male"
   },
   {
      "sequence":"[CLS] Hello I'm a fashion model. [SEP]",
      "score":0.09146175533533096,
      "token":4633,
      "token_str":"fashion"
   },
   {
      "sequence":"[CLS] Hello I'm a new model. [SEP]",
      "score":0.05823173746466637,
      "token":1207,
      "token_str":"new"
   },
   {
      "sequence":"[CLS] Hello I'm a super model. [SEP]",
      "score":0.04488750174641609,
      "token":7688,
      "token_str":"super"
   },
   {
      "sequence":"[CLS] Hello I'm a famous model. [SEP]",
      "score":0.03271442651748657,
      "token":2505,
      "token_str":"famous"
   }
]

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)

#end standard code to import

import pandas as pd

label_map = {'Fake': 1, 
             'True': 0}

data = pd.read_csv('model_prof.tsv', sep='\t', names=['surname', 'Target', 'topic'])
data['label'] = data['Target'].replace(label_map)
print('debug data')

from sklearn.model_selection import train_test_split

# Train-temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['topic'], data['label'],
                                                        random_state=2018,
                                                        test_size=0.7,
                                                        stratify=data['Target'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                    random_state=2018,
                                                    test_size=0.8,
                                                    stratify=temp_labels)
# Max tokens
MAX_LENGHT = 500
# Tokenize and encode sequences in the train set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#define a batch size
batch_size = 32                                               

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train_data = TensorDataset(train_seq, train_mask, train_y)    
train_sampler = RandomSampler(train_data)                     
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# Freezing the parameters and defining trainable BERT structure
for param in model.parameters():
    param.requires_grad = False 
print('debug')

# Defining the hyperparameters (optimizer, weights of the classes and the epochs)
# Define the optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr = 2e-5) 

import torch.nn as nn
# Define the loss function
loss_fn  = nn.CrossEntropyLoss()
# Number of training epochs
epochs = 5

# Define architecture
from bert_ml.BERT_arch import BERT_Arch
model_def = BERT_Arch(model)
# Defining training and evaluation functions
def train():
#    model.enable_input_require_grads()
    model_def.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
#        if step == 3:
#            break
        batch = [r for r in batch]
        optimizer.zero_grad()
        outputs = model(input_ids = batch[0], attention_mask = batch[1])
        pred = outputs[1]
        loss = loss_fn(pred, batch[2])
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        # Calculating the running loss for logging purposes
        train_batch_loss = loss.item()
        train_last_loss = train_batch_loss / batch_size
        print('Training batch {} last loss: {}'.format(step + 1, train_last_loss), flush=True)
    
    # Logging epoch-wise training loss
    print(f"\nTraining epoch {epoch + 1} loss: ", train_last_loss) 
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate():
    print("\nEvaluating...")
    model_def.eval()                                    # Deactivate dropout layers
    correct = 0
    for step,batch in enumerate(val_dataloader):
        batch = [t for t in batch]
        # We don't need gradients for testing
        with torch.no_grad():
            outputs = model(input_ids = batch[0], attention_mask = batch[1])   # Push the batch to GPU
        
        # Calculating total batch loss using the logits and labels
        logits = outputs[1]
        loss = loss_fn(logits, batch[2])
        loss.requires_grad_(True)
        test_batch_loss = loss.item()
        
        test_last_loss = test_batch_loss / batch_size
        print('Testing batch {} loss: {}'.format(step + 1, test_last_loss), flush=True)
        
        # Comparing the predicted target with the labels in the batch
        # maledetto give me even zero
        correct = correct + (logits.argmax(1) == batch[2]).sum().item()
        print("Testing accuracy: ",correct/((step + 1) * batch_size))
        
    # compute the validation loss of the epoch    
    print(f"\nTesting epoch {epoch + 1} last loss: ",test_last_loss)         
    avg_loss = correct / len(val_dataloader) 
    return avg_loss
  
# Train and predict
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]                   
valid_losses=[]
#with alive_bar(epochs) as bar:
for epoch in range(epochs):
    print('Training {:} di {:}'.format(epoch + 1, epochs), flush=True)
    train_loss = train()
    valid_loss = evaluate()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'unclepask_propaganda_alpha5.pt')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.9f}')
    print(f'Validation Loss: {valid_loss:.9f}')
#       bar()
