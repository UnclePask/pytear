'''
Update on 11 May 2025

@author: pasquale
'''
import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu =  nn.ReLU()                  
        self.fc1 = nn.Linear(768,512)         
        self.fc2 = nn.Linear(512,2)        
 #       self.softmax = nn.LogSoftmax(dim=1)  
    
    def forward(self, sent_id, mask):           
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        nex = self.fc1(cls_hs)
        nex = self.relu(nex)
        nex = self.dropout(nex)
        nex = self.fc2(nex)                       
#        nex = self.softmax(nex)                   
        return nex

    def set_all_trainable(self):
        for param in self.parameters():
            param.requires_grad = True

    def set_all_untrainable(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_classifier_trainable(self):
        self.set_all_untrainable()
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
