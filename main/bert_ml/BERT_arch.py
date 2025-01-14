'''
Created on 4 gen 2025

@author: pasquale
'''
import torch.nn as nn

class BERT_Arch(nn.Module):
    '''
    Architettura del modello Home Made UnclePask
    '''
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)       
        self.relu =  nn.ReLU()               
        self.fc1 = nn.Linear(768,512)        
        self.fc2 = nn.Linear(512,2)          
        self.softmax = nn.LogSoftmax(dim=1)  
    
    def forward(self, sent_id, mask):           
        '''
        il batch contiene la rappresentazione lineare (2 pretrained)
        '''
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        nex = self.fc1(cls_hs)
        nex = self.relu(nex)
        nex = self.dropout(nex)
        nex = self.fc2(nex)                       
        nex = self.softmax(nex)                   
        return nex

    