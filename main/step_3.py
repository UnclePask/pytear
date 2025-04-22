'''
Created on 30 dic 2024

@author: pasquale
'''
from openai import OpenAI
from thinc.compat import torch
from transformers import BertTokenizer, BertModel
from bert_ml.BERT_arch import BERT_Arch
from alive_progress import alive_bar
import pandas as pd
import numpy as np
import convert as make

class step_3(object):
    
    def __init__(self, speech_df):
        try:
            self.df = speech_df
            if 'PropDetectionSynth' not in self.df:
                self.df = self.df.assign(PropDetectionSynth=lambda x: ' ')
            if 'PropDetectionNoSynth' not in self.df:
                self.df = self.df.assign(PropDetectionNoSynth=lambda y: ' ')
        except:
            print('\nError (5): reindex of data frame function failed in the node 3 \n\n')
        
        try:
            self.path = '../unclepask_propaganda_alpha5.pt'
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.uncle_pask = torch.load(self.path)
            self.model = BERT_Arch(self.bert)
            self.model.load_state_dict(self.uncle_pask, strict=False)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.MAX_LENGHT = 500
        except:
            print('\nError (6): instance of BERT model failed \n\n')
        
    def propaganda_detection(self):
        try:
            TEXT_POS = self.df.columns.get_loc('Abstract')
            TXTN_POS = self.df.columns.get_loc('topic')
            SDET_POS = self.df.columns.get_loc('PropDetectionSynth')
            NDET_POS = self.df.columns.get_loc('PropDetectionNoSynth')
        except:
            print('\nError (3): index of column not found in the input dataframe in the node 3 \n\n')
        
        progress_index = self.df.shape[0]
        data = []
        print('\n(3) Running - Propaganda detection (synth & no synth data):')
        with alive_bar(progress_index) as bar:
            for i, row in self.df.iterrows():
                try:
                    text = row[TEXT_POS]
                    text_nosynth = row[TXTN_POS]
                except:
                    text = 'Riga ' + i + ' non e\' una stringa\n'
                    text_nosynth = 'Riga ' + i + ' non e\' una stringa\n'
                text_speech = [ text ]
                text_speech_ns = [ text_nosynth ]
                
                #synthetic data block
                tokens_unseen = self.tokenizer.batch_encode_plus(text_speech,
                                                                 max_length = self.MAX_LENGHT,
                                                                 pad_to_max_length=True,
                                                                 truncation=True
                                                                 )
                
                unseen_seq = torch.tensor(tokens_unseen['input_ids'])
                unseen_mask = torch.tensor(tokens_unseen['attention_mask'])
                #end synth data blocco
                
                #no synthetic data block
                tokens_unseen_ns = self.tokenizer.batch_encode_plus(text_speech_ns,
                                                                    max_length = self.MAX_LENGHT,
                                                                    pad_to_max_length=True,
                                                                    truncation=True
                                                                    )
                
                unseen_seq_ns = torch.tensor(tokens_unseen_ns['input_ids'])
                unseen_mask_ns = torch.tensor(tokens_unseen_ns['attention_mask'])
                #fine dati dal file bloco
        
                with torch.no_grad():
                    preds = self.model(unseen_seq, unseen_mask)
                    preds = preds.detach().cpu().numpy()
                    preds_ns = self.model(unseen_seq_ns, unseen_mask_ns)
                    preds_ns = preds_ns.detach().cpu().numpy()
        
                preds = np.argmax(preds, axis = 1)
                if preds[0] == 1:
                    row[SDET_POS] = 'Propaganda detection non superata'
                else: 
                    row[SDET_POS] = 'Propaganda detection superata'
                    
                preds_ns = np.argmax(preds_ns, axis = 1)
                if preds_ns[0] == 1:
                    row[NDET_POS] = 'Propaganda detection non superata'
                else: 
                    row[NDET_POS] = 'Propaganda detection superata'
                    
                data.append(row)
                tb4 = pd.DataFrame(data, columns=['fullName', 'whoIs', 'surname', 'topic', 'title', 'metadata', 'source', 'FleschReadIndex', 'SmogIndex', 'PolarityScore', 'EmpathyScore', 'UglyScore', 'StyleText', 'Abstract', 'Keywords', 'PropDetectionSynth', 'PropDetectionNoSynth'])
                bar()
        
        return tb4
    
    def report(self, pathAnagr):
        '''
        Integrazione con AI generativa attualmente non implementata
        '''
        tb5 = make.convert.importJson(pathAnagr)

        try:
            SURN_POS = tb5.columns.get_loc('surname')
            if 'AuthorAnalysis' not in self.df:
                self.df = self.df.assign(AuthorAnalysis=lambda x: ' ')
        except:
            return FileExistsError
        client = OpenAI()
#        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        data = []
        progress_index = tb5.shape[0]
        print('Author Analysis:')
        with alive_bar(progress_index) as bar:
            for i, row in tb5.iterrows():
                surname = row[SURN_POS]
                
                try:
                    prompt = 'author analysis of {0}'.format(surname)                                           
                    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                                messages=[{"role": "user", 
                                                                           "content": prompt }]
                                                                           )
                except:
                    print(f'\nriga {i} senza informazioni\n\n')
                    continue
                
                row['AuthorAnalysis'] = completion.choices[0].message.content
                data.append(row)
                tb5 = pd.DataFrame(data, columns=['name', 'surname', 'holiday', 'death', 'nationality', 'AuthorAnalysis'])
                bar()
        
        return tb5
