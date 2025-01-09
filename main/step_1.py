'''
Created on 21 dic 2024

@author: pasquale

Web scraping without permission is illegal in many countries around the world.
The author is not responsible for your use!
If you have any doubts DO NOT USE
'''
import pandas as pd
import requests
import bs4
import convert as make
import json
from random import randrange
from time import sleep
from alive_progress import alive_bar

class step_1:
    '''
    Questa classe specifica la prima file del flusso, recuperando il file di input nel formato .tsv
    
    con intestazione
    
    |nome (CHARACTERISTIC)|valore (KEY FIGURE)|testo (CHARACTERISTIC)|
    
    con carattere di tabulazione TAB (\t)
    
    senza intestazione
    
    '''
    def __init__(self, fileSpeech, fileAnagraph, force):
        '''
        fileSpeech: Path dei file di input
        '''
        self.fileSpeech = fileSpeech
        self.fileAnagraph = fileAnagraph
        self.force_scraping = force
        
    def __phase1(self):
        '''
        Fase 1 della trasformazione: inner join della sorgente in input .tsv con la base dati anagrafica in json
        Private Method
        '''
        anagrafica = make.convert.jsonToString(self.fileAnagraph)
        json_load = json.loads(anagrafica)
        dataJSON_pandas = pd.read_json(json_load, orient='records')
        dataFile = pd.read_csv(self.fileSpeech, sep = '\t', names=['surname', 'value', 'topic'])
        outtab1 = pd.merge(dataJSON_pandas, dataFile, how='inner', on=['surname', 'surname'])
        
        return outtab1
    
    def __phase2(self):
        '''
        Fase 2 della trasformazione: ricerca sul web dei metadati associati ai singoli topic.
        Infine effettua la union tra la tabella in ingrsso e quella generata con i metadati
        
        (il risultato cambia in base ai dati transazionali estratti da Google)
        
        Private Method
        '''
        agent={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}       
        url='https://125.0.0.1/search?q='
        tb2 = pd.DataFrame({})
        Data = [ ]
        tb1 = self.__phase1()
        TEXT_POS = tb1.columns.get_loc('topic')
        print('(1) Running - Web Scraping:')
        progress_index = tb1.shape[0]
        with alive_bar(progress_index) as bar:
            for i, row in tb1.iterrows():
#               if i == 3:
#               break
                if i%10 == 0:
                    sec_sleep = randrange(4)
                    sleep(sec_sleep)
            
                    l={}
                try:
                    string_text = row.values[TEXT_POS]
                    query = {'query': string_text}
                    if self.force_scraping == True:
                        num_attempt = 0
                        while self.force_scraping:
                            try:
                                num_attempt = num_attempt + 1
                                html = requests.get(url, headers=agent, params=query)
                                break
                            except:
                                if num_attempt >= 3:
                                    break
                                else:
                                    continue
                    else:
                        html = requests.get(url, headers=agent, params=query)
                    soup = bs4.BeautifulSoup(html.text, 'html.parser')
                    allData = soup.find_all("div",{"class":"g"})
                    try:
                        link = allData[0].find('a').get('href')
                        if(link is not None):
                            try:
                                l["title"]=allData[0].find('h3',{"class":"DKV0Md"}).text
                            except:
                                l["title"]=None
                            try:
                                l["metadata"]=allData[0].find("div",{"class":"VwiC3b"}).text
                            except:
                                l["metadata"]=None
                            try:
                                l["source"]=allData[0].find("span",{"class":"VuuXrf"}).text
                            except:
                                l["source"]=None
                
                            Data.append(l)
                    
                    except:
                        l["title"]=None
                        l["metadata"]=None
                        l["source"]=None
                        Data.append(l)
                        
                except:
                    l["title"]=None
                    l["metadata"]=None
                    l["source"]=None
                    print(f'Warning (!): row {i} without values because connection refused\n')
                    Data.append(l)
                
                tb2 = pd.DataFrame(Data, columns=['title', 'metadata', 'source'])
                bar()
        
        outtab2 = pd.merge(tb1, tb2, how='outer', validate = 'one_to_many', indicator=True, left_index = True, right_index = True)
        
        return outtab2
    
    def data_scout(self):
        '''
        anagraph_scout(self): Richiama i metodi privati __phase1() e __phse2() che effettuano l'import dei dati anagrafici e l'attività di web scrapping su google
        '''
        return self.__phase2()
