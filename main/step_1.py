'''
Created on 21 dic 2024

@author: pasquale
'''
from random import randrange
from sys import exc_info
from time import sleep
from alive_progress import alive_bar
import bs4
import requests
import pandas as pd


class step_1:
    
    def __init__(self, fileSpeech, force):
        self.fileSpeech = fileSpeech
        self.force_scraping = force
        
    def __phase0(self):
        t_masterdata = pd.DataFrame({})
        Data = [ ]
        print('(0) Running - Create Masterdata:')
        dataFile = pd.read_csv(self.fileSpeech, sep = '\t', names=['surname', 'value', 'topic'])
        surnames = dataFile['surname'].unique()
        progress_index = surnames.shape[0]
        agent = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
        url = 'https://api.wikimedia.org/core/v1/wikipedia/en/search/page'
        with alive_bar(progress_index) as bar:
            i = 0
            for i in range(len(surnames)):
                self.__wait(i)
                l = {}
                try:
                    string_text = surnames[i]
                    query = {'q': string_text,
                             'limit': 1}
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
                    tb_tmp = pd.read_json(soup.text)
                    dict_from_json = tb_tmp.values[0][0]
                    l["fullName"]=dict_from_json["title"]
                    l["whoIs"]=dict_from_json["excerpt"]
                    l["surname"] = string_text
                    Data.append(l)
                except:
                    l["fullName"]=None
                    l["whoIs"]=None
                    print(f'Warning (!): row {i} without values because connection refused\n')
                    Data.append(l)
                
                t_masterdata = pd.DataFrame(Data, columns=['fullName', 'whoIs', 'surname'])
            
                bar()
            
        return t_masterdata
             
    def __phase1(self, tb0):
        dataFile = pd.read_csv(self.fileSpeech, sep = '\t', names=['surname', 'value', 'topic'])
        outtab1 = pd.merge(tb0, dataFile, how='inner', on=['surname', 'surname'])
        return outtab1
    
    def __phase2(self):
        # temporaneal msg
        print('\n\n(1) Running - Web Scraping:\n[Abort] - Sorry i need to fix it :-) ')
        # end of temp msg
#        agent={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
#        url=''
#        tb2 = pd.DataFrame({})
#        Data = [ ]
#        tb0 = self.__phase0()
#        tb1 = self.__phase1(tb0)
#        TEXT_POS = tb1.columns.get_loc('topic')
#        print('(1) Running - Web Scraping:')
#        progress_index = tb1.shape[0]
#        with alive_bar(progress_index) as bar:
#            for i, row in tb1.iterrows():
#                self.__wait(i)
#                l={}
#                try:
#                    string_text = row.values[TEXT_POS]
#                    query = {'query': string_text}
#                    if self.force_scraping == True:
#                        num_attempt = 0
#                        while self.force_scraping:
#                            try:
#                                num_attempt = num_attempt + 1
#                                html = requests.get(url, headers=agent, params=query)
#                                break
#                            except:
#                                if num_attempt >= 3:
#                                    break
#                                else:
#                                    continue
#                    else:
#                        html = requests.get(url, headers=agent, params=query)
#                    soup = bs4.BeautifulSoup(html.text, 'html.parser')
#                    allData = soup.find_all("div",{"class":"g"})
#                    try:
#                        link = allData[0].find('a').get('href')
#                        if(link is not None):
#                            try:
#                                l["title"]=allData[0].find('h3',{"class":""}).text
#                            except:
#                               l["title"]=None
#                           try:
#                                l["metadata"]=allData[0].find("div",{"class":""}).text
#                            except:
#                                l["metadata"]=None
#                                l["source"]=None
#                                l["source"]=allData[0].find("span",{"class":""}).text
#                            except:
#                            try:
#                
#                            Data.append(l)
#                    
#                    except:
#                        l["title"]=None
#                        l["metadata"]=None
#                        l["source"]=None
#                        Data.append(l)
#                        
#                except:
#                    l["title"]=None
#                    l["metadata"]=None
#                    l["source"]=None
#                    print(f'Warning (!): row {i} without values because connection refused\n')
#                    Data.append(l)
#                
#                tb2 = pd.DataFrame(Data, columns=['title', 'metadata', 'source'])
#                bar()
#        
#        outtab2 = pd.merge(tb1, tb2, how='outer', validate = 'one_to_many', indicator=True, left_index = True, right_index = True)
#        
#        return outtab2
    
    def __getCaller(self):
        try:
            raise Exception
        except Exception:
            frame = exc_info()[3].tb_frame.f_back
        #string
        return frame.f_code.co_name
    
    def __wait(self, i):
        if i%10 == 0 and i != 0:
            sec_sleep = randrange(4)
            sleep(sec_sleep)
    
    def data_scout(self):
        tab0 = self.__phase0()
        #temporaneal fix
        self.__phase2()
        #end temporaneal fix
        return self.__phase1(tab0)
