'''
Created on 19 may 2025

Class convert di metodi statici per convertire i dataframe verso formati file

@author: pasquale
'''
import json
import pandas as pd
from pathlib import Path
from os import name as nameos

class convert(object):
    '''
    La classe convert contiene solo metodi statici per convertire i formati dataframe verso il formato file
    '''

    @staticmethod
    def toJSON(df):
        '''
        toJSON: converte un dataframe pandas in un file json
        '''    
        resultJSON = df.to_json(orient='records')
        try:
            with open('anagraphic.json', 'w') as file:
                json.dump(resultJSON, file)
            return resultJSON
        except:
            return None
        
    @staticmethod
    def jsonToString(fileJSON):
        '''
        jsonToString: converte un json restituendo una stringa
        '''
        try:
            FILE_JSON = open(fileJSON, 'r')
            str_json = FILE_JSON.read()
            return str_json
        except:
            return None
    
    @staticmethod
    def csvToDataframe(fileCSV, separator):
        '''
        Converte un file CSV in un data frame
        '''
        try:
            data_Frame = pd.read_csv(fileCSV, sep=separator)
            return data_Frame
        except:
            return None
    
    @staticmethod    
    def importJson(fileJSON):
        '''
        importa un JSON e restituisce un dataframe
        '''
        try:
            anagrafica = convert.jsonToString(fileJSON)
            json_load = json.loads(anagrafica)
            dataJSON_pandas = pd.read_json(json_load, orient='records') 
            return dataJSON_pandas
        except:
            return None

    @staticmethod
    def getPath(io):
        if nameos == 'nt':
            if io == 'input':
                path_string = str(Path.cwd()) + '\\inputFile\\speech-a.tsv'
            elif io == 'output':
                path_string = str(Path.cwd()) + '\\outputFile\\speech-b.tsv'
            else:
                raise Exception 
        else:
            if io == 'input':
                path_string = str(Path.cwd()) + '/inputFile/speech-a.tsv'
            elif io == 'output':
                path_string = str(Path.cwd()) + '/outputFile/speech-b.tsv'
            else:
                raise Exception
        return path_string
