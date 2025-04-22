'''
Created on 22 dic 2024

@author: pasquale
'''
import pandas as pd
import en_core_web_trf
import summarize as make
from textstat.textstat import textstatistics
from nltk.sentiment import SentimentIntensityAnalyzer
from numpy import round
from alive_progress import alive_bar

class step_2(object):
    '''
    classdocs Questa classe specifica la seconda fase 
    del flusso della pipeline, in cui vengono generate le informazioni 
    su readability, leve emotive, caratteristiche linguistiche e conteggio di parole specifiche
    
    
    in questa fase vengono misurati: 
        > readability: secondo la scala di Flesch da 0 a 100 (Flesch read easy)
        > SMOG index
    
    Viene effettuata la Sentiment analysis calcolando
        > Polarity Score
        > Dettaglio del polarity score
        > Emphaty Score
        > Eventuale nota sulla leva emotiva basata sul Polarity Score
    
    vengono:
        > generti gli abstract del discorso
        > calcolate ed estratte le 6 keywords principali secondo il punteggio assegnato dal modello (occorrenza e rilevanza nel discorso)
    
    '''
    
    def __init__(self, speech_df):
        self.model = en_core_web_trf.load()
        self.sid = SentimentIntensityAnalyzer()
        try:
            self.df = speech_df
            if 'FleschReadIndex' not in self.df:
                self.df = self.df.reindex(columns=self.df.columns.tolist() + ['FleschReadIndex'])
            if 'SmogIndex' not in self.df:
                self.df = self.df.reindex(columns=self.df.columns.tolist() + ['SmogIndex'])
            if 'PolarityScore' not in self.df:
                self.df = self.df.reindex(columns=self.df.columns.tolist() + ['PolarityScore'])
            if 'EmpathyScore' not in self.df:
                self.df = self.df.reindex(columns=self.df.columns.tolist() + ['EmpathyScore'])
            if 'UglyScore' not in self.df:
                self.df = self.df.reindex(columns=self.df.columns.tolist() + ['UglyScore'])
            if 'StyleText' not in self.df:
                self.df = self.df.assign(StyleText=lambda x: ' ')
            if 'Abstract' not in self.df:
                self.df = self.df.assign(Abstract=lambda y: ' ')
            if 'Keywords' not in self.df:
                self.df = self.df.assign(Keywords=lambda z: ' ')
        except:
            print(f'\nError (5): reindex of data frame function failed in node 2 \n\n')
   
    def __neighbor_to_one(self, my_sent_score):
        ref_score = abs(my_sent_score['pos']) - abs(my_sent_score['neg'])
        if ref_score == 0:
            if my_sent_score['neu'] > 0.75:
                return 'Il discorso è bilanciato e non prevalgono discriminanti notevoli per il modello'
            else:
                return 'Il discorso mira a creare un senso di incertezza verso il pubblico'
        elif ref_score > 0:
            if my_sent_score['neu'] > my_sent_score['pos']:   
                return 'Il discorso promuove impressioni positive'
            else:
                return 'Il discorso esalta gli aspetti positivi in oggetto'
        elif ref_score < 0:
            if my_sent_score['neu'] > my_sent_score['neg']:   
                return 'Il discorso promuove sensazioni negative'
            else:   
                return 'Il discorso fa una forte leva su senzazioni negative'
        else:
            return 'Fuori scala'
            
    def text_analysis(self):
        try:
            line_check_exception = 0
            TEXT_POS = self.df.columns.get_loc('topic')
            FRES_POS = self.df.columns.get_loc('FleschReadIndex')
            SMOG_POS = self.df.columns.get_loc('SmogIndex')
            POLS_POS = self.df.columns.get_loc('PolarityScore')
            EMPS_POS = self.df.columns.get_loc('EmpathyScore')
            UGLY_POS = self.df.columns.get_loc('UglyScore')
            STYL_POS = self.df.columns.get_loc('StyleText')
            ABST_POS = self.df.columns.get_loc('Abstract')
            KEYW_POS = self.df.columns.get_loc('Keywords')
            data = []
            print('\n(2) Running - Text analysis:')
            progress_index = self.df.shape[0]
            with alive_bar(progress_index) as bar:
                for i, row in self.df.iterrows():
                    line_check_exception = i
                    text = row.values[TEXT_POS]
                    doc = self.model(text)
                    
                    num_words = textstatistics().lexicon_count(text, True)
                    num_token = len(doc.ents)
                    language = doc.vocab.lang
                    try:
                        ASL = 1.025 * (num_words / num_token)
                    except:
                        row.values[FRES_POS] = -1
                        continue
                    num_syll = textstatistics().syllable_count(text)
                    try:
                        ASW = 84.6 * (num_syll / num_words)
                    except:
                        row.values[FRES_POS] = -1
                        continue
                    if ASL != -1 and ASW != -1:
                        flesch_read_easy = 206.835 - ASL - ASW
                        row.values[FRES_POS] = round(flesch_read_easy, 2)
                        smog_index = textstatistics().smog_index(text)
                        row.values[SMOG_POS] = round(smog_index, 2)
                        sentiment_score = self.sid.polarity_scores(text).values().mapping
                        row.values[POLS_POS] = round(sentiment_score['compound'], 2)
                        row.values[EMPS_POS] = round(sentiment_score['pos'], 2)
                        row.values[UGLY_POS] = round(sentiment_score['neg'], 2)
                        row.values[STYL_POS] = self.__neighbor_to_one(sentiment_score)
                        row.values[ABST_POS] = make.summarize.createAbstract(text)
                        row.values[KEYW_POS] = make.summarize.extractKeywords(text, language)
                    else:
                        row.values[STYL_POS] = 'Non classificabile'
                        row.values[ABST_POS] = 'Non classificabile'
                        
                    data.append(row)
                    tb3 = pd.DataFrame(data, columns=['fullName', 'whoIs', 'surname', 'topic', 'FleschReadIndex', 'SmogIndex', 'PolarityScore', 'EmpathyScore', 'UglyScore', 'StyleText', 'Abstract', 'Keywords'])
                    bar()

            return tb3
                    
        except:
            if line_check_exception == 0:
                print('\nError (3): index of column not found in the input dataframe in the node 2 \n\n')
            else:
                print(f'\nError (4): row {line_check_exception} missmatch type \n\n')
        
        
