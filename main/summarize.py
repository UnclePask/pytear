'''
Created on 29 dic 2024

@author: pasquale
'''

from yake import KeywordExtractor
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer

class summarize(object):
    '''
    classdocs Classe statica con i metodi che convertono il risultato della text analysis da LIST a STRING
    '''
    @staticmethod   
    def createAbstract(text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=5)
        return ' '.join([str(sentence) for sentence in summary])
    
    @staticmethod
    def extractKeywords(text, language):
        keywords_list = KeywordExtractor(lan=language, n=1, dedupLim=0.8, top=7).extract_keywords(text)
        top_word_list = list(map(lambda x: x[0], keywords_list))
        top_words_str = ' '.join(map(str,top_word_list))
        top_words_str = top_words_str.replace(' ', ',')
        top_words_str = top_words_str.upper()
        return top_words_str
    
    @staticmethod
    def extractPrincipalToken(text, language):
        keywords_list = KeywordExtractor(lan=language, n=1, dedupLim=0.9, top=20).extract_keywords(text)
        return keywords_list