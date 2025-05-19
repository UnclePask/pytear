'''
Created on 19 may 2025

@author: pasquale
'''
import convert as make
import step_1 as at1_do
#import step_2 as at2_do
#import step_3 as at3_do
from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')

def start_pipeline(FilePath):
    print('\n\nCiao, questa è PyTeAR (alpha 0)!\nthe routine transformation pipeline to identify the propaganda content \nin the speechs and articles\n\n')
    node1 = at1_do.step_1(FilePath, True)
    table1 = node1.data_scout()
    table1.to_csv('../pytear/outputfile/speech-1_2.csv', sep=';')
    print('\n')
    node2 = at2_do.step_2(table1)
    table2 = node2.text_analysis()
    table2.to_csv('../pytear/outputfile/speech-3_4.csv', sep=';')
    print('\n')
    node3 = at3_do.step_3(table2)
    table3 = node3.propaganda_detection()
    table3.to_csv('../pytear/outputFile/speech-b.csv', sep=';')
    new_file_path = Path('../pytear/outputFile/speech-b.csv')
    if new_file_path.exists():
        print("\nThe output file ../speech-b.csv generated\n\n")
        return 0
    else:
        print("\nError (1): the output file wasn't create\n\n")
        return -1
    #print(table3)
    
def main():
#    anagraphPath = '../inputFile/anagraphic.json'
    inputFile = make.convert.getPath('input')
#    exit_val = start_pipeline(inputFile, anagraphPath)
    exit_val = start_pipeline(inputFile)
    return exit_val
    
if __name__ == '__main__':
    main()
