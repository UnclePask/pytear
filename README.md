# PyTeAR


# Warning use python 3.11 or less, please read -> https://github.com/explosion/spaCy/issues/13658

Python Text Analysis Routine, pipeline for identify propaganda in the speechs or articles.
In this Alpha 0 version the pipeline support only English language

<code>
-----------------------------------
| Step 1: Web Scraping            |
-----------------------------------
                |                  
                V                  
-----------------------------------
|Step 2: Text Analysis with nklt  |
-----------------------------------
                |                  
                V                  
-----------------------------------
|Step 3: Run homemade Bert Model  |
|        to check the propaganda  |
-----------------------------------
</code>

Input transactional data file have to TSV format (text file with tab separator), with no header in the first line.
Input master data file have to JSON format, you could use <i>anagraphical.json</i> as a master.

Output transactional data file will be generated in CSV format with semicolon separator (;).

<code>
--------------------------       --------------------------
| INPUT:                 |       | OUTPUT:                |
|     spheech-a.tsv      |   ->  |      spheech-b.Csv     |
|    anagraphic.json     |       |    (Report analysis)   |
--------------------------       --------------------------
</code>


In the future, I would like to implement an OpenAI-based model to integrate text analysis with author profiling and graphic propaganda analysis.

