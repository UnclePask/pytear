# PyTeAR


#Warning use python 3.11 or less, please read -> https://github.com/explosion/spaCy/issues/13658

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
|       spheech-a.tsv    |   ->  |       spheech-b.Csv    |
|       anagraphic.json  |       |                        |
--------------------------       --------------------------
</code>
<br>
Header of spheech-a.tsv:
<ol>
  <li>surname (author of topic)</li>
  <li>values (0 or 1 for the future training)</li>
  <li>topic (speech or news)</li>
</ol>
<br>
Header of spheech-b.csv:
<ol>
  <li>name, from master data</li>
  <li>surname</li>
  <li>holiday, from master data</li>
  <li>death, from master data</li>
  <li>nationality, from masterdata</li>
  <li>political_party, from masterdata</li>
  <li>value</li>
  <li>topic</li>
  <li>title, synth STEP 1</li>
  <li>metadata, synth STEP 1</li>
  <li>source, synth STEP 1</li>
  <li>FleschReadIndex, synth STEP 2</li>
  <li>SmogIndex, synth STEP 2</li>
  <li>PolarityScore, synth STEP 2</li> 
  <li>EmpathyScore, synth STEP 2</li>
  <li>UglyScore, synth STEP 2</li>
  <li>StyleText, calc based on the Polarity Score</li>
  <li>Abstract, synth STEP 2</li>
  <li>Keywords, synth STEP 2</li> 
  <li>PropDetectionSynth, synth STEP 3</li>
  <li>PropDetectionNoSynth, synth STEP 3</li>
</ol>

In the future, I would like to implement an OpenAI-based model to integrate text analysis with author profiling and graphic propaganda analysis.

