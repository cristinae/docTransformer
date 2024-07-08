#! /usr/bin/python3

import sys
import json
import re

#json_file = 'en_meta_part_1.jsonl'

lan = 'es'
test = ['vozlibre.com',
'www.elplural.com',
'www.gironafc.cat/es',
'losviajesdeclaudia.com',
'losviajesdedomi.com',
'www.elrincondesele.com',
'www.telva.com',
'www.mujerhoy.com',
'www.recetasderechupete.com',
'www.donquijote.org/es',
'www.milenio.com',
'www.reforma.com',
'www.tolucafc.com',
'worldtravelfeet.com',
'www.marieldeviaje.com',
'viajerosvagabundos.com',
'www.yovivolamoda.com',
'https://mx.hola.com',
'https://culturacolectiva.com',
'www.cocinadelirante.com',
'www.mexicoenmicocina.com',
'www.cacentralcordoba.com',
'imaginateaca.com',
'elplanetaurbano.com',
'elle.clarin.com',
'www.fondodeolla.com',
'www.lostiempos.com',
'poracayporalla.com',
'unarecetadecocina.com',
'comidaperuanaweb.org',
'perudelicias.com',
'www.elclosetdegiuliana.com',
'www.semana.com',
'www.disfrutandoparaguay.com',
'chile.as.com',
'www.cocina-chilena.com']


fragmentSep = ' <NS> '
def extract(json_file):
    output_file = json_file+'.4testall'
    with open(json_file, 'r') as json_data:
      for line in json_data:
        data = json.loads(line)
        # extract the field with the URL of the document
        domain = data['warc_headers']['warc-target-uri']
        if any(web in domain for web in test):
           document = data['content'].split('\n')
           scores = data['metadata']['sentence_identifications']
           # we only extract the sentences in a document that have been identified as Spanish
           documentLang = ''
           for score, sentence in zip(scores, document):
               if score and score['label']==lan:
                  documentLang = documentLang + sentence.replace('\t',' ') + fragmentSep
           #print(data[sectionField]['warc-record-id'], '\t', documentLang)
           with open(output_file, 'a') as output:
                output.write(data['warc_headers']['warc-record-id']+'\t'+domain+'\t'+documentLang.rstrip(fragmentSep)+'\n')


if __name__ == "__main__":
   try:
      arg = sys.argv[1]
      extract(arg)
   except IndexError:
      raise SystemExit(f"Usage: {sys.argv[0]} <oscar_file>")


# Download Oscar from Hugging Face
# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
# git lfs pull --include compressed/es_meta/es_meta_part_1*.jsonl.gz
