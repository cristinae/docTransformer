#! /usr/bin/python3

import sys
import json
import re

#json_file = 'es_meta_part_1.jsonl'

lan = 'es'
variants = ['ad', 'ar', 'bo', 'cl', 'co', 'cr', 'cu', 'do', 'ec', 'es', 'gq', 'gt', 'hn', 'mx', 'ni', 'pa', 'pe', 'ph', 'pr', 'py', 'sv', 'uy', 've', 'us']

fragmentSep = ' <NS> '
def extract(json_file):
    output_file = json_file+'.unk'
    with open(json_file, 'r') as json_data:
      for line in json_data:
        data = json.loads(line)
        # extract the field with the URL of the document
        domain = data['warc_headers']['warc-target-uri']
        match = re.search(r'\.\w\w\/', domain.lower())
        if match:
           icc = match.group().lstrip('.').rstrip('/')
           # select only those domains from countries where the language is spoken
           if icc in variants:
              output_file = lan+'_meta.'+icc # This overrides the complete file
           else:
              output_file = json_file+'.unk'
        else:
           output_file = json_file+'.unk'
        
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

