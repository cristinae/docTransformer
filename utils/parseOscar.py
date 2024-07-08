#! /usr/bin/python3

import sys
import json
import re

fragmentSep = ' <NS> '
def extract(json_file):
    output_file = json_file+'.extracted'
    with open(json_file, 'r') as json_data:
      for line in json_data:
        data = json.loads(line)
       # extract the field with the URL of the document
        domain = data['warc_headers']['warc-target-uri']
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

# Run the script in the sge cluster
# for i in {301..383}
# do
#   bunzip2 es_meta_part_$i.jsonl.bz2
#   python3 parseOscar.py es_meta_part_$i.jsonl
#   bzip2 es_meta_part_$i.jsonl &
# done
