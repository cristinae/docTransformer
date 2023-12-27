#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Extract raw text from PoliOscar
    Date: 03.12.2023
    Author: cristinae
'''

import sys, os
import argparse

def get_parser():
    '''
    Creates a new argument parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iFile',
                    required=True,
                    type=str,
                    metavar="<inputFile>",
                    help="File with the corpus" )
    parser.add_argument('-d', '--sentence_delimiter', 
                    required=False,
                    type=str, 
                    default=' <NS> ', 
                    help="Delimiter used to separate fragments in a document. Default: <NS>")
    return parser


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)

    INfile = args.iFile
    OUTfile = INfile+'.snt'
    lineNum = 1
    num_cols = 3
    with open(INfile, 'r') as file, open(OUTfile, 'w') as output:
        while True:
          line = file.readline()
          columns = line.split('\t')
          if(len(columns) != num_cols):
              print('Format error in datafile, line',lineNum)
              print(columns)
              break
          doc = columns[num_cols-1]  #5 for politics
          sentences = doc.split(args.sentence_delimiter)
          lineNum += 1
          for sentence in sentences:
              sentence = sentence.strip("\n")
              if (not sentence.isspace()):
                 output.write(sentence+"\n")


        
        
if __name__ == "__main__":
   main()
