#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
"""

import sys, os
import argparse

import matplotlib.pyplot as plt
import numpy as np

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
                    default='<NS>', 
                    help="Delimiter used to separate fragments in a document. Default: <NS>")
    parser.add_argument('--histogram', 
                    required=False,
                    type=int, 
                    default=1, 
                    help="Create a histogram 0/1. Default: 1")

    return parser


def createHistogram(counts, INfile):
    n, bins, patches = plt.hist(x=counts, bins=40, color='#607c8e',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.7)
    plt.ylabel('Documents')
    plt.xlabel('Segments (<NS>)')
    #plt.title('My Very Own Histogram')
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.xlim(xmin=0, xmax=200)
    plt.savefig(INfile+'.NS.png')


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
 
    INfile = args.iFile
    OUTfile = INfile+'.count'
    counts = []
    with open(INfile, 'r') as file, open(OUTfile, 'w') as output:
        while True:
          line = file.readline()
          columns = line.split('\t')
          if(len(columns) != 5):
              print('Format error in datafile')
              print(columns)
              break
          doc = columns[4]  #5 for politics
          # to count separators
          seps = doc.count(args.sentence_delimiter)+1
          # to count words
          tokens = doc.count(" ")+1
          words = tokens - seps
          output.write(str(words)+" "+str(seps)+"\n")
          if (args.histogram==1):
              counts.append(seps)
        if (args.histogram==1):
           createHistogram(counts, INfile)   

        
        
if __name__ == "__main__":
   main()
