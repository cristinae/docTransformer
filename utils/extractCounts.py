#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
"""

import sys, os
import argparse

import matplotlib.pyplot as plt

import numpy as np


plt.rcParams.update({
 "axes.linewidth":2,
 "xtick.major.width":2,
 "xtick.minor.width":2,
 "ytick.major.width":2,
 "ytick.minor.width":2,
 "xtick.major.size":8,
 "ytick.major.size":8,
 "xtick.minor.size":6,
 "ytick.minor.size":6
})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 20
})
fontTit = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'heavy',
        'size': 22,
        }
fontAx = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 18,
        }


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
    plt.figure(figsize=(8,5.5))
    n, bins, patches = plt.hist(x=counts, bins=40, color='#801515',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.7)
    plt.ylabel('\# Documents')
    plt.xlabel('Segments ($<$NS$>$)')
    #plt.xlabel('Words')
    #plt.title('kk')
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.ylim(ymin=0, ymax=452000)
    plt.xlim(xmin=0, xmax=153)
    #plt.xlim(xmin=0, xmax=3100)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,1000))
    plt.tight_layout(pad=0.3)
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
