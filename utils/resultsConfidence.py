#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Mean number of instances per class in a binary classification exercise
    Instances are stored in file --iFile one per line in a column format with 
    the class label [0/1] in column --column
    Confidence intervals obtained via bootstrap resampling
"""

import sys, os
import argparse
import re

import random
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
                    help="File with the results of a binary classifier" )
    parser.add_argument('-c', '--column',
                    required=False,
                    type=int,
                    default=7, 
                    metavar="<column#>",
                    help="Number of colum where the result of the classification is" )
    parser.add_argument('-b', '--bootstraps',
                    required=False,
                    type=int,
                    default=1000, 
                    metavar="<#bootstraps>",
                    help="Number of bootstrap samples for CI (default 1000)" )
    parser.add_argument('-l', '--confidenceLevel',
                    required=False,
                    type=int,
                    default=95, 
                    metavar="<confidence level (%)>",
                    help="Confidence level in percentage (default 95)" )
    return parser



def ci_bs_class0(distribution, n, confLevel):
    ''' Calculates confidence intervals for distribution at confLevel after the 
        generation of n boostrapped samples 
    '''

    bsScores = np.zeros(n)
    size = len(distribution)
    random.seed(16) 
    for i in range(0, n):
        # generate random numbers with repetitions, to extract the indexes of the sysScores array
        bootstrapedSys = np.array([distribution[random.randint(0,size-1)] for x in range(size)])
        # scores for all the bootstraped versions
        # we look for the percentage of the 0 class
        bsScores[i] =  np.count_nonzero(bootstrapedSys=='0')/(len(bootstrapedSys)-1) * 100

    # we assume distribution of the sample mean is normally distributed
    # number of bootstraps > 100
    mean = np.mean(bsScores,0)
    stdDev = np.std(bsScores,0,ddof=1)
    # Because it is a bootstraped distribution
    alpha = (100-confLevel)/2
    confidenceInterval = np.percentile(bsScores,[alpha,100-alpha])

    return (mean, mean-confidenceInterval[0])
    

def ci_bs_class1(distribution, n, confLevel):
    ''' Calculates confidence intervals for distribution at confLevel after the 
        generation of n boostrapped samples 
    '''

    bsScores = np.zeros(n)
    size = len(distribution)
    random.seed(16) 
    for i in range(0, n):
        # generate random numbers with repetitions, to extract the indexes of the sysScores array
        bootstrapedSys = np.array([distribution[random.randint(0,size-1)] for x in range(size)])
        # scores for all the bootstraped versions
        #### this works because we assume the MT metric is calculated at sentence level
        ### bsScores[i] = np.mean(bootstrapedSys,0)
        # we look for the percentage of the 0 class
        bsScores[i] = np.count_nonzero(bootstrapedSys=='1')/(len(bootstrapedSys)-1) * 100

    # we assume distribution of the sample mean is normally distributed
    # number of bootstraps > 100
    mean = np.mean(bsScores,0)
    stdDev = np.std(bsScores,0,ddof=1)
    # Because it is a bootstraped distribution
    alpha = (100-confLevel)/2
    confidenceInterval = np.percentile(bsScores,[alpha,100-alpha])

    return (mean, mean-confidenceInterval[0])
    


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
 
    INfile = args.iFile
    column = args.column
    
    confLevel = args.confidenceLevel  
    precision = 1
    
    articles = []
    with open(INfile, 'r') as file:
          articles=file.read().split('\n')
    
    classes = [0]*len(articles)
    i = 0
    for art in articles:
        tmp = art.split('\t')
        if(len(tmp)<column): continue
        classes[i] = tmp[column]
        i=i+1
    
    mean0, interval0 = ci_bs_class0(classes, args.bootstraps, confLevel)  
    #value0 = str(np.around(mean0, precision)) + ' $\pm$ ' + str(np.around(interval0, precision))
    value0 = str(round(mean0)) + '$\pm$' + str(round(interval0))
    mean1, interval1 = ci_bs_class1(classes, args.bootstraps, confLevel)   
    #value1 = str(np.around(mean1, precision)) + ' $\pm$ ' + str(np.around(interval1, precision))
    value1 = str(round(mean1)) + '$\pm$' + str(round(interval1))
    print('&', value0, '&', value1)
       
        
        
if __name__ == "__main__":
   main()
