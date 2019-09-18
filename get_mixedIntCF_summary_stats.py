import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pandas as pd
import numpy as np
import pickle
import timeit

def decode_mixedIntCF(summary):

    summary_decoded = []
    colnames = list(inputs_unique.columns)

    for ix, chsumm in enumerate(summary):
        x1 = list(inputs_unique.iloc[ix].values)
        for feature,col in zip(x1, colnames):
            if col in dict_num_feat:
                x1[colnames.index(col)] = dict_num_feat[col][feature]

        summ = []

        for jx, cf in enumerate(chsumm[:-2]):

            if cf.startswith('You'):
                startline = 2
            else:
                startline = 1

            newcf = [e for e in x1]
            add_cf = True
            for line in cf.splitlines()[startline:]:
                feature = line.split()[0]
                value = int(line.split()[4])

                # categorical features
                if feature in dict_num_feat:
                    try:
                        value = dict_num_feat[feature][value]
                    except:
                        add_cf = False
                        break
                else:
                # continuous features
                    if((value < inputs_unique[feature].min()) | (value > inputs_unique[feature].max())):
                        add_cf = False
                        break

                newcf[colnames.index(feature)] = value

            if add_cf:
                summ.append(newcf)

        summ += chsumm[-2:]
        summ.append(len(summ[:-2]))

        summary_decoded.append(summ)

    return summary_decoded

def get_mixedIntCF_valid_ix(data):
    valid_ix = {'1':[], '2':[], '4':[], '6':[], '8':[], '10':[]}
    for total_CFs in [1,2,4,6,8,10]:
        with open('mixedIntCF_results/'+ data + '_cf_'+str(total_CFs)+'.data', 'rb') as filehandle:
            chris_summary = pickle.load(filehandle)

        chris_summary = chris_summary[0:500]
        results = chris_summary_decode(chris_summary)

        for ix in range(len(results)):
            if results[ix][-1] == total_CFs:
                valid_ix[str(total_CFs)].append(ix)

def get_mixedIntCF_summary():

    mixedIntCF_valid_ix = get_mixedIntCF_valid_ix()


if __name__ == "__main__":
    get_mixedIntCF_summary()
