# -*- coding: utf-8 -*-

"""
Python script for evaluating PRSs with AUROC
TODO: 
"""

# Built-in/Generic Imports
import argparse

# Libs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Options
matplotlib.rcParams['pdf.fonttype'] = 42

# Tags
__author__ = "Adam Klie"
__data__ = "09/16/2021"


def plot_AUROC_curve(fprs, tprs, aucs, labels=None, file=None):
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    for model in list(fprs.keys()):
        ax.plot(fprs[model], tprs[model], ls='-', lw=3, alpha=0.4, label='%s (auROC= %0.4f)' % (model, aucs[model]))    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=3, color='k', label='No skill', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.set_title('Receiver operating characteristic', fontsize=18)
    ax.legend(loc="lower right", fontsize=14);
    if file != None:
        print("Plotted curve to {}.pdf".format(file))
        plt.savefig(file + ".pdf")

        
def main(args):
    
    print("\nAUROC Performance\n" + "-"*len("AUROC Performance"))
    
    #Step 1: Read in all the values
    score_file_list = args.score_files
    
    # Load in scores
    score_file_content = pd.read_csv(score_file_list, sep="\t", header=None)

    # Add names
    if args.names != None:
        names = np.loadtxt(args.names, dtype=str)
    else:
        if score_file_content.shape[1] == 1:
            names = ["score_" + str(lab+1) for lab in range(len(score_file_content))]
        else:
            names = ["score"]
    
    if score_file_content.shape[1] == 1:
        print("\nLoading scores from the following {} files:".format(len(score_file_content)))
        [print("\t" + names[i] + "\t" + fname) for i, fname in enumerate(score_file_content[0].values)]
        df = pd.DataFrame()
        for i, file in score_file_content.iterrows():
            curr_score = pd.read_csv(file[0], sep="\t", header=None, index_col=0)
            if (i > 0) & ((~curr_score.index.isin(df.index)).sum() != 0):
                print("\t\t{} IDs in {} do not match previous score files, they will be excluded".format((~curr_score.index.isin(df.index)).sum(), file[0]))
            df = pd.concat([df, curr_score], axis=1)
            df = df.dropna()
    else:
        print("Loading scores from {}:".format(score_file_list))
        df = pd.read_csv(score_file_list, sep="\t", header=None, index_col=0)

    df.columns = names
    df.index = df.index.astype(str)
    df.index.name = "ID"
    
    if args.ids != None:
        ids = np.loadtxt(args.ids, dtype=str)
        df = df.loc[ids]
        print("\tSubsetted to {} samples".format(len(ids)))
    
    print("\nLoading true labels from: {}".format(args.labels))
    labels = pd.read_csv(args.labels, sep="\t", header=None, index_col=0) 
    labels.index = labels.index.astype(str)
    print("\t{} IDs with scores but no labels, these are excluded from the AUROC calculation".format((~df.index.isin(labels.index)).sum()))
    df = df[df.index.isin(labels.index)]

    labels = labels.loc[df.index]
    print("\t{} duplicated IDs with labels, taking the first".format(labels.index.duplicated().sum()))
    labels = labels[~labels.index.duplicated()]

    if not (labels.index == df.index).all():
        print("\nIndexes for labels and scores do not match, exiting")
        quit()

    if len(labels[1].unique()) == 2:
        #labels[1] = labels[1].replace({1:0, 2:1})
        #print(labels[1])
        labels[1] = label_binarize(labels, classes=sorted(labels[1].unique()))
    else:
        print("\nLabels are not binary, exiting")
        quit()
                                     
    #Step 2: Calculate and store TPRs/FPRs/thresholds
    print("\nCalculating TPRs, FPRs and AUROCs")
    fprs, tprs, threshs, aucs = {}, {}, {}, {}
    for col in df.columns:
        curr_scores = df[col].values
        true_labels = labels[1].values
        fprs[col], tprs[col], threshs[col] = roc_curve(y_true=true_labels, y_score=curr_scores)
        aucs[col] = auc(fprs[col], tprs[col])
    
    #Step 3: Plot curves
    if args.plot:
        plot_AUROC_curve(fprs, tprs, aucs, file=args.out)
        
    #Step 4: Save all files
    print("Saving AUROCs to tsv file: {}".format(args.out + ".aucs.txt"))
    pd.Series(aucs).to_csv(args.out + ".aucs.txt", sep="\t")
    
    print("Saving TPR, FPR, Thresholds to MultiIndexed tsv file: {}".format(args.out + ".summary.txt"))
    tpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tprs.items() ])) 
    fpr_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in fprs.items() ]))
    thresh_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in threshs.items() ]))
    tpr_df.columns = pd.MultiIndex.from_tuples(list(zip(*[["tpr"]*len(tpr_df.columns), list(tpr_df.columns)])))
    fpr_df.columns = pd.MultiIndex.from_tuples(list(zip(*[["fpr"]*len(fpr_df.columns), list(fpr_df.columns)])))
    thresh_df.columns = pd.MultiIndex.from_tuples(list(zip(*[["thresh"]*len(thresh_df.columns), list(thresh_df.columns)])))
    summary_df = pd.concat([tpr_df, fpr_df, thresh_df], axis=1)
    summary_df.swaplevel(0, 1, axis=1).to_csv(args.out + ".summary.txt", sep="\t", index=False)

    print("-"*len("AUROC Performance") + "\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_files", type=str, help="Path to a two column file with IDs and scores or list of such files")
    parser.add_argument("--labels", type=str, help="Path to two column file with IDs and label")
    parser.add_argument("--plot", type=bool, default=True, help="True/False on whether to generate pdf plots or not")
    parser.add_argument("--ids", type=str, default=None, help="(optional) list of IDs (one per line) to subset the individuals on")
    parser.add_argument("--names", type=str, default=None, help="(optional) list of labels (one per line) for the AUROC in '--score_files'")
    parser.add_argument("--out", type=str, default="./test", help="Output prefix to write output files to. Currently outputs 3 files (AUCROC value(s), FPR/TPR/thresholds, AUROC figure)")
    args = parser.parse_args()
    main(args)
