"""
Adam Klie
03/15/2021
Set of useful functions for analyses cell line databases
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal


GDSC_DIR = '/cellar/users/aklie/Data2/cell_lines/GDSC/processed'
CCLE_DIR = '/cellar/users/aklie/Data2/cell_lines/DepMap/processed'
COSMIC_DIR= '/cellar/users/aklie/Data2/cell_lines/COSMIC/processed/'
    
    
def find_valid_features(mtx, feature_list):
    m = np.array([gene in (mtx.index) for gene in feature_list])
    valid_features = np.array(feature_list)[m]
    missing_features = np.array(feature_list)[~m]
    if len(missing_features) > 0:
        print("Found {0} missing features: {1}".format(len(missing_features), missing_features))
    return valid_features


def correlate_to(mtx_df, col, feature_list=None, col_list=None):
    """
    Function to return correlations to a column of interest

    :param mtx_df (pd.DataFrame): feature by sample pandas DataFrame. Must contain 'col' in columns.
    :param col (str): column name to find correlations to
    :param col_list (list-like, optional): other columns to find correlations to col
    :return: pandas Series with col_list as indeces and correlation to col as values
    """
    if feature_list != None:
        m = [gene in (mtx_df.index) for gene in feature_list]
        valid_features = np.array(feature_list)[m]
        if (~np.array(m)).sum() > 0:
            print("Found invalid features, removing: {}".format(np.array(feature_list)[~np.array(m)]))
    else:
        valid_features = mtx_df.index
    
    corr_df = mtx_df.loc[valid_features].corr()
    
    if col_list != None:
        return corr_df[col][col_list]
    else:
        return corr_df[col]
    
    
def read_txt_file(fname, sep='\n'):
    """
    Function to return a list object from text file

    :param fname (str): filepath to read in
    :param sep (str): delimeter between entries
    :return: python list of objects read in from file
    """
    with open(fname, 'r') as f:
        lst = [line.rstrip() for line in f.readlines()]
    return lst


def get_var_features(mtx, n=2000):
    """
    Function to output out a reduced DataFrame of the most variable n features in a matrix
    
    :param mtx (pandas DataFrame): feature by sample matrix
    :param n (int): number of features to subset
    :return: pandas DataFrame consisting of n most variable features
    """
    print("Make sure your matrix is feature by sample")
    return mtx.loc[mtx.var(axis=1).sort_values(ascending=False).index[:n]]


def make_skree_plot(pca_obj, n_comp=30):
    """
    Function to generate and output a Skree plot
    """
    mp_variance={}
    for i,val in enumerate(pca_obj.explained_variance_ratio_.tolist()):
        key="PC"+str(i+1)
        mp_variance[key]=val
    plt.bar(["PC"+str(i) for i in range(1,n_comp+1)],pca_obj.explained_variance_ratio_.tolist())
    plt.xticks(rotation=90)
    plt.ylabel("Variance Explained")
    plt.xlabel("Principal Component")
    return mp_variance
    
    
def scaled_PCA(mtx, n_comp=30, index_name='index'):
    """
    Function to perform scaling and PCA on an input matrix

    :param mtx sample by feature
    :return: sklearn pca object
    """
    print("Make sure your matrix is sample by feature")
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    pca_obj = PCA(n_components=n_comp)
    pca_obj.fit(mtx_scaled)
    pca_df = pd.DataFrame(pca_obj.fit_transform(mtx_scaled))
    pca_df.columns = ['PC' + str(col+1) for col in pca_df.columns]
    pca_df.index = mtx.index
    pca_df.index.name = index_name
    return pca_obj, pca_df

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    

def stacked_barplot(data, col, color_col, pal, axes, leg=True, xlab='x-axis', title='', legend_col=1):
    sns.histplot(data,
                 x = col,
                 hue=color_col,
                 multiple="stack",
                 palette=pal,
                 edgecolor=".3",
                 linewidth=.5,
                 ax=axes,
                 legend=leg)
    axes.set_xlabel(xlab, fontsize=12)
    axes.set_title(title, fontsize=14)
    axes.xaxis.set_tick_params(labelbottom=True)
    if leg:
        move_legend(axes, 2, bbox_to_anchor=(1.05, 1), ncol=legend_col)
    

def kruskal_wallis_test(data, val_col, group_col):
    """
    Function 
    """
    groups = data[group_col].dropna().unique()
    if len(groups) != 2:
        raise Exception("Comparison needs to be between 2 groups")
    group1_vals = data[data[group_col] == groups[0]][val_col]
    group2_vals = data[data[group_col] == groups[1]][val_col]
    return(kruskal(group1_vals, group2_vals))

def filter_numerical_data(df, stat='mean', lower=-np.inf, upper=np.inf, q=None, axis=0):
    """
    Function to filter a numerical dataframe based on mean or variance
    """
    stat_types = ['mean', 'var']
    if stat not in stat_types:
        raise ValueError("Invalid stat type. Expected one of: %s" % stat_types)
    if q != None:
        print("Using quantile to filter for {0}s {1}-{2}%: ".format(stat, round((q[0]*100), 2), round((q[1]*100), 2)), end="")
        if stat == 'mean':
            upper_q = df.mean(axis=axis).quantile(q[1])
            lower_q = df.mean(axis=axis).quantile(q[0])
            print("Removing {} features".format((~((df.mean(axis=axis) >= lower_q) & (df.mean(axis=axis) <= upper_q))).sum()))
            return df[(df.mean(axis=axis) >= lower_q) & (df.mean(axis=axis) <= upper_q)]
        elif stat == 'var':
            upper_q = df.var(axis=axis).quantile(q[1])
            lower_q = df.var(axis=axis).quantile(q[0])
            print("Removing {} features".format((~((df.var(axis=axis) >= lower_q) & (df.var(axis=axis) <= upper_q))).sum()))
            return df[(df.var(axis=axis) >= lower_q) & (df.var(axis=axis) <= upper_q)]
    else:
        print("Using value filter for features with {0} >= {1} and <= {2}: ".format(stat, lower, upper), end="")
        if stat == 'mean':
            print("Removing {} features".format((~((df.mean(axis=axis) >= lower) & (df.mean(axis=axis) <= upper))).sum()))
            return df[(df.mean(axis=axis) >= lower) & (df.mean(axis=axis) <= upper)]
        elif stat == 'var':
            print("Removing {} features".format((~((df.var(axis=axis) >= lower) & (df.var(axis=axis) <= upper))).sum()))
            return df[(df.var(axis=axis) >= lower) & (df.var(axis=axis) <= upper)]
        

def score_gene_sig(exp_mtx, sig_series, z_scored=False):
    if not z_scored:
        score = np.dot(exp_mtx.loc[sig_series.index].fillna(0).values.T, sig_series.values)
        score = (score - score.mean())/score.std()
    else:
        score = np.dot(exp_mtx.loc[sig_series.index].fillna(0).values.T, sig_series.values)
    return score


def scatter_columns(df, xcol, ycol, axes, drop_cols=[]):
    df = df.drop(drop_cols)
    sns.scatterplot(data=df,
                    x=xcol, 
                    y=ycol,
                    ax=axes)
    corr = np.corrcoef(df[xcol], df[ycol])[0][1]
    axes.set_title('$R^2$=' + str(round(corr, 3)))
    axes.set_xlabel(xcol)
    axes.set_ylabel(ycol)
    
    
#http://www.wellho.net/resources/ex.php4?item=y104/tessapy
def tessa(source):
    result = []
    for p1 in range(len(source)):
            for p2 in range(p1+1,len(source)):
                    result.append([source[p1],source[p2]])
    return result


def database_load(db_name, exp_fpath=None, mdata_fpath=None):
    dbs = ['GDSC', 'CCLE', 'COSMIC']
    if db_name not in dbs:
        raise ValueError("Invalid database type. Expected one of: %s" % dbs)
    if db_name == 'GDSC':
        exp_mtx = pd.read_csv(os.path.join(GDSC_DIR, 'cleaned_RMAproc_basalExp.tsv'), sep='\t', index_col=0)
        mdata = pd.read_csv(os.path.join(GDSC_DIR, 'cleaned_metadata.tsv'), sep='\t', dtype={'COSMIC_ID': str}).set_index('COSMIC_ID')
    elif db_name == 'CCLE':
        exp_mtx = pd.read_csv(os.path.join(CCLE_DIR, 'cleaned_RSEMtpm_exp.tsv'), sep='\t', index_col=0)
        mdata = pd.read_csv(os.path.join(CCLE_DIR, 'cleaned_metadata.tsv'), sep='\t', index_col=0).astype({'COSMICID': object})
    elif db_name == 'COSMIC':
        exp_mtx = pd.read_csv(os.path.join(COSMIC_DIR, 'cleaned_basalExp.tsv'), sep='\t', index_col=0)
        mdata = pd.read_csv(os.path.join(COSMIC_DIR, 'cleaned_metadata_from_CCLE.tsv'), sep='\t', dtype={'COSMICID': str}).set_index('COSMICID')
    return exp_mtx, mdata


def columnize(lst, columns=5):
    out = ""
    for i, item in enumerate(lst):
        if i % columns == 0 and i != 0:
            out = out + item + '\n'
        else:
            out = out + item +', '
    return(out[:-2])