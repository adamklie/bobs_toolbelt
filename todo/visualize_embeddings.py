#!/usr/bin/env python

import pandas as pd
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
import random
from multiprocessing import Pool
import os
import argparse
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Function to load in embedding and return np arrays for processing
def load_embedding(filename):
    embedding = pd.read_csv(filename, sep=' |\t', engine='python', header=None, index_col=0)
    embedding_shuf = embedding.sample(frac=1)
    embedding_genes = embedding_shuf.index
    return embedding_shuf, np.asarray(embedding_genes)

def perform_PCA(data, num_PCs=50):
    pca = PCA(n_components=num_PCs)
    pca.fit(data)
    pca_transform = pca.transform(data)
    print('PC1:', str(round(pca.explained_variance_ratio_[0]*100, 2)))
    print('PC2:', str(round(pca.explained_variance_ratio_[1]*100, 2)))
    return pca_transform


def perform_TSNE(iter, data, out_dir, n_cores=32):
    print(iter + " begin")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=int(iter), learning_rate=200, n_jobs=n_cores)
    np.set_printoptions(suppress=False)
    
    # save tsne data
    Y = tsne.fit_transform(data)
    np.savetxt(os.path.join(out_dir, "TSNE_data_gene2vec.txt_" + iter + ".txt"), Y)
    print(iter + " tsne done!")
    

def perform_umap(data):
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(data)
    umap_ = reducer.fit_transform(scaled_data)
    return umap_

# Function to generate tsne of different iteration in parallel
def mp_handler(data, out, n_threads):
    p = Pool(6)
    p.starmap(perform_TSNE, [("100", data, out, n_threads),
                             ("5000", data, out, n_threads),
                             ("10000", data, out, n_threads),
                             ("20000", data, out, n_threads),
                             ("50000", data, out, n_threads),
                             ("100000", data, out, n_threads)])
    

def shiftedColorMap(cmap, start=0.0, midpoint=0.75, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# Main function
def run_multicore_tsne(args):
    
    # Read in file
    embeddings_file = args.embedding_file
    wv, vocabulary = load_embedding(embeddings_file)
    
    indexes = list(range(len(wv)))
    random.shuffle(indexes)

    topN = len(wv)

    rdWV = wv[indexes][:topN,:]
    rdVB = vocabulary[indexes][:topN]
    print("PCA!")
    pca = PCA(n_components=50)
    pca.fit(rdWV)
    pca_rdWV=pca.transform(rdWV)
    print("PCA done!")
    print("tsne!")

    with open(os.path.join(outdir, "TSNE_label_gene2vec.txt"), 'w') as out:
        for str in rdVB:
            out.write(str + "\n")
    out.close()
    
    mp_handler(args.num_threads)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please specify embedding file trained by gene2vec.py')
    parser.add_argument('-f', '--embedding_file', type=str, help='Embedding file trained with gene2vec.py')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Directory to output TSNE loadings')
    parser.add_argument('-p', '--num_threads', type=int, default=32, help='Number of threads to parallelize over')
    args = parser.parse_args()
    main(args)
