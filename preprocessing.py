# -*- coding: utf-8 -*-

"""
Python script with functions for preprocessing data
"""

def make_skree_plot(pca_obj, n_comp=30):
    """
    Function to generate and output a Skree plot
    """
    variance={}
    for i,val in enumerate(pca_obj.explained_variance_ratio_.tolist()):
        key="PC"+str(i+1)
        variance[key]=val
    plt.bar(["PC"+str(i) for i in range(1,n_comp+1)],pca_obj.explained_variance_ratio_.tolist())
    plt.xticks(rotation=90)
    plt.ylabel("Variance Explained")
    plt.xlabel("Principal Component")
    return variance


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