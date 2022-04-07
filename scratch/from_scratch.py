# -*- coding: utf-8 -*-

"""
Python script with functions for methods implemented from scratch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

###
# PCA
###

# Primary calculation of eigenvalues and eigenvectors of the data covariance matrix
def get_pcs(scaled_data):
	cov = np.cov(scaled_data.T)
	values, vectors = np.linalg.eig(cov)
	order = np.argsort(values)[::-1]
	return values[order], vectors[:, order]


# Transform the data using a basis of vectors
def transform_data(scaled_data, vectors, k=None):
	if k != None:
		return (scaled_data.dot(vectors[:, :k]))
	return (scaled_data.dot(vectors))


# Calcualte and return variance explained ratios
def get_var_explained(eigvals):
	total = eigvals.sum()
	return eigvals/total


# Plot a skree plot of variance explained
def skree_plot(eigvals):
	var_exp = get_var_explained(eigvals)
	plt.bar(x=["PC{}".format(str(i+1)) for i in range(len(eigvals))], height=var_exp*100)
	plt.show()


def loadings_plot(eigvecs):
	fig, ax = plt.subplots()
	im = ax.imshow(eigvecs, cmap="bwr", norm=colors.CenteredNorm())
	ax.set_xticks(np.arange(eigvecs.shape[1]))
	ax.set_yticks(np.arange(eigvecs.shape[0]))
	ax.set_xticklabels(["PC{}".format(str(i+1)) for i in range(eigvecs.shape[1])])
	ax.set_yticklabels(["feature_{}".format(str(i+1)) for i in range(eigvecs.shape[0])])
	for i in range(eigvecs.shape[1]):
		for j in range(eigvecs.shape[0]):
			text = ax.text(j, i, round(eigvecs[i, j], 2), ha="center", va="center", color="k")
	fig.tight_layout()
	plt.show()


def pca_plot(pc_data, pc1=0, pc2=1, color="b", loadings=None, labels=None, n=5):
	xs = pc_data[:, pc1]
	ys = pc_data[:, pc2]
	scalex = 1.0 / (xs.max() - xs.min())
	scaley = 1.0 / (ys.max() - ys.min())
	plt.scatter(xs * scalex, ys * scaley, c=color)
	if n > loadings.shape[0]:
		n = loadings.shape[0]
	print(n)
	for i in range(n):
		plt.arrow(0, 0, loadings[0, i], loadings[1, i], color='r', alpha=0.5, head_width=0.07, head_length=0.07, overhang=0.7)
		if labels is None:
			plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, "Var" + str(i + 1), color='g', ha='center', va='center')
		else:
			plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, labels[i], color='g', ha='center', va='center')
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.xlabel("PC{}".format(1))
	plt.ylabel("PC{}".format(2))
	plt.grid()
	plt.show()


# Run the whole pca computation using the method of eigendecomposition
def pca_from_eigendecomposition(data, n_comps=None, return_df=True):
	data -= data.mean(axis=0)
	data /= data.std(axis=0)
	eig_vals, eig_vecs = get_pcs(scaled_data=data)
	transformed_data = transform_data(scaled_data=data, vectors=eig_vecs, k=n_comps)
	skree_plot(eigvals=eig_vals)
	pca_plot(transformed_data, loadings=eig_vecs)
	if return_df:
		transformed_data = pd.DataFrame(transformed_data, columns=["PC{}".format(str(i+1)) for i in range(data.shape[1])])
	loadings_plot(eigvecs=eig_vecs)
	return transformed_data


