# -*- coding: utf-8 -*-

"""
Python script with functions for creating, saving and loading Colormaps in Matplotlib
"""

# Built-in/Generic Imports
import glob
import os

# Libs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import colorsys

# Tags
__author__ = 'Adam Klie'


def load_cmap_from_csv(filename, column, cmap_type):
	"""Load a Matplotlib Colormap from a CSV file
	Parameters
	----------
	filename : str
		path to csv file to load
	column : str or int
		column name or number for hexidecimal or rgb values for each color
	cmap_type : str
		Matplotlib Colormap type.
		Currently supports LinearSegmentedColormap and ListedColormap
	"""
	if cmap_type == 'LinearSegmentedColormap':
		return LinearSegmentedColormap.from_list("mycmap", pd.read_csv(filename)[column].values)
	elif cmap_type == 'ListedColormap':
		return ListedColormap(pd.read_csv(filename)[column].values)


def load_cmap_from_cpt(cptfile, name=None):
	"""Read a GMT color map from a cpt file
	Parameters
	----------
	cptfile : str or open file-like object
		path to .cpt file
	name : str, optional
		name for color map
		if not provided, the file name will be used
	"""
	with open(cptfile, 'r') as cptf:
		return gmtColormap_openfile(cptf, name=name)


def gmtColormap_openfile(cptf, name=None):
	"""Read a GMT color map from an OPEN cpt file
	Parameters
	----------
	cptf : open file or url handle
		path to .cpt file
	name : str, optional
		name for color map
		if not provided, the file name will be used
	"""
	# generate cmap name
	if name is None:
		name = '_'.join(os.path.basename(cptf.name).split('.')[:-1])

	# process file
	x = []
	r = []
	g = []
	b = []
	lastls = None
	for l in cptf.readlines():
		ls = l.split()

		# skip empty lines
		if not ls:
			continue

		# parse header info
		if ls[0] in ["#", b"#"]:
			if ls[-1] in ["HSV", b"HSV"]:
				colorModel = "HSV"
			else:
				colorModel = "RGB"
			continue

		# skip BFN info
		if ls[0] in ["B", b"B", "F", b"F", "N", b"N"]:
			continue

		# parse color vectors
		x.append(float(ls[0]))
		r.append(float(ls[1]))
		g.append(float(ls[2]))
		b.append(float(ls[3]))

		# save last row
		lastls = ls

	x.append(float(lastls[4]))
	r.append(float(lastls[5]))
	g.append(float(lastls[6]))
	b.append(float(lastls[7]))

	x = np.array(x)
	r = np.array(r)
	g = np.array(g)
	b = np.array(b)

	if colorModel == "HSV":
		for i in range(r.shape[0]):
			# convert HSV to RGB
			rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
			r[i] = rr;
			g[i] = gg;
			b[i] = bb
	elif colorModel == "RGB":
		r /= 255.
		g /= 255.
		b /= 255.

	red = []
	blue = []
	green = []
	xNorm = (x - x[0]) / (x[-1] - x[0])
	for i in range(len(x)):
		red.append([xNorm[i], r[i], r[i]])
		green.append([xNorm[i], g[i], g[i]])
		blue.append([xNorm[i], b[i], b[i]])

	# return colormap
	cdict = dict(red=red, green=green, blue=blue)
	return matplotlib.colors.LinearSegmentedColormap(name=name, segmentdata=cdict)


def get_cmap(name, filepath='./matplotlib_cmaps', file_type='csv', column='RGB Hex',
             cmap_type='LinearSegmentedColormap'):
	"""
	Load a Colormap object from a name
	Parameters
	----------
	name : str
		name for color map, must be same as file name without extension
	filepath : str
		name for location of colormap files. Files may be in subdirectories of this path
	file_type : str
		Type of file.
		Currently supports .csv and .cpt files
	column : str
		For csv files, the column name or number to use for colors (hexidecimal)
	cmap_type : str
		For csv files, the Colormap type to load.
		Currently supports LinearSegmentedColormap and ListedColormap
	"""
	if file_type == 'csv':
		files = glob.glob(filepath + '/**/' + name + '.csv', recursive=True)
		if len(files) > 0:
			params = {'column': column, 'cmap_type': cmap_type}
			return load_cmap_from_csv(files[0], **params)
	elif file_type == 'cpt':
		files = glob.glob(filepath + '/**/' + name + '.cpt', recursive=True)
		if len(files) > 0:
			return load_cmap_from_cpt(files[0])
	else:
		print('Could not find Colormap file with name: {}.{} in {}'.format(name, file_type, filepath))


def save_cmap_to_cpt(cmap, vmin=0, vmax=1, N=255, filename="test.cpt", **kwargs):
	"""
	Parameters
	----------
	cmap : str
		Matplotlib Colormap to save to a file
	vmin : int
		Minimum value of Colormap
	vmax : int
		Maximum value of Colormap
	N : int
		Number of colors betwwen
	filename : str
		Name of the file to save the Colormap to
	kwargs : dict
		keyword arguments
	"""
	# create string for upper, lower colors
	b = np.array(kwargs.get("B", cmap(0.)))
	f = np.array(kwargs.get("F", cmap(1.)))
	na = np.array(kwargs.get("N", (0, 0, 0))).astype(float)
	ext = (np.c_[b[:3], f[:3], na[:3]].T * 255).astype(int)
	extstr = "B {:3d} {:3d} {:3d}\nF {:3d} {:3d} {:3d}\nN {:3d} {:3d} {:3d}"
	ex = extstr.format(*list(ext.flatten()))
	# create colormap
	cols = (cmap(np.linspace(0., 1., N))[:, :3] * 255).astype(int)
	vals = np.linspace(vmin, vmax, N)
	arr = np.c_[vals[:-1], cols[:-1], vals[1:], cols[1:]]
	# save to file
	fmt = "%e %3d %3d %3d %e %3d %3d %3d"
	np.savetxt(filename, arr, fmt=fmt,
	           header="# COLOR_MODEL = RGB",
	           footer=ex, comments="")


def plot_examples(colormaps):
	"""
	Helper function to plot data with associated colormap.
	"""
	np.random.seed(19680801)
	data = np.random.randn(30, 30)
	n = len(colormaps)
	fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
	                        constrained_layout=True, squeeze=False)
	for [ax, cmap] in zip(axs.flat, colormaps):
		psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
		fig.colorbar(psm, ax=ax)
	plt.show()
