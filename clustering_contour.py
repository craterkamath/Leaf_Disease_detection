import argparse
from imageio import imread


import cv2
import numpy as np

import skimage as sk
from skimage.segmentation import morphological_geodesic_active_contour as mgac
from skimage.filters import gaussian
from skimage.transform import resize

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import morphsnakes as ms
from pprint import pprint

import warnings

# warnings.filter("ignore")

from feature_aggregation import BagOfWords, LLC, FisherVectors, Vlad



refPt = []
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-t", "--transform", help = "Do you want resizing?", action = "store_true")
ap.add_argument("-r", "--resize", help = "Resizes the image into a x b x 3", type = int, nargs = 2 , default = [256, 256])
args = vars(ap.parse_args())

IMG_SIZE = (args['resize'][0],args['resize'][1] ,3)
GRAY_IMAGE_SIZE = (256, 256)

# load the image, clone it, and setup the mouse callback function
image = imread(args["image"])
if args["transform"]:
	image = resize(image, IMG_SIZE)
clone = image.copy()

n_clusters = 3

for n_clusters in range(2, 4):
	model = KMeans(n_clusters = n_clusters)
	X = image.reshape(image.shape[0] * image.shape[1], 3)
	model.fit(X)
	results = model.labels_

	mask = results.reshape(image.shape[:2])
	filtr = np.zeros(image.shape)
	for i in range(len(filtr)):
		for j in range(len(filtr[i])):
			filtr[i,j,mask[i,j]] = 255.


	cv2.imwrite("cluster_output/{1}_{0}_cluster.png".format(str(n_clusters), args['image'].split("/")[-1].split(".")[0]), filtr)
# cv2.imshow("Clustering mask", mask)
# cv2.waitKey(0)
# import pdb
# pdb.set_trace()

