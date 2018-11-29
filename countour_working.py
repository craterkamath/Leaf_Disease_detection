import argparse
from imageio import imread


import cv2
import numpy as np

import skimage as sk
from skimage.segmentation import morphological_geodesic_active_contour as mgac
from skimage.filters import gaussian
from skimage.transform import resize


import matplotlib.pyplot as plt
import morphsnakes as ms
from pprint import pprint

import warnings

# warnings.filter("ignore")

from feature_aggregation import BagOfWords, LLC, FisherVectors, Vlad


drawing = False


def click_and_crop(event, x, y, flags, param):
	global refPt, cropping, drawing

	def draw():
		cv2.circle(image, tuple(refPt[-1]), 3, (0, 255, 0), -1)
		# image[refPt[-1]] = np.array([0,255,0])
		cv2.imshow("image", image)

	press_and_hold = 1

	if not press_and_hold:
		if event == cv2.EVENT_LBUTTONDOWN:
			# refPt.append(np.array([x, y]))
			pass

		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			refPt.append(np.array([x, y]))
			cropping = False
			draw()

	else:
		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			refPt.append(np.array([x, y]))
			draw()
		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				refPt.append(np.array([x, y]))
				draw()

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False


def visual_callback_2d(background, fig=None):

	# Prepare the visual environment.
	if fig is None:
		fig = plt.figure()
	fig.clf()
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.imshow(background, cmap=plt.cm.gray)

	ax2 = fig.add_subplot(1, 2, 2)
	ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
	plt.pause(0.001)

	def callback(levelset):

		if ax1.collections:
			del ax1.collections[0]
		ax1.contour(levelset, [0.5], colors='r')
		ax_u.set_data(levelset)
		fig.canvas.draw()
		plt.pause(0.001)

		plt.savefig("contour_images/contour_{0}.png".format(args["image"].split(".")[0].split("/")[-1]))

	return callback


def rgb2gray(img):
	return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


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
image = cv2.imread(args["image"])
if args["transform"]:
	image = resize(image, IMG_SIZE)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, start drawing
	elif key == ord("c"):

		cv2.destroyAllWindows()

		input_image = imread(args['image'])

		if args["transform"]:
			input_image = resize(input_image, IMG_SIZE , preserve_range = True)

		imgcolor = input_image / 255.0
		img = rgb2gray(imgcolor)

		gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)

		init_ls = np.zeros(img.shape, dtype=np.int8)
		cv2.fillPoly(init_ls, np.array(
			[np.array(refPt, dtype=np.int32)], dtype=np.int32), 255)

		# Callback for visual plotting
		callback = visual_callback_2d(imgcolor)

		# MorphGAC.
		final_level_set = ms.morphological_geodesic_active_contour(gimg, iterations=10000,
												 init_level_set=init_ls,
												 smoothing=3, threshold=0.4,
												 balloon=-1, iter_callback=callback)
		
		plt.close('all')
		plt.imsave("contour_images/output.png", final_level_set)

		# np.set_printoptions(threshold='nan')

		mask = np.array(final_level_set, dtype=np.uint8) * 255

		# cv2.imshow("Final Mask", final_level_set * 255.)
		# cv2.waitKey(0)

		input_image = imread(args['image'])

		if args["transform"]:

			input_image = resize(input_image, IMG_SIZE, preserve_range = True)

		cv_image = cv2.cvtColor(np.array( input_image, dtype = np.uint8), cv2.COLOR_RGB2BGR)

		src_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
		mask_out = cv2.subtract(src_mask, np.array(cv_image, dtype = np.uint8))
		cropped_image = cv2.subtract(src_mask,mask_out)

		
		# for i in range(50, 5000, 50):
		if(cv2.__version__[0] == '3'):
			surf = cv2.xfeatures2d.SURF_create(1000)
		else:
			surf = cv2.SURF(1000)

		surf_key_points, surf_descriptors = surf.detectAndCompute(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY), mask = mask / 255)

		# print("{0} : {1}".format(i, len(surf_key_points)))

		img_with_keypoints = cv2.drawKeypoints(cropped_image,surf_key_points,None,(255,0,0),4)
		
		cv2.imshow("SURF Keypoints", img_with_keypoints)
		cv2.imwrite("keypoints/surf_keypoints{0}.png".format(args["image"].split(".")[0].split("/")[-1]), img_with_keypoints)


		if(cv2.__version__[0] == '3'):
			sift = cv2.xfeatures2d.SIFT_create()
		else:
			sift = cv2.SIFT()

		sift_keypoints, sift_descriptors = sift.detectAndCompute(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY), mask = mask / 255)

		print("SIFT : {0}".format(len(sift_keypoints)))

		sift_img_with_keypoints = cv2.drawKeypoints(cropped_image,sift_keypoints,None,(255, 0, 0),4)

		cv2.imshow("SIFT Keypoints", sift_img_with_keypoints)
		cv2.imwrite("keypoints/sift_keypoints{0}.png".format(args["image"].split(".")[0].split("/")[-1]), sift_img_with_keypoints)

		cv2.waitKey(0)

		cv2.destroyAllWindows()
		
		# Aggregate those features 
		
		run_cluster = 1

		if run_cluster == 1:
			print "Running Bag of Visual Words"
			bow = BagOfWords(10)
			for i in range(2):
			    for j in range(0, len(sift_descriptors), 20):
			        bow.partial_fit(sift_descriptors[j:j+20])
			faces_bow = bow.transform([sift_descriptors])

		elif run_cluster == 2:
		    print "Running Vlad"
		    vlad = Vlad(10)

		    for i in range(2):
		        for j in range(0,len(sift_descriptors), 10):
		            vlad.partial_fit(sift_descriptors[j:j+10])

		    faces_vlad = vlad.transform(np.array([sift_descriptors]))


		# SIFT descriptors

		import pdb 
		pdb.set_trace()

		break

cv2.destroyAllWindows()
