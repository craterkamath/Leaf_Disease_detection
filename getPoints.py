import argparse
import cv2
import numpy as np
import skimage as sk
from skimage.segmentation import morphological_geodesic_active_contour as mgac
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import morphsnakes as ms
from imageio import imread


drawing = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, drawing
 

	def draw():
			cv2.circle(image, tuple(refPt[-1]) ,3 , (0, 255, 0), -1)
			# image[refPt[-1]] = np.array([0,255,0])
			cv2.imshow("image", image)


	press_and_hold = 1

	if not press_and_hold:
		if event == cv2.EVENT_LBUTTONDOWN:
			# refPt.append(np.array([x, y]))
			pass
	 
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
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




def store_evolution_in(lst):
	"""Returns a callback function to store the evolution of the level sets in
	the given list.
	"""

	def _store(x):

		# plt.imshow(x, cmap = 'gray')
		lst.append(np.copy(x))

	return _store

def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    
    """
    
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

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


refPt = []
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 


# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, start drawing
	elif key == ord("c"):
		

		'''
		cnt = np.array(refPt).copy()
		hull = cv2.convexHull(np.array(refPt))
		
		defects = cv2.convexityDefects(np.array(refPt),hull)

		for i in range(defects.shape[0]):
		    s,e,f,d = defects[i,0]
		    start = tuple(cnt[s][0])
		    end = tuple(cnt[e][0])
		    far = tuple(cnt[f][0])
		    cv2.line(image,start,end,[0,255,0],2)
		    cv2.circle(image,far,5,[0,0,255],-1)

		cv2.destroyAllWindows()
		cv2.imshow("Image", image)
		cv2.waitKey(0)
		'''
		cv2.destroyAllWindows()

		# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 2

		# sharp_image = cv2.filter2D(clone, -1, kernel)
		img = cv2.cvtColor(cv2.GaussianBlur(clone,(3,3),0), cv2.COLOR_BGR2GRAY)
		# cv2.drawContours(img, hull, -1, (255, 255, 0), 5)
		
		# import pdb;pdb.set_trace()
		# cv2.polylines(img, np.array([np.array(refPt, dtype = np.int32)], dtype = np.int32), 3, (0,0,255))

		cv2.imshow("Image", img)
		cv2.waitKey(0)

		cv2.destroyAllWindows()


		iterations = []
		callback = store_evolution_in(iterations)

		img = np.array(img, dtype = np.float64)
		# cv2.imshow("Output", img.astype(np.uint8))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		'''
		init_ls = np.zeros(img.shape, dtype=np.int8)
		cv2.fillPoly(init_ls, np.array([np.array(refPt, dtype = np.int32)], dtype = np.int32), 255)
		
		output = mgac(img, 100, init_ls, balloon=-1,iter_callback=callback)

		print(output.shape)
		
		for i in range(0, len(iterations), 10):
			print(i)
			plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
			plt.contour(iterations[i], [0.5], color = 'r')
			plt.show()
			# cv2.imshow("Output", iterations[i].astype(np.uint8))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		'''

		'''
		from skimage.segmentation import active_contour

		# import pdb; pdb.set_trace()
		snake = active_contour(img, np.array(refPt), alpha = 0.1, beta = 0.05, w_line = 0, w_edge = 8, max_px_move=1.0, max_iterations=30000, convergence = 0.05)
		init = np.array(refPt)

		fig, ax = plt.subplots(figsize=(7, 7))
		ax.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
		ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
		ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
		ax.set_xticks([]), ax.set_yticks([])
		ax.axis([0, img.shape[1], img.shape[0], 0])

		plt.show()

		'''

		
		

		imgcolor = imread(args['image']) / 255.0
		img = rgb2gray(imgcolor)

		# g(I)
		gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)

		# Initialization of the level-set.
		# init_ls = ms.circle_level_set(img.shape, (163, 137), 135)

		init_ls = np.zeros(img.shape, dtype=np.int8)
		cv2.fillPoly(init_ls, np.array([np.array(refPt, dtype = np.int32)], dtype = np.int32), 255)

		# Callback for visual plotting
		callback = visual_callback_2d(imgcolor)

		# MorphGAC. 
		ms.morphological_geodesic_active_contour(gimg, iterations=100, 
		                                         init_level_set=init_ls,
		                                         smoothing=2, threshold=0.3,
		                                         balloon=-1, iter_callback=callback)



		break

cv2.destroyAllWindows()