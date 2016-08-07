# Image and feature analysis

Let's start by loading the libraries we'll need:

	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	%matplotlib inline


## Extract Images

Included in these workshop materials is a compressed file ("data.tar.gz") containg the images that we'll be classifying today. Once you extract this file, you should have a directory called "data" which contains the following directories:

Directory            |  Contents
:-------------------------:|:-------------------------:
I | Contains rectangle tag images
O    | Contains circle tag images
Q      | Contains blank tag images

Feel free to have a look through these directories, and we'll show you how to load these images into Python using OpenCV next.

## Reading Images

We're now going to be using OpenCV's "imread" command to load one of the images from each type of tag into Python and then use Matplotlib to plot the images:

	rect_image =    cv2.imread('data/I/27.png', cv2.IMREAD_GRAYSCALE)
	circle_image =  cv2.imread('data/O/11527.png', cv2.IMREAD_GRAYSCALE)
	queen_image =   cv2.imread('data/Q/18027.png', cv2.IMREAD_GRAYSCALE)

	plt.figure(figsize = (10, 7))
	plt.title('Rectangle Tag')
	plt.axis('off')
	plt.imshow(rect_image,  cmap = cm.Greys_r)

	plt.figure(figsize = (10, 7))
	plt.title('Circle Tag')
	plt.axis('off')
	plt.imshow(circle_image,  cmap = cm.Greys_r)

	plt.figure(figsize = (10, 7))
	plt.title('Queen Tag')
	plt.axis('off')
	plt.imshow(queen_image,  cmap = cm.Greys_r)


## Image Properties

One of the really useful things about using OpenCV to manipulate images in Python is that all images are treated as NumPy matrices. This means we can use NumPy's functions to manipulate and understand the data we're working with. To demonstrate this, we'll use use NumPy's "shape" and "dtype" commands to take a closer look at the rectangular tag image we just read in:

	print (rect_image.shape)
	print (rect_image.dtype)	

This holds the same values, which is good. When you're working with your own datasets in the future, it would be highly beneficial to write your own little program to check the values and structure of your data to ensure that subtle bugs don't creep in to your analysis.


## Cropping

One of the things you've probably noticed is that there's a dark area around the edges of the tags. As we're only interested in the pattern in the middle of the tags, we should try to crop this out. Have a little play with the code below and experiment with different pixel slices.


	cropped_rect_image = rect_image[4:20,4:20]
	cropped_circle_image = circle_image[4:20,4:20]
	cropped_queen_image = queen_image[4:20,4:20]

	plt.figure(figsize = (10, 7))
	plt.title('Rectangle Tag ' + str(cropped_rect_image.shape))
	plt.axis('off')
	plt.imshow(cropped_rect_image,  cmap = cm.Greys_r)

	plt.figure(figsize = (10, 7))
	plt.title('Circle Tag ' + str(cropped_circle_image.shape))
	plt.axis('off')
	plt.imshow(cropped_circle_image,  cmap = cm.Greys_r)

	plt.figure(figsize = (10, 7))
	plt.title('Queen Tag ' + str(cropped_queen_image.shape))
	plt.axis('off')
	plt.imshow(cropped_queen_image,  cmap = cm.Greys_r)


## Feature Engineering

When people think of machine learning, the first thing that comes to mind tends to be the fancy algorithms that will train the computer to solve your problem. Of course this is important, but the reality of the matter is that the way you process the data you'll eventually feed into the machine learning algorithm is often the thing you'll spend the most time doing and will have the biggest effect on the accuracy of your results.

Now, when most people think of features in data, they think that this is what it is:

	plt.figure(figsize = (10, 7))
	plt.title('Rectangle Tag')
	plt.axis('off')
	plt.imshow(rect_image,  cmap = cm.Greys_r)

In fact this is not actualy the case. In the case of this dataset, the features are actually the pixel values that make up the images - those are the values we'll be training the machine learning algorithm with:

	print(rect_image)


So what can we do to manipulate the features in out dataset? We'll explore three methods to acheive this:

1. Image smoothing
2. Modifying brightness
3. Modifying contrast

Techniques like image smoothing can be useful when improving the features you train the machine learning algorithm on as you can eliminate some of the potential noise in the image that could confuse the program.

## Smoothing

Image smoothing is another name for blurring the image. It involves passing a rectangular box (called a kernel) over the image and modifying pixels in the image based on the surrounding values.

As part of this exercise, we'll explore 3 different smoothing techniques:

Smoothing Method            |  Explanation
:-------------------------:|:-------------------------:
Mean | Replaces pixel with the mean value of the surrounding pixels
Median    | Replaces pixel with the median value of the surrounding pixels
Gaussian      | Replaces pixel by placing different weightings on surrrounding pixels according to the gaussian distribution


## Brightness and Contrast

Modifying the brightness and contrast of our images is a surprisingly simple task, but can have a big impact on the appearance of the image. Here is how you can increase and decrease these characteristics in an image:

Characteristic            |  Increase/Decrease   | Action
:-------------------------:|:-------------------------:|:-------------------------
Brightness | Increase | Add an integer to every pixel
Brightness    | Decrease | Subtract an integer from every pixel
Constrast      | Increase | Multiply every pixel by a number greater than 1
Constrast      | Decrease | Multiple every pixel by a floating point number less than 1

Now we can see how this affects our rectangular tag image. Again, feel free to experiment with different values in order to see the final effect.

	increase_brightness = rect_image + 30
	decrease_brightness = rect_image - 30
	increase_contrast = rect_image * 1.5
	decrease_contrast = rect_image * 0.5

	brightness_compare = np.hstack((increase_brightness, decrease_brightness))
	constrast_compare = np.hstack((increase_contrast, decrease_contrast))

	plt.figure(figsize = (15, 12))
	plt.title('Brightness')
	plt.axis('off')
	plt.imshow(brightness_compare, cmap = cm.Greys_r) 

	plt.figure(figsize = (15, 12))
	plt.title('Contrast')
	plt.axis('off')
	plt.imshow(constrast_compare, cmap = cm.Greys_r)

## Module Summary

In this section we have covered:

* Reading images
* Image properties
* Feature engineering
* Image smoothing
* Brightness/constrast operations

In the next section of this workshop we'll cover how to put these skills together to train a machine learning algorithm to recognise these images.