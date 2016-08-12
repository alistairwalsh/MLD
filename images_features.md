# Image and feature analysis

Let's start by loading some libraries we'll need:

	import numpy as np

	# Optional for plotting
	import matplotlib.pyplot as plt

	# Optional for notebook display
	%matplotlib inline


## Load the dataset

For the convenience of this workshop the images have already been loaded into a numpy array format. 

	data = np.load('data.npz')
	images = data['images']
	labels = data['labels']

For now we're going to focus on the data stored in the images - we'll come back to the labels and how we use them in the next section.



## Image Properties

Let take a look at how our image data is stored. You will generally spend more time cleaning and manipulating your data than actually creating a model, so it's important to understand how NumPy is used. 

Lets get started with NumPy's functions to manipulate and understand the data we're working with. To demonstrate this, we'll use use NumPy's "shape" and "dtype" commands to take a closer look at the rectangular tag image we just read in:

	print (images.shape)
	print (images.dtype)	

The images are stored in a 3D array of numbers. The first axis is the 730 individual images, the remaining two axes are the rows and columns of the pixels respectively.

Just like Python builtins we can index and slice these arrays:

	# Get the first image:
	first_image = images[0] # Or equivalently: images[0, :, :]
	print(first_image.shape)

	# Get first three images:
	several_images = images[:3]
	print(several_images.shape)


## Cropping

One of the things you've probably noticed is that there's a dark area around the edges of the tags. As we're only interested in the pattern in the middle of the tags, we should try to crop this out. 

	cropped_image = first_image[4:20,4:20]

	plt.figure(figsize = (10, 7))
	plt.title('Cropped Image')
	plt.axis('off')
	plt.imshow(cropped_image,  cmap = 'gray')


## Feature Engineering

When people think of machine learning, the first thing that comes to mind tends to be the  algorithms that will train the computer to solve your problem. Of course this is important, but even more important is the way you process the data you'll eventually feed into the machine learning algorithm. You will generally spend much more time with your data and preprocessing than your model.

Now, when most people think of features in data, they think that this is what it is:

	plt.figure(figsize = (10, 7))
	plt.title('Rectangle Tag')
	plt.axis('off')
	plt.imshow(first_image,  cmap = 'gray')

In fact this is not actualy the case. In the case of this dataset, the features are actually the pixel values that make up the images - those are the values we'll be training the machine learning algorithm with:

	print(first_image)


## Representing your data for scikit-learn

Scikit-learn expects your training data to be in a particular format. Data should be represented as a 2D array: each row is the data for a single example (ie, each row should be the pixels from a single image), each column is the value for each variable we are using to represent that feature.

But our image data is not currently in that format, so let's fix that:

	X = np.reshape(images, (730, -1))
	print(X.shape)

X is the naming convention used in scikit-learn calls and documentation to represent the data that is passed into a machine learning algorithm.


## Other transformations: Brightness and Contrast

Modifying the brightness and contrast of our images is a surprisingly simple task, but can have a big impact on the appearance of the image. Here is how you can increase and decrease these characteristics in an image:

Characteristic            |  Increase/Decrease   | Action
:-------------------------:|:-------------------------:|:-------------------------
Brightness | Increase | Add an integer to every pixel
Brightness    | Decrease | Subtract an integer from every pixel
Constrast      | Increase | Multiply every pixel by a number greater than 1
Constrast      | Decrease | Multiple every pixel by a floating point number less than 1

Now we can see how this affects our rectangular tag image. Again, feel free to experiment with different values in order to see the final effect.

	increase_brightness = first_image + 30
	decrease_brightness = first_image - 30
	increase_contrast = first_image * 1.5
	decrease_contrast = first_image * 0.5

	brightness_compare = np.hstack((increase_brightness, decrease_brightness))
	constrast_compare = np.hstack((increase_contrast, decrease_contrast))

	plt.figure(figsize = (15, 12))
	plt.title('Brightness')
	plt.axis('off')
	plt.imshow(brightness_compare, cmap='gray') 

	plt.figure(figsize = (15, 12))
	plt.title('Contrast')
	plt.axis('off')
	plt.imshow(constrast_compare, cmap='gray')


# Questions/Exercises

1. Why do we have to be careful with the datatype for numerical operations?