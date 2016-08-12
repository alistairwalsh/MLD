# Building a machine learning program

In this section we put together everything we learned about images and features so that we can train a machine learning algorithm to distinguish between the images of different tags.

	# Default imports
	import numpy as np

	# Optional for plotting and notebooks
	import matplotlib.pyplot as plt
	%matplotlib inline


## Image Labels

If you recall from earlier, the three different tag types indicated a different group in the experiment.

Tag Number             |  Pattern   | Details
:-------------------------:|:-------------------------: | :-------------------------:
0  |  Rectangle | 100 bees that were the control group
1  |  Circle    | 100 bees that were treated with caffeine
2  | Blank      | Single queen in the colony received this tag


This is represented in our labels variable we loaded along with the images.

	labels = data['labels']

	print(labels.shape)
	print(labels[::100])

So our labels array is one dimensional. There is one label for each image, and each integer corresponds to a different image class.

By convention we will refer to this as y, to correspond with our data variable X

	y = labels


## Splitting a training and testing dataset

Ok, now that our data is ready, we have one more thing to consider before we can train our machine learning program: testing. How do I know that my program will be more accurate if I change the brightness and contrast? How do I that after all this work, my program is doing no better than random chance at determining which tag is in an image?

The solution to this is to split up our data into two segments: a training set and a testing set. The training set is what we will allow our machine learning program to learn from, while the testing set allows us to then see how accurate the program is with data that it has not been exposed to.

	from sklearn.cross_validation import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)
	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

We now have 584 images that we will train our program on and 146 images to test the accuracy of its predictions. Each of the pixels that each image has is now a feature or dimension that we can train our machine learning program on.


## Feature Engineering

Now we have our images, let's use some of the skills we learned earlier to apply some manipulations to the images. We can play with this a lot, and it's often the section you'll come back to once you test out the classification system you train to see if you can improve performance.


# Visualising the Data

Now that we have hundreds of images, each with hundreds of dimensions, we need to try to find a way to visualise all this data. To acheive this, we'll try to use a dimensionality reduction technique called PCA. PCA is an unsupervised technique which tries to collapse the number of dimensions in your data down by looking for variables with the highest variation. Here, we'll try to collapse the data down to just 2 dimensions:

	from sklearn.decomposition import PCA

	pca = PCA(n_components=2)
	fit_trans_X = pca.fit(X).transform(X)
	plt.figure(figsize = (35, 20))
	plt.scatter(fit_trans_X[:, 0], fit_trans_X[:, 1], c=y, s=400)

This looks promising, it looks like PCA was able to separate out the clusters which correpsond to the different image types in our data. Things are looking good, but I think we can do better with a supervised dimensionality techinuqe called LDA. LDA is very similar to PCA, except we tell it what groups we want to separate out with our data and it looks for the variation which will help us achieve this.

	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

	lda = LDA(n_components=2)
	lda_model = lda.fit(X_train, y_train)
	X_trans = lda_model.transform(X_train)
	plt.figure(figsize = (35, 20))
	plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_train, s=400)

That's looking really good now, we have 3 neat clusters for each of the tag types. Now we can can try to use the data output by LDA to train a machine learning algorithm called a support vector machine (SVM).

## SVM Classification

A support vector machine is a machine learning techinque which tries to separate out classes by working out lines which separate out the different groups. Once we have trained an SVM, it will try to use these lines to predict which class a new datapoint should belong to. One of the really powerful things about this technique is that while the image below shows it separating out groups in two dimensions, it can work with data that has so many dimensions we have difficulty visualising it.

![](images/svm.png)

Below we will train an SVM and experiment a little with a couple of the different parameters.

	from sklearn import svm

	clf = svm.SVC(gamma=0.0001, C=10)
	clf.fit(X_trans, y_train)

## Accuracy

Now that we have trained our SVM on our LDA transformed dataset, we should transform our testing dataset and use the SVM to make predictions:

	transform_testing_set = lda.transform(X_test)
	y_pred = clf.predict(transform_testing_set)

Great, we have now successfully made some predictions with the testing dataset, but how do we tell if any of them were right? To help with this, we can use scikit-learn's metrics to evaluate how accurate our predictions were. This will give us a number between 0 and 1 which will tell us if it got 0% of the predictions correct all the way through to 100%.

	from sklearn import metrics

	print (metrics.accuracy_score(y_test, y_pred))

Congratulations, you have just trained and evaluated your first classifier.

# Exercises

1. Replace the SVM classifier with any other classifier from the scikit-learn library.
2. Compare the classification performance both with and without LDA.
3. What happens if we evaluate our performance on the training set?