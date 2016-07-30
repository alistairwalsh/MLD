# Pipelines and Validation

So far we have:

- Loaded and reshaped images to use the pixel values as features for our classifier
- Used LDA and PCA to transform our raw pixel data into a better representation
- Experimented with a few different classifiers
- Evaluated performance using a held out test set

There's two things to note about this:

1. We've made quite a few decisions along the way, including the number of components in the LDA/PCA transforms and the regularisation parameter C for the support vector machine. How do we know these numbers are any good?
2. This doesn't look like much fun to code and maintain.

Fortunately we can tackle both of these concerns using the scikit-learn API.


# Building pipelines

A scikit-learn [pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is a tool to compose a series of scikit-learn and compatible objects. The resulting object behaves like a single classifier with fit, predict and other methods as appropriate.

We can build a pipeline combining our LDA transformation and SVM classifier like so:

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    # Start by creating the individual steps in the pipeline
    lda = LDA(n_components=20)
    svc = LinearSVC()

    pipeline = Pipeline([('lda', lda),
                         ('svc', svc)])

    pipeline.fit(transformed_pixels, labels)

    pipeline.score(held_out, held_out_labels)


# Adding your own transformer

To add your own transformer that works with the rest of scikit-learn you need to make a class that implements a few methods and comply with some constraints:

1. It should implement the following methods: fit, transform and predict.
2. Each of these methods should take two arguments: X, Y
3. The class initialisation (__init__) should only assign to attributes (no *args or **kwargs).
4. Inherit from some utility classes.

Lets build an example for the unwrapping step:

    from sklearn.base import TransformerMixin, BaseEstimator

    class CropUnwrap(TransformerMixin, BaseEstimator):

        def __init__(self, crop_pixels):
            self.crop_pixels = crop_pixels
            return None

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_crop = X[:, 
                       self.crop_pixels:24-self.crop_pixels, 
                       self.crop_pixels:24-self.crop_pixels]
            rows, cols, depth = X.shape
            return X.reshape((rows, cols*depth))


And now we can combine it together with the rest of the pipeline:
    
    unwrap = CropUnwrap()
    lda = LDA(n_components=20)
    svm = LinearSVC()

    pipeline = Pipeline([('unwrap', unwrap),
                         ('lda', lda),
                         ('svm', svm)])

    pipeline.fit(images, labels)

You can use this kind of pattern as a bridge between the particular input data you have to the format scikit-learn expects. For example, the above code have taken a list of images, loaded them into numpy arrays then unwrapped them. The advantage of doing this is that we can collect all of the logic of our pipeline into a single class. The pipeline object with our custom transformer can be used wherever would we have used one of scikit-learn's inbuilt classes.


# Choosing parameters

Now we have our whole process from image data to classification wrapped up in a single pipeline. Calling fit on this model with input data will start with the first phase of the pipeline, call fit on that phase, call transform and then pass that on to the next phase. This is repeated until the end, where we get the final output.

We can now think of our model as a single unit, with different parameters for each component in the pipeline. When we're trying to build a system, we really care about the how the overall system behaves. Choosing these parameters is a challenging problem. A good baseline method for choosing these hyper parameters is [randomised search.](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

We can conduct a random search like in the following code. To specify the parameters for each element of the pipeline we use the name we specified for that element of the pipeline, followed by double underscore then the actual parameter name.

    from sklearn.grid_search import RandomizedSearchCV

    search_range = {'lda__n_components': [5, 10, 15, 20, 30, 50],
                    'unwrap__crop_pixels': [0, 2, 4, 6, 8],
                    'svm__C': [1, 10, 100, 1000, 10e3, 10e4]}

    searcher = RandomizedSearchCV(pipeline, search_range, n_iter=20)

    searcher.fit(images, labels)


To run a random search we specify ranges of values for each parameter in the model. The RandomSearch algorithm selects random combinations for each value and runs the pipeline from beginning to end with those values. Note that the RandomSearch takes a classifier, and in this case we fed in our pipeline of operations from earlier.

The trained classifier at the end is the RandomSearch object, we can call predict on this object to classify the final objects.


# Final validation

Remember the test set we held out? Now we can use it for a final validation. Note that the scores from the randomised search are likely to be optimistic since we've selected the best combination of parameters for that dataset --- only by testing on data that we didn't use for training can we get an accurate measure of our performance. 

    searcher.score(test_images, test_labels)

