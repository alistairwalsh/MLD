# Pipelines and Validation

So far we have:

- Loaded and reshaped images to use the pixel values as features for our classifier
- Used LDA and PCA to transform our raw pixel data into a better representation
- Experimented with a few different classifiers
- Evaluated performance using a held out test set

Note though that if we put everything we have done together the code looks something like this:

    # Load the images and labels

    # Reshape the images into one row per image

    # Train an LDA transformer on the reshaped images

    # Train a classifier on this data

    # Evaluate on the test set


There's two things to note about this:

1. We've made quite a few decisions along the way, including the number of components in the LDA/PCA transforms and the regularisation parameter C for the support vector machine. How do we know these numbers are any good?
2. This doesn't look like much fun to code and maintain.

Fortunately we can tackle both of these concerns using the scikit-learn API.


# Building pipelines



## Adding your own transformer

To add your own transformer that works with the rest of scikit-learny you need to make a class that implements a few methods and comply with a some simple constraints:

1. It should implement the following methods: fit, transform and predict.
2. Each of these methods should take two arguments: X, Y
3. The class initialisation (__init__) should only assign to attributes (no *args or **kwargs).
4. Inherit from the appropriate mixin for your transformer.


Lets build an example for the unwrapping step:

    from sklearn.base import TransformerMixin

    class Unwrap(TransformerMixin):

        def __init__(self):
            return self

        def fit(X, y=None):
            return self

        def transform(X, y=None):
            rows, cols, depth = X.shape
            return X.reshape((rows, cols*depth))


And now we can combine it together with the rest of the pipeline:

    # Unwrapper --> LDA --> SVM pipeline

    # Build a model to show it works just the same for training

    # Use it to classify the same testing set as before, but in the original shape

You can use this kind of pattern as a bridge between the particular input data you have to the format scikit-learn expects. For example, the above code have taken a list of images, loaded them into numpy arrays then unwrapped them. The advantage of doing this is that we can collect all of the logic of our pipeline into a single class. The pipeline object with our custom transformer can be used wherever would we have used one of scikit-learn's inbuilt classes.


# Choosing parameters

Now we have our whole process from image data to classification wrapped up in a single pipeline. Calling fit on this model with input data will start with the first phase of the pipeline, call fit on that phase, call transform and then pass that on to the next phase. This is repeated until the end, where we get the final output.

We can now think of our model as a single unit, with different parameters for each component in the pipeline. When we're trying to build a system, we care about the how the overall system behaves... Choosing these parameters is a challenging problem. A good baseline method for choosing these hyper parameters is [randomised search.](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

We can conduct a random search like this:

    from sklearn.grid_search import RandomizedSearchCV

To run a random search we specify ranges of values for each parameter in the model. The RandomSearch algorithm selects random combinations for each value and runs the pipeline from beginning to end with those values. Note that the RandomSearch takes a classifier, and in this case we fed in our pipeline of operations from earlier.

The trained classifier at the end is the RandomSearch object, we can call predict on this object to classify the final objects.
