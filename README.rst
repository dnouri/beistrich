Abstract
========

beistrich tries to predict where to put commas in sentences.  I
personally make a lot of errors when putting commas in German
sentences.  So the idea was born to try and create a machine learning
model that can tell me where to put commas.

The best results with the current model, with a training set of 225000
cases, that has twice as many cases without a comma as with a comma,
the ``f1-score`` is **0.89**.

::

               precision    recall  f1-score   support

 training set       0.93      0.93      0.93    225000

            0       0.91      0.93      0.92     50000
            1       0.86      0.82      0.84     25000

  avg / total       0.89      0.89      0.89     75000

  Confusion matrix:
  [[46657  3343]
   [ 4545 20455]]


Installation
============

Install from source with `pip <http://www.pip-installer.org>`_:

.. code-block:: bash

  $ pip install .

Install the latest released version from PyPI:

.. code-block:: bash

  $ pip install beistrich

beistrich does not declare ``numpy`` or ``scipy`` as dependencies.  So
you may have to install these separately *before* installing beistrich:

.. code-block:: bash

  $ pip install numpy
  $ pip install scipy

beistrich also expects you to have the Stanford Tagger installed.
After installation, you'll have to adjust the ``claspath`` and
``stanford_models`` environment variables in ``beistrich.ini`` to
point to the location of ``stanford-postagger.jar`` and the
``models/`` directory in your Stanford Tagger installation.


Usage
=====

create
------

The first step is to download and create a dataset from Gutenberg
books online.  To do this, run:

.. code-block:: bash

  $ beistrich-dataset create beistrich.ini

This will download books, process them, and create files
``data/X.npy`` and ``data/y.npy``.


stratify
--------

The dataset created through ``create`` has many more cases *with* a
comma than without a comma.  The first number in the ``bincount`` here
is the number of training cases without a comma:

.. code-block:: bash

  $ beistrich-dataset introspect beistrich.ini 
  data/y.npy                    :  1478815  (bincount: [1363410, 115405])

Let's stratify the dataset, so we'll get better results when doing
training later:

.. code-block:: bash

  $ beistrich-dataset stratify beistrich.ini 

``introspect`` will now show us the stratified ``y`` matrix, which has
twice as many training cases with comma:

.. code-block:: bash

  $ beistrich-dataset introspect beistrich.ini 
  data/y-strat-large.npy        :   300000  (bincount: [200000, 100000])
  data/y.npy                    :  1478815  (bincount: [1363410, 115405])


report
------

We're now ready to actually train a model.  ``report`` will give us a
report on the result of our training:

.. code-block:: bash

  $ beistrich-learn report lr beistrich.ini


search, curve and analyze
-------------------------

The ``search`` command allows you to run a grid search to find the
best hyperparameters for the model.

The ``curve`` command will plot a learning curve, and thus help you
find out if the model is suffering from high bias or high variance.

The ``analyze`` command displays a list of test cases for which the
model made the best predictions (i.e. those cases where the estimated
probability was closest to the actual class), and the worst
predictions (where predictions were off).

You can call these commands just like you call ``report``:

.. code-block:: bash

  $ beistrich-learn search lr beistrich.ini
  $ beistrich-learn curve lr beistrich.ini
  $ beistrich-learn analyze lr beistrich.ini

If you wanna tune the models, take a look at the models and their
parameters (specifically ``default_params`` and
``grid_search_params``) in ``beistrich/model.py``.


train and correct
-----------------

Once you're happy with your model it's time to save it:

.. code-block:: bash

  $ bin/beistrich-learn train lr beistrich.ini
  Saved file to data/model.pickle

And finally, you can use it to correct sentences:

.. code-block:: bash

  $ bin/beistrich-learn correct beistrich.ini 

The text to correct lives in the ``beistrich.ini`` configuration file.
