Quickstart
################

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import LLMlight

    # Retrieve URLs of malicous and normal urls:
    X, y = LLMlight.load_example()

    # Learn model on the data
    model = LLMlight.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = LLMlight.plot(model)



Learn new model with gridsearch and train-test set
################################################################

AAA

.. code:: python

    # Import library
    import LLMlight

    # Load example data set    
    X,y_true = LLMlight.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = LLMlight.fit_transform(X, y_true, pos_label='bad', train_test=True, gridsearch=True)

    # The test error will be shown
    results = LLMlight.plot(model)


Learn new model on the entire data set
################################################

BBBB


.. code:: python

    # Import library
    import LLMlight

    # Load example data set    
    X,y_true = LLMlight.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = LLMlight.fit_transform(X, y_true, pos_label='bad', train_test=False, gridsearch=True)

    # The train error will be shown. Such results are heavily biased as the model also learned on this set of data
    results = LLMlight.plot(model)


