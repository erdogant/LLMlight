
Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``LLMlight`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`LLMlight.save` and function :func:`LLMlight.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`LLMlight.save`:

.. code:: python

    import LLMlight

    # Load example data
    X,y_true = LLMlight.load_example()

    # Learn model
    model = LLMlight.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = LLMlight.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`LLMlight.load`:

.. code:: python

    import LLMlight

    # Load model
    model = LLMlight.load(model, 'learned_model_v1')

