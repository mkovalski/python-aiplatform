Diagnostics tool
=======================================

The cloud diagnostics tool for Vertex AI allows users to use the provided
plugins to enable debugging and profiling options for training.

Usage
~~~~~~~

From your python training script:

.. code-block:: python

    from google.cloud.aiplatform import training_utils
    ...
    training_utils.start_diagnostics()

TF Profiler Plugin
^^^^^^^^^^^^^^^^^^^6

To use the TF Profiler, Vertex AI Tensorboard must be enabled in your training
job. For this, see
https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training.

From there, one can use the `capture_profile` option within Tensorboard to
capture profiling traces from the training job.
