Welcome!  This codebase contains the code we provide for working on Project 1: Evaluating large language models.

# Setup

1. Create a virtual environment: `python3 -m venv venv` or `virtualenv venv`
2. Active virtual environment (on Linux/Mac: `source venv/bin/activate`)
3. Install packages: `pip install -r requirements.txt`

# Access to language models

In addition to directly accessing the API (see project writeup for more
details), you can use this codebase to access the LLMs programmatically.
To do so, you will need to create a file named `api_key.txt` with your API key.
Each student will receive their own API key; please be mindful of the token quota for the assignment. 

# Using the codebase

In `src/demo.py` we provide a simple example for how to use the codebase for the ANLI dataset.  
To run it, type `python src/demo.py`.

1. We load the ANLI dataset.
To load the other datasets, specify a `dataset_name` of _anli_ or _crows_pairs_ or _stereoset_.
2. For each example in a dataset, we retrieve the input.
3. We give an example for mapping from the input to the query that will be submitted to the API.
4. In addition, we give an example for specifying the decoding hyperparameters. 
See `src/request.py` for the full list of specifiable hyperparameters.
5. We submit the query and decoding hyperparameters, receiving a response from the API.
6. We give an example for mapping from the response to a prediction. Note that this will need to be adjusted based on the decoding parameters/format of the response you generate.
7. We compute the accuracy given the gold standard labels and predicted labels for all examples.

# Understanding the codebase

For the purposes of the assignment, you should be able to do every part only modifying/specifying the functions in `src/demo.py`. 
However, in case you are curious, here are the other parts of the codebase.
1. `src/remote_service.py` and `src/request.py` provide the code for querying the API. We do not currently anticipate any reason for modifying the functions here.
2. `src/data.py` is the data loader for the code involved in the assignment. You should not need to modify it.
