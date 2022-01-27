'''
Demo for CS 324 Winter 2022: Project 1.

This file demonstrates using ANLI data how to:

1. How to construct queries
2. How to specify decoding parameters
3. How to submit queries/decoding parameters to LLM API
4. How to generate a prediction from API response
5. How to evaluate the accuracy of predictions
'''


import getpass
import csv
import pprint

from authentication import Authentication
from request import Request, RequestResult
from accounts import Account
from remote_service import RemoteService
from data import load_datasets

def get_query(premise: str, hypothesis:str) -> str:
    """Construct a query by specifying a prompt to accompany the input x."""
    prompt = 'How are the Premise and Hypothesis related?'
    query = 'Premise: ' + premise + ' ' + 'Hypothesis: ' + hypothesis + ' ' + prompt
    return query


def get_decoding_params():
    """
    Specify decoding parameters.
    See `request.py` for full list of decoding parameters.
    """
    decoding_params = {'top_k_per_token' : 10, 'max_tokens' : 3}
    return decoding_params


def get_request(query, decoding_params) -> Request:
    """
    Specify request given query and decoding parameters.
    See `request.py` for how to format request.
    """
    return Request(prompt = query, **decoding_params)


def make_prediction(request_result: RequestResult) -> int:
    """
    Map the API result to a class label (i.e. prediction)
    """
    # TODO: this is a stub, please improve!
    completion = request_result.completions[0].text.lower()
    if completion == 'entailment':
        return 0
    elif completion == 'neutral':
        return 1
    elif completion == 'contradiction':
        return 2
    else:
        return 1


# Writes results to CSV file.
def write_results_csv(predictions, run_name, header):
    file_name = 'results/{}.csv'.format(run_name)
    with open(file_name, 'w', encoding = 'utf8') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        for entry in predictions:
            row = [entry[column_name] for column_name in header]
            writer.writerow(row)


# An example of how to use the request API.
auth = Authentication(api_key=open('api_key.txt').read().strip())
service = RemoteService()

# Access account and show current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# As an example, we demonstrate how to use the codebase to work with the ANLI dataset
dataset_name = 'anli'
train_dataset, dev_dataset, test_dataset = load_datasets(dataset_name)
predictions = []

max_examples = 1  # TODO: set this to -1 once you're ready to run it on the entire dataset

for i, row in enumerate(test_dataset):
    if i >= max_examples:
        break
    premise, hypothesis, y = row['premise'], row['hypothesis'], row['label']

    query: str = get_query(premise, hypothesis)

    decoding_params = get_decoding_params()

    request: Request = get_request(query, decoding_params)
    print('API Request: {}\n'.format(request))

    request_result: RequestResult = service.make_request(auth, request)
    print('API Result: {}\n'.format(request_result))

    yhat = make_prediction(request_result)

    if 'uid' in row:
        uid = row['uid']
    else:
        uid = i

    predictions.append({'uid' : uid, 'y' : y, 'yhat' : yhat, 'correct' : y == yhat})

metric_name = 'accuracy'
accuracy = sum([pred['correct'] for pred in predictions]) / len(predictions)
print('{} on {}: {}'.format(metric_name, dataset_name, accuracy))

run_name = 'demo'
header = ['uid', 'y', 'yhat']
write_results_csv(predictions, run_name, header)
