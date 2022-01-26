''' Demo for CS 324 Winter 2022: Project 1
This file demonstrates using IMDB data how to:
1. How to construct queries
2. How to specify decoding parameters
3. How to submit queries/decoding params. to LLM API
4. How to generate a prediction from API response
5. How to evaluate the accuracy of predictions'''


import getpass
import csv

from src.common.authentication import Authentication
from src.common.request import Request, RequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService
from data import load_datasets


# Construct a query by specifying a prompt to accompany the input x
def get_query(x):
	prompt = 'Predict the sentiment of this review.'
	query = x + ' ' + prompt
	return query


# Specify decoding parameters.
# See src.common.request for full list of decoding parameters
def get_decoding_params():
	decoding_params = {'top_k_per_token' : 10, 'max_tokens' : 3}
	return decoding_params


# Specify request given query and decoding parameters
# See src.common.request for how to format request.
def get_request(query, decoding_params):
	return Request(prompt = query, **decoding_params)


# Map the API result to a class label (i.e. prediction)
def make_prediction(request_result):
	print('Look at the form of the API result; the demo method is very bad for using the result to generate a prediction')
	completion = request_result.completions[0].text.lower()
	if completion == 'positive':
		return 1
	elif completion == 'negative':
		return 0
	else:
		return 1


# Writes results to CSV file
def write_results_csv(predictions, run_name, header):
	file_name = 'results/{}.csv'.format(run_name)
	with open(file_name, 'w', encoding = 'UTF8') as f:
		writer = csv.writer(f)

		writer.writerow(header)

		for entry in predictions:
			row = [entry[column_name] for column_name in header]
			writer.writerow(row)


# An example of how to use the request API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("http://crfm-models.stanford.edu")

# Access account and show current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# As an example, we demonstrate how to use the codebase to work with the IMDB dataset
# For the first part of the assignment, you will use the ANLI dataset ('anli').
dataset_name = 'imdb'
train_dataset, dev_dataset, test_dataset = load_datasets(dataset_name)
predictions = []

for i, row in enumerate(test_dataset):
	x, y = row['text'], row['label']

	query: str = get_query(x)

	decoding_params = get_decoding_params()

	request: Request = get_request(query, decoding_params)
	print('API Request: {} \n'.format(request))
	
	request_result: RequestResult = service.make_request(auth, request)
	print('API Result: {} \n'.format(request_result))

	yhat = make_prediction(request_result)

	if 'uid' in row:
		uid = row['uid']
	else:
		uid = i

	predictions.append({'uid' : uid, 'y' : y, 'yhat' : yhat, 'correct' : y == yhat})

	# Terminates after first example to avoid burning API credits accidentally on demo
	break


metric_name = 'accuracy'
accuracy = sum([pred['correct'] for pred in predictions]) / len(predictions)
print('{} on {}: {}'.format(metric_name, dataset_name, accuracy))
	
run_name = 'demo'
header = ['uid', 'y', 'yhat']
write_results_csv(predictions, run_name, header)