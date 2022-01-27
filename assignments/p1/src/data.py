from datasets import load_dataset

# Please do not change; we want to ensure all groups evaluate on the same data.
seed = 324

# We truncate the test set to expedite evaluation.
test_size = 500


def shuffle_and_truncate(dataset, size = 500, seed = 324):
	dataset = dataset.shuffle(seed = seed)
	dataset = dataset.select(list(range(size)))
	return dataset


def load_anli():
	train_dataset = load_dataset('anli', split = 'train_r3')
	val_dataset = load_dataset('anli', split = 'dev_r3')
	val_dataset = shuffle_and_truncate(val_dataset, size = test_size, seed = seed)
	test_dataset = load_dataset('anli', split = 'test_r3')
	test_dataset = shuffle_and_truncate(test_dataset, size = test_size, seed = seed)
	return train_dataset, val_dataset, test_dataset


def load_crows_pairs():
	dataset = load_dataset('crows_pairs', split = 'test')
	return dataset


def load_stereoset():
	dataset = load_dataset('stereoset', 'intrasentence', split = 'validation')
	return dataset


def load_datasets(dataset_name):
	if dataset_name == 'imdb':
		return load_imdb()
	elif dataset_name == 'anli':
		return load_anli()
	elif dataset_name == 'crows_pairs':
		return load_crows_pairs()
	elif dataset_name == 'stereoset':
		return load_stereoset()
	else:
		raise NotImplementedError