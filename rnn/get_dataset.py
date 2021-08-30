from torchtext.legacy import data
import torch
import random

SEED = 42  # The answer to life, the universe, and everything



def get_dataset(include_lengths=False):
    """
    returns the
    TEXT, LABEL, train_set, val_set and test_set
    include_lengths=False
    """
    TEXT = data.Field(tokenize='spacy',  # tokenizer
                  tokenizer_language='en_core_web_sm',  # none
                  include_lengths=include_lengths
                  )

    LABEL = data.LabelField(dtype=torch.float)

    FIELDS = [('REVIEWS', TEXT), ('LABEL', LABEL)]

    dataset = data.TabularDataset(path='movie_data.csv',
                                  format='csv',
                                  fields=FIELDS,
                                  skip_header=True)
    print(len(dataset))

    train_set, test_set = dataset.split(split_ratio=[0.9, 0.1],
                                        random_state=random.seed(SEED))

    train_set, val_set = train_set.split(split_ratio=[0.9, 0.1],
                                         random_state=random.seed(SEED))

    return TEXT, LABEL, train_set, val_set, test_set
