import numpy as np
import pandas as pd

from os.path import join

DATA_DIR = '/Users/johnviscelsangkal/Projects/mini_project_cs280/all'


class KaggleFakeNewsDatasetParser(object):
    def __init__(self):
        pass

    def parse(self):
        data = pd.read_csv(join(DATA_DIR, 'train.csv'))
        data.info()

        data = data.fillna(' ')
        data['total'] = data['title'] + ' ' + data['text']

        X = data['total'].values
        y = data['label'].values

        return X, y
