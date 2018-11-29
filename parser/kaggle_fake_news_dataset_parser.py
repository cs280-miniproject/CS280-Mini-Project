import numpy as np
import pandas as pd

from os.path import join

DATA_DIR = '../all'


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

        print(data.groupby(['label']).size())

        return X, y
