import numpy as np
import pandas as pd

from os.path import join

DATA_DIR = '../fake_real_news_dataset-master'


class FakeRealNewsDatasetParser(object):
    def __init__(self):
        pass

    def parse(self):
        data = pd.read_csv(join(DATA_DIR, 'fake_or_real_news.csv'))
        data.info()

        data = data.fillna(' ')
        data['total'] = data['title'] + ' ' + data['text']

        X = data['total'].values
        y = data['label'].replace({ 'REAL': 0, 'FAKE': 1 })

        print(data.groupby(['label']).size())

        return X, y
