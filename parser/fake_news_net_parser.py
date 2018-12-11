import json
import numpy as np

from os import listdir
from os.path import join

DATA_DIR = '../FakeNewsNet/Data'


class FakeNewsNetParser(object):
    def __init__(self):
        pass

    def parse(self, data_directories=['BuzzFeed', 'PolitiFact']):
        real_news_files = []
        fake_news_files = []
        for directory in data_directories:
            real_news_dir = join(DATA_DIR, directory, 'RealNewsContent')
            fake_news_dir = join(DATA_DIR, directory, 'FakeNewsContent')

            real_news_files += [join(real_news_dir, f)
                                for f in listdir(real_news_dir)]
            fake_news_files += [join(fake_news_dir, f)
                                for f in listdir(fake_news_dir)]

        real_news = []
        fake_news = []
        for real_news_file in real_news_files:
            with open(real_news_file) as f:
                real_news.append(json.load(f))

        for fake_news_file in fake_news_files:
            with open(fake_news_file) as f:
                fake_news.append(json.load(f))

        X = []
        y = []

        self.x_fake = []
        self.x_real = []

        for news in fake_news:
            n = news['text'] + ' ' + news['title']
            self.x_fake.append(n)
            X.append(n)
            y.append(1)

        for news in real_news:
            n = news['text'] + ' ' + news['title']
            self.x_real.append(n)
            X.append(n)
            y.append(0)

        X = np.array(X)
        y = np.array(y)
        self.x_fake = np.array(self.x_fake)
        self.x_real = np.array(self.x_real)

        return X, y

    def get_fake_news(self):
        return self.x_fake

    def get_real_news(self):
        return self.x_real
