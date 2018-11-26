import json
import numpy as np

from os import listdir
from os.path import join, isfile
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

DATA_DIR = '/Users/johnviscelsangkal/Projects/mini_project_cs280/FakeNewsNet/Data'
BUZZFEED_DATA_DIR = join(DATA_DIR, 'BuzzFeed')
POLITIFACT_DATA_DIR = join(DATA_DIR, 'PolitiFact')


def main():
    data_directories = [BUZZFEED_DATA_DIR, POLITIFACT_DATA_DIR]

    real_news_files = []
    fake_news_files = []
    for directory in data_directories:
        real_news_dir = join(directory, 'RealNewsContent')
        fake_news_dir = join(directory, 'FakeNewsContent')

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

    print('Real News Count: {}'.format(len(real_news)))
    print('Fake News Count: {}'.format(len(fake_news)))

    X = []
    y = []
    for news in fake_news:
        X.append(news['text'])
        y.append(1)

    for news in real_news:
        X.append(news['text'])
        y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    print(np.mean(predicted == y_test))

if __name__ == '__main__':
    main()
