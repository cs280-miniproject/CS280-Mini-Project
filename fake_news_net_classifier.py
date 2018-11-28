import json
import numpy as np

from os import listdir
from os.path import join, isfile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
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

    X = np.array(X)
    y = np.array(y)

    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, y):
        print('--------------------------------------------------------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        text_clf = text_clf.fit(X_train, y_train)

        y_pred = text_clf.predict(X_test)
        print('ACCURACY: {}'.format(np.mean(y_pred == y_test)))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
