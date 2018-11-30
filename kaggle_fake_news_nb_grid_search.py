import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from parser.kaggle_fake_news_dataset_parser import KaggleFakeNewsDatasetParser


def main():
    X, y = KaggleFakeNewsDatasetParser().parse()

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (0.1, 1e-2, 1e-3)
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5)
    gs_clf = gs_clf.fit(X, y)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)


if __name__ == '__main__':
    main()
