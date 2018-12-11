from parser.fake_news_net_parser import FakeNewsNetParser
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pprint


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def get_top_n_words(X, n):
    vec_pipe = TfidfVectorizer(stop_words='english')
    Xtr = vec_pipe.fit_transform(X)

    features = vec_pipe.get_feature_names()
    return top_mean_feats(Xtr, features, top_n=n)


def main():
    pp = pprint.PrettyPrinter(indent=4)

    BuzzFeedDataset = FakeNewsNetParser()
    BuzzFeedDataset.parse(['BuzzFeed'])

    print('==================================================================')
    print('BuzzFeed Fake News Top Words')
    print(get_top_n_words(BuzzFeedDataset.get_fake_news(), 50))
    print('==================================================================')
    print('BuzzFeed Real News Top Words')
    print(get_top_n_words(BuzzFeedDataset.get_real_news(), 50))

    PolitiFactDataset = FakeNewsNetParser()
    PolitiFactDataset.parse(['PolitiFact'])

    print('==================================================================')
    print('PolitiFact Fake News Top Words')
    print(get_top_n_words(PolitiFactDataset.get_fake_news(), 50))
    print('==================================================================')
    print('PolitiFact Real News Top Words')
    print(get_top_n_words(PolitiFactDataset.get_real_news(), 50))

if __name__ == '__main__':
    main()
