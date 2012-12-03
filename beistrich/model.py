import sys

from nltk.tag import stanford
from nolearn.cache import cached
from nolearn.model import AbstractModel
from nolearn.model import FeatureStacker
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def _split_tokenizer(doc):
    return doc.split()


class ModelWithFeatures(AbstractModel):
    default_params = dict()

    def count(self, ngrams=1, max_features=10000):
        return CountVectorizer(
            ngram_range=((ngrams, ngrams)),
            max_features=max_features,
            lowercase=False,
            tokenizer=_split_tokenizer,
            )

    @property
    def left_ngram_1(self):
        return Pipeline([
            ('extract', TextExtractor(9, 10)),
            ('count', self.count(1)),
            ])

    @property
    def right_ngram_1(self):
        return Pipeline([
            ('extract', TextExtractor(10, 11)),
            ('count', self.count(1)),
            ('debugger', Debugger()),
            ])

    @property
    def left_ngram_2(self):
        return Pipeline([
            ('extract', TextExtractor(8, 10)),
            ('count', self.count(2)),
            ])

    @property
    def right_ngram_2(self):
        return Pipeline([
            ('extract', TextExtractor(10, 12)),
            ('count', self.count(2)),
            ('debugger', Debugger()),
            ])

    @property
    def left_pos(self):
        return Pipeline([
            ('pos', POSFeatures(8, 11)),
            ('count', self.count(3)),
            ('debugger', Debugger()),
            ])

    @property
    def right_pos(self):
        return Pipeline([
            ('pos', POSFeatures(9, 12)),
            ('count', self.count(3)),
            ])

    @property
    def pos4(self):
        return Pipeline([
            ('pos', POSFeatures(8, 12)),
            ('count', self.count(4)),
            ('debugger', Debugger()),
            ])

    @property
    def features(self):
        return FeatureStacker([
            ('left_ngram_1', self.left_ngram_1),
            ('right_ngram_1', self.right_ngram_1),
            ('left_ngram_2', self.left_ngram_2),
            ('right_ngram_2', self.right_ngram_2),
            ('left_pos', self.left_pos),
            ('right_pos', self.right_pos),
            ('pos4', self.pos4),
            ])


class LogisticRegressionModel(ModelWithFeatures):
    default_params = dict(
        clf__C=1.0,
        )

    grid_search_params = dict(
        clf__C=[1.0, 3.0],
        )

    @property
    def pipeline(self):
        return Pipeline([
            ('features', self.features),
            ('clf', LogisticRegression()),
            ])


class SGDModel(ModelWithFeatures):
    default_params = dict(
        clf__alpha=0.00001,
        )
    grid_search_params = dict(
        clf__alpha=[0.00003, 0.00001, 0.000003],
        )

    @property
    def pipeline(self):
        return Pipeline([
            ('features', self.features),
            ('clf', SGDClassifier(loss='log')),
            ])


class Debugger(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = sys._getframe(1)
        pipeline = frame.f_locals['self']

        n_nonzero = float(len(X.nonzero()[0]))
        print "\t{}".format(pipeline.steps[0][-1]),
        print "\tX.shape == {}    ".format(repr(X.shape)),
        print "\t{:.1f}% non-zero cases".format(n_nonzero / X.shape[0] * 100)
        return X


class TextExtractor(BaseEstimator):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extracted = X[:, self.start:self.end]
        return map(lambda r: ' '.join(r), extracted)


def transform_cache_key(self, X):
    return ','.join([
        str(X[:20]),
        str(X[-20:]),
        str(X.shape),
        str(sorted(self.get_params().items())),
        ])


class POSFeatures(BaseEstimator):
    def __init__(self, start, end, model='german-fast.tagger'):
        self.start = start
        self.end = end
        self.model = model

    def fit(self, X, y=None):
        self.tagger = stanford.POSTagger(self.model)
        return self

    def _split(self, doc):
        return ' '.join(doc).split()[self.start:self.end]

    @cached(transform_cache_key)
    def transform(self, X):
        docs = map(self._split, X)
        # I tried to give the POS tagger more words for context, but
        # surprisingly that has a (small) negative effect on the
        # overall performance, and the number of items in the
        # vocabulary are about the same.
        docs = self.tagger.batch_tag(docs)
        for i, doc in enumerate(docs):
            docs[i] = ' '.join([word[1] for word in doc])
        return docs
