from mock import patch
from mock import Mock
from nolearn.model import FeatureStacker
import numpy as np
import pytest
from sklearn.pipeline import Pipeline


def test_modelwithfeatures():
    from ..model import ModelWithFeatures

    model = ModelWithFeatures()
    with pytest.raises(NotImplementedError):
        model.pipeline

    assert isinstance(model.features, FeatureStacker)


def test_logisticregressionmodel():
    from ..model import LogisticRegressionModel

    model = LogisticRegressionModel()
    assert isinstance(model.pipeline, Pipeline)


def test_textextractor():
    from ..model import TextExtractor

    estimator = TextExtractor(1, 3)
    X = np.array([
        ['one', 'two', 'three', 'four'],
        ['five', 'six', 'seven', 'eight'],
        ])

    assert estimator.fit(X) is estimator
    result = estimator.transform(X)
    assert result == ['two three', 'six seven']


@patch('beistrich.model.stanford.POSTagger')
def test_posfeatures(POSTagger):
    from ..model import POSFeatures

    estimator = POSFeatures(1, 3)
    X = np.array([
        ['one', 'two', 'three', 'four'],
        ['five', 'six', 'seven', 'eight'],
        ])

    tagger = Mock()
    POSTagger.return_value = tagger
    tagger.batch_tag.return_value = [
        [('one', 'T1'), ('two', 'T2'), ('three', 'T3'), ('four', 'T4')],
        [('five', 'T5'), ('six', 'T6'), ('seven', 'T7'), ('eight', 'T8')],
        ]

    assert estimator.fit(X) is estimator
    assert estimator.transform(X) == [
        'T1 T2 T3 T4',
        'T5 T6 T7 T8',
        ]


def test_transform_cache_key():
    from ..model import transform_cache_key

    estimator = Mock()
    estimator.get_params.return_value = {'foo': 'fannick'}
    assert transform_cache_key(estimator, np.arange(100)) == (
        "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19],"
        "[80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99],"
        "(100,),"
        "[('foo', 'fannick')]"
        )
