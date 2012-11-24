# -*- coding: utf-8 -*-

from mock import patch
from mock import Mock
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@patch('beistrich.learn.main')
@patch('beistrich.learn.Dataset')
@patch('beistrich.learn.grid_search')
def test_search(grid_search, Dataset, main):
    from ..learn import search

    dataset = Mock()
    dataset.data = np.array([[1], [2]])
    dataset.target = np.array([1, 2])
    Dataset.return_value = dataset
    main.arguments = {'<model_name>': 'lr'}
    result = search()
    assert result is grid_search.return_value
    assert grid_search.call_args[0][0] is dataset
    assert isinstance(grid_search.call_args[0][1], Pipeline)
    assert isinstance(grid_search.call_args[0][2], dict)


@patch('beistrich.learn.main')
@patch('beistrich.learn.Dataset')
@patch('beistrich.learn.models')
@patch('beistrich.learn.pl')
def test_curve(pl, models, Dataset, main):
    from ..learn import curve
    from ..learn import curve_logloss

    models.__getitem__.return_value = lambda: LogisticRegression()
    dataset = Mock()
    dataset.train_test_split.return_value = (
        np.array([[0], [1]]), np.array([[0], [1]]),
        np.array([0, 1]), np.array([0, 1]),
        )
    Dataset.return_value = dataset
    main.arguments = {'<model_name>': 'lr'}

    learning_curve = Mock()
    learning_curve.return_value = ([0.9, 0.8], [0.5, 0.6], [5, 10])

    curve(learning_curve=learning_curve)
    pl.plot.assert_called_with([5, 10], [0.5, 0.6], 'r', label='test set')
    assert pl.plot.call_count == 2
    pl.reset_mock()

    curve_logloss(learning_curve=learning_curve)
    pl.plot.assert_called_with([5, 10], [0.5, 0.6], 'r', label='test set')
    assert pl.plot.call_count == 2
    pl.reset_mock()


@patch('beistrich.learn.main')
@patch('beistrich.learn.Dataset')
@patch('beistrich.learn.models')
@patch('beistrich.learn.pl')
def test_report(pl, models, Dataset, main):
    from ..learn import report

    models.__getitem__.return_value = lambda: LogisticRegression()
    dataset = Mock()
    dataset.train_test_split.return_value = (
        np.array([[0], [1]]), np.array([[0], [1]]),
        np.array([0, 1]), np.array([0, 1]),
        )
    Dataset.return_value = dataset
    main.arguments = {'<model_name>': 'lr'}
    report()


@patch('beistrich.learn.Dataset')
def test_analyze(Dataset):
    from ..learn import _analyze

    clf = Mock()

    X_test = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Dataset.return_value.train_test_split.return_value = (
        None, X_test, None, y_test)
    y_pred = np.array([
        [+1.0, -1.0], [-1.0, +1.0],  # fully correct
        [+0.5, +0.0], [-0.5, +0.0],  # quite unsure
        [-0.5, +0.5], [+0.5, -0.5],  # quite incorrect
        [-1.0, +1.0], [+1.0, -1.0],  # completely wrong
        ])
    clf.predict_proba.return_value = y_pred

    true_pos, true_neg, false_pos, false_neg = _analyze(clf)

    # Examples
    assert true_pos[0].tolist() == [[1], [3], [5], [7]]
    assert true_neg[0].tolist() == [[0], [2], [4], [6]]
    assert false_pos[0].tolist() == [[6], [4], [2], [0]]
    assert false_neg[0].tolist() == [[7], [5], [3], [1]]

    # Probabilities
    assert true_pos[1].tolist() == [2.0, 0.5, -1.0, -2.0]
    assert false_pos[1].tolist() == [2.0, 1.0, -0.5, -2.0]


@patch('beistrich.learn.main')
@patch('beistrich.learn._analyze')
def test_analyze_print(_analyze, main):
    from ..learn import analyze

    true_pos = (
        ([['pos', 'pos']], [1])
        )
    true_neg = (
        ([['neg', 'neg']], [-1])
        )
    false_pos = (
        ([['neg', 'neg']], [1])
        )
    false_neg = (
        ([['pos', 'pos']], [-1])
        )

    _analyze.return_value = (true_pos, true_neg, false_pos, false_neg)
    main.arguments = {'<model_name>': 'lr'}
    analyze()
