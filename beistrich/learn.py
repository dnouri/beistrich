"""Learn and fine-tune models.

Usage:
  beistrich-learn search <model_name> <config_file> [options]
  beistrich-learn curve <model_name> <config_file> [options]
  beistrich-learn curve_logloss <model_name> <config_file> [options]
  beistrich-learn report <model_name> <config_file> [options]
  beistrich-learn analyze <model_name> <config_file> [options]
"""

import matplotlib.pyplot as pl
from nolearn.console import Command
from nolearn.dataset import Dataset
from nolearn.grid_search import grid_search
from nolearn.metrics import learning_curve
from nolearn.metrics import learning_curve_logloss
from nolearn.model import AveragingEstimator
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from . import schema
from .model import LogisticRegressionModel
from .model import SGDModel


models = {
    'lr': LogisticRegressionModel(),

    'sgd': SGDModel(),

    'avg': lambda: AveragingEstimator((
        SGDModel()(),
        LogisticRegressionModel()(),
        ), n_jobs=-1, verbose=1),
    }


def search(infile_x='data/X-strat.npy', infile_y='data/y-strat.npy',
           verbose=4, n_jobs=1):
    model = models[main.arguments['<model_name>']]
    dataset = Dataset(infile_x, infile_y)
    return grid_search(
        dataset,
        model(),
        model.grid_search_params,
        verbose=verbose,
        cv=dataset.split_indices,
        score_func=f1_score,
        n_jobs=n_jobs,
        )


def curve(infile_x='data/X-strat.npy', infile_y='data/y-strat.npy',
          learning_curve=learning_curve):
    dataset = Dataset(infile_x, infile_y)
    clf = models[main.arguments['<model_name>']]()
    scores_train, scores_test, sizes = learning_curve(dataset, clf, verbose=1)
    pl.plot(sizes, scores_train, 'b', label='training set')
    pl.plot(sizes, scores_test, 'r', label='test set')
    pl.xlabel('n training cases')
    pl.ylabel('score')
    pl.title('Learning curve')
    pl.legend(loc='lower right')
    pl.show()


def curve_logloss(infile_x='data/X-strat.npy', infile_y='data/y-strat.npy',
                  learning_curve=learning_curve_logloss):
    return curve(infile_x, infile_y, learning_curve=learning_curve)


def report(infile_x='data/X-strat.npy', infile_y='data/y-strat.npy'):
    dataset = Dataset(infile_x, infile_y)
    clf = models[main.arguments['<model_name>']]()
    X_train, X_test, y_train, y_test = dataset.train_test_split()

    clf.fit(X_train, y_train)
    y_pred_probas = clf.predict_proba(X_test)
    y_pred = ((y_pred_probas[:, 1] - y_pred_probas[:, 0]) > 0).astype(int)
    y_pred_train = clf.predict(X_train)

    # Classification report
    print "Classification report for training set:"
    print
    print classification_report(y_train, y_pred_train)
    print "Classification report for test set:"
    print
    print classification_report(y_test, y_pred)

    # Compute confusion matrix
    print "Confusion matrix:"
    cm = confusion_matrix(y_test, y_pred)
    print cm
    print

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(
        y_test, y_pred_probas[:, 1])
    area = auc(recall, precision)
    print "Area Under Curve: %0.2f" % area

    # Plot
    pl.clf()
    pl.plot(recall, precision, label='precision-recall curve')
    pl.xlabel('recall')
    pl.ylabel('precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall; AUC=%0.2f' % area)
    pl.legend(loc='lower left')
    pl.show()


def _analyze(clf, infile_x='data/X-strat.npy', infile_y='data/y-strat.npy'):
    dataset = Dataset(infile_x, infile_y)
    X_train, X_test, y_train, y_test = dataset.train_test_split()

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1] - y_pred[:, 0]

    X_test_pos = X_test[y_test == 1]
    y_pred_pos = y_pred[y_test == 1]
    y_pred_pos_sorted = y_pred_pos.argsort()

    X_test_neg = X_test[y_test == 0]
    y_pred_neg = y_pred[y_test == 0]
    y_pred_neg_sorted = y_pred_neg.argsort()

    true_pos = (X_test_pos[y_pred_pos_sorted][::-1],
                y_pred_pos[y_pred_pos_sorted][::-1])
    true_neg = (X_test_neg[y_pred_neg_sorted],
                y_pred_neg[y_pred_neg_sorted])
    false_pos = (X_test_neg[y_pred_neg_sorted][::-1],
                 y_pred_neg[y_pred_neg_sorted][::-1])
    false_neg = (X_test_pos[y_pred_pos_sorted],
                 y_pred_pos[y_pred_pos_sorted])

    return true_pos, true_neg, false_pos, false_neg


def analyze(infile_x='data/X-strat.npy', infile_y='data/y-strat.npy',
            display=5):
    clf = models[main.arguments['<model_name>']]()
    true_pos, true_neg, false_pos, false_neg = _analyze(
        clf, infile_x, infile_y)

    def print_examples(examples, scores):
        for ex, score in zip(examples, scores):
            ex = ex.tolist() if hasattr(ex, 'tolist') else ex
            center = len(ex) / 2
            print u" - %s  [%.3f]" % (
                u' '.join(ex[:center] + [','] + ex[center:]),
                score,
                )

    print "True positives:"
    print "==============="
    print_examples(true_pos[0][:display], true_pos[1][:display])
    print

    print "True negatives:"
    print "==============="
    print_examples(true_neg[0][:display], true_neg[1][:display])
    print

    print "False positives:"
    print "================"
    print_examples(false_pos[0][:display], false_pos[1][:display])
    print

    print "False negatives:"
    print "================"
    print_examples(false_neg[0][:display], false_neg[1][:display])
    print


class Main(Command):
    __doc__ = __doc__
    schema = schema
    funcs = [
        search,
        curve,
        curve_logloss,
        report,
        analyze,
        ]

main = Main()
