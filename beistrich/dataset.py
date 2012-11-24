"""Dataset related functions.

Usage:
  beistrich-dataset create <config_file> [options]
  beistrich-dataset stratify <config_file> [options]
  beistrich-dataset introspect <config_file> [options]
"""

from glob import glob
import logging
from urllib import FancyURLopener

import numpy as np
from nltk import wordpunct_tokenize
from nolearn.cache import cached
from nolearn.console import Command
from sklearn.utils import shuffle

from . import schema


class MyOpener(FancyURLopener):
    version = ('Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) '
               'Gecko/20071127 Firefox/2.0.0.11')

logger = logging.getLogger(__name__)
urlopen = MyOpener().open


@cached()
def download(url):  # pragma: no cover
    return urlopen(url).read()


def read(urls, start_marker, end_marker):
    texts = []
    for url in urls:
        logger.info("Downloading {}...".format(url))
        text = download(url)
        if start_marker:
            start_index = text.find(start_marker)
            end_index = text.find(end_marker)
            text = text[start_index:end_index]
        texts.append(text)
    return '\n\n'.join(texts)


def make_examples(words, size=10):
    index = size * 2
    while index < len(words) - size * 2:
        ex = []

        # Collect words to the right:
        label = False
        curr = index
        if words[index] == ',':
            label = True
            curr += 1

        while len(ex) < size:
            if words[curr] != ',':
                ex.append(words[curr])
            curr += 1

        curr = index - 1
        while len(ex) < size * 2:
            if words[curr] != ',':
                ex.insert(0, words[curr])
            curr -= 1

        yield ex, label
        index += 1
        if label:
            index += 1


def create(
    urls,
    start_marker='*** START OF THIS',
    end_marker='*** END OF THIS',
    outfile_x='data-X', outfile_y='data-y',
    ):
    texts = read(urls, start_marker, end_marker)
    words = wordpunct_tokenize(texts)
    examples = list(make_examples(words))
    X = np.array([e[0] for e in examples])
    y = np.array([1 if e[1] else 0 for e in examples])
    np.save(outfile_x, X)
    np.save(outfile_y, y)


def stratify(infile_x='data-X.npy', infile_y='data-y.npy',
             outfile_x='data-X-strat', outfile_y='data-y-strat',
             npos=1000, nneg=5000, random_state=42):
    X = np.load(infile_x)
    y = np.load(infile_y)
    pos_indices = shuffle(
        np.where(y == 1)[0], random_state=random_state)[:npos]
    neg_indices = shuffle(
        np.where(y == 0)[0], random_state=random_state)[:nneg]
    X2 = np.vstack([X[pos_indices, :], X[neg_indices, :]])
    y2 = np.hstack([y[pos_indices], y[neg_indices]])
    X2, y2 = shuffle(X2, y2, random_state=random_state)
    np.save(outfile_x, X2)
    np.save(outfile_y, y2)


def introspect(files):
    filenames = sum([glob(fn) for fn in files], [])
    for fn in sorted(filenames):
        a = np.load(fn)
        print "%s: %s  (bincount: %s)" % (
            fn.ljust(30),
            str(a.shape[0]).rjust(8),
            np.bincount(a).tolist(),
            )


class Main(Command):
    __doc__ = __doc__
    schema = schema
    funcs = [
        create,
        stratify,
        introspect,
        ]

main = Main()
