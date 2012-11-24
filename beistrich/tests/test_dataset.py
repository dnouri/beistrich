from nltk import wordpunct_tokenize
import numpy as np

EXAMPLE_TEXT = """
Split ends, known formally as trichoptilosis, happen when the
protective cuticle has been stripped away from the ends of hair
fibers.

This condition involves a longitudinal splitting of the hair
fiber. Any chemical or physical trauma, such as heat, that weathers
the hair may eventually lead to split ends. Typically, the damaged
hair fiber splits into two or three strands and the split may be two
to three centimeters in length. Split ends are most often observed in
long hair but also occur in short hair that is not in good condition.

As hair grows, the natural protective oils of the scalp can fail to
reach the ends of the hair. The ends are considered old once they
reach about 10 centimeters since they have had long exposure to the
sun, gone through many shampoos and may have been overheated by hair
dryers and hot irons. This all results in dry, brittle ends which are
prone to splitting. Infrequent trims and lack of hydrating treatments
can intensify this condition.
"""


def test_make_examples():
    from ..dataset import make_examples

    words = wordpunct_tokenize(EXAMPLE_TEXT)
    make_examples(words)
    result = list(make_examples(words))
    example = (
        "10 centimeters since they have had long exposure to the sun "
        "gone through many shampoos and may have been overheated by hair"
        )
    example = example.split()

    # There's a comma here:
    assert (example[1:-1], True) in result
    assert (example[1:-1], False) not in result

    # No commas here:
    assert (example[:-2], False) in result
    assert (example[2:], False) in result
    assert (example[:-2], True) not in result
    assert (example[2:], True) not in result

    assert len(result) == 141


def test_read():
    from ..dataset import read

    urls = ['http://www.gutenberg.org/cache/epub/12108/pg12108.txt',
            'http://www.gutenberg.org/cache/epub/34811/pg34811.txt']
    start_marker = '*** START OF THIS'
    end_marker = '*** END OF THIS'
    result = read(urls, start_marker, end_marker)
    assert len(result) == 1758455
    assert "Thomas Mann" in result


def test_create(tmpdir):
    from ..dataset import create

    urls = ['http://www.gutenberg.org/files/25791/25791-0.txt']
    outfile_x = tmpdir.join('data-X.npy')
    outfile_y = tmpdir.join('data-y.npy')
    create(urls, outfile_x=str(outfile_x), outfile_y=str(outfile_y))
    X = np.load(str(outfile_x))
    y = np.load(str(outfile_y))
    assert X.shape == (12426, 20)
    assert y.shape == (12426,)


def test_stratify(tmpdir):
    from ..dataset import stratify

    test_create(tmpdir)
    outfile_x = tmpdir.join('data-X-strat.npy')
    outfile_y = tmpdir.join('data-y-strat.npy')
    stratify(
        str(tmpdir.join('data-X.npy')),
        str(tmpdir.join('data-y.npy')),
        str(outfile_x),
        str(outfile_y),
        npos=500,
        nneg=3000,
        )
    X = np.load(str(outfile_x))
    y = np.load(str(outfile_y))
    assert X.shape == (3500, 20)
    assert y.shape == (3500,)
    assert len(y.nonzero()[0]) == 500


def test_introspect(tmpdir, capsys):
    from ..dataset import introspect

    test_create(tmpdir)
    introspect([str(tmpdir.join('data-y.npy'))])

    expected = u"data-y.npy:    12426  (bincount: [11477, 949])"
    assert expected in capsys.readouterr()[0]
