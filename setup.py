import os

from setuptools import setup, find_packages

version = '0.1dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''

install_requires = [
    'docopt',
    'mock',
    'nltk>=2.0.1',
    'nolearn>=0.2b1',
    'pytest',
    'pytest-cov',
    ]


setup(name='beistrich',
      version=version,
      description="Predict where to put commas in sentences.",
      long_description=README,
      classifiers=[
          'Development Status :: 3 - Alpha',
        ],
      keywords='',
      author='Daniel Nouri',
      author_email='daniel.nouri@gmail.com',
      url='https://github.com/dnouri/beistrich',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points="""
      [console_scripts]
      beistrich-dataset = beistrich.dataset:main
      beistrich-learn = beistrich.learn:main
      """,
      dependency_links=[
          ],
      )
