from os.path import join, dirname
from setuptools import setup, find_packages

def read(fname):
    try:
        return open(join(dirname(__file__), fname)).read()
    except:
        return 'See https://github.com/qxmd/ImplicitMF/tree/master'


setup(
    name='implicitmf',
    version='0.1',
    author='QxMD team',
    author_email='cates.jill@gmail.com',
    description='Matrix factorization for implicit feedback datasets',
    long_description=read('README.md'),
    license='BSD-3',
    keywords='recommendation system',
    url='https://github.com/qxmd/ImplicitMF',
    # download_url='url here',
    packages=find_packages(exclude=("tests",)),
    install_requires=['numpy', 'scipy', 'pandas', 'implicit','lightfm'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)