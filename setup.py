from setuptools import setup
import pip

setup(
    name='py_tensorflow_runner',
    version='0.0.1',
    description="Common interface for sharing tensorflow sessions across multiple processes",
    url='https://github.com/uniquetrij/py-tensorflow-runner',
    author='Trijeet Modak',
    author_email='uniquetrij@gmail.com',
    install_requires=[
       # 'tensorflow==1.12.0',
       # 'tensorflow-gpu==1.12.0',
        'py_pipe' if pip.__version__ < '19.0' else '',
    ],
    dependency_links=[
        'https://github.com/uniquetrij/py-pipe/tarball/release#egg=py-pipe-v0.0.1'
    ],
    packages=['py_tensorflow_runner'],
    zip_safe=False
)
