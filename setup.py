import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "flypy",
    version = "0.0.1",
    author = "Bob Scharmann",
    author_email = "bobby.scharmann+flypy@gmail.com",
    description = ("Simple package for using neural networks - mostly by my own learning"),
    license = "MIT",
    keywords = "neural networks, reinforcement learning",
    url = "http://packages.python.org/flypy",
    packages=['flypy'],
    long_description=read('README'),
    install_requires=['numpy', 'plotly'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.6',
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)
