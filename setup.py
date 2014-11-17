from distutils.core import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name = 'cudnn-python-wrappers',
    py_modules = ['libcudnn'],
    version = '0.1',
    license = 'MIT',
    description = 'Python wrappers for the NVIDIA cuDNN libraries.',
    long_description = read_md('README.md'),
    author = 'Hannes Bretschneider',
    author_email = 'habretschneider@gmail.com',
    url = 'https://github.com/hannes-brt/cudnn-python-wrappers',
    keywords = ['cuda', 'nvidia', 'cudnn', 'convolutional neural networks',
            'machine learning', 'deep learning'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
       ],
)