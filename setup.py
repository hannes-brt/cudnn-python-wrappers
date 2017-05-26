from distutils.core import setup

setup(
    name = 'cudnn-python-wrappers',
    py_modules = ['libcudnn'],
    version = '2.2b1',
    license = 'MIT',
    description = 'Python wrappers for the NVIDIA cudnn 8.0 libraries.',
    long_description = open('README.rst', 'r').read(),
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
