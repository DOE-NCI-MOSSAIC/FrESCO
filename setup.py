from setuptools import setup, find_packages

setup(
    name='nci-fresco',
    version='0.2.7',
    description='',
    url='https://github.com/DOE-NCI-MOSSAIC/FrESCO',
    author='Adam Spannaus',
    author_email='spannausat@ornl.gov',
    license='MIT',
    packages=find_packages(include=['fresco', 'fresco.*']),
    install_requires=['pandas',
                      'bardi==0.5.0',
                      'scikit-learn',
                      'pyyaml',
                      'gensim',
                      'numpy',
                      'tqdm',
                      'polars'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
