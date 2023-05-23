from setuptools import setup, find_packages

setup(
    name='fresco',
    version='0.1.0',    
    description='',
    url='https://github.com/DOE-NCI-MOSSAIC/FrESCO',
    author='Adam Spannaus',
    author_email='spannausat@ornl.gov',
    license='MIT',
    packages=find_packages(include=['fresco', 'fresco.*']),
    install_requires=[ 'pandas',                     
                      'scikit-learn',                     
                      'pyyaml',                     
                      'gensim',                     
                      'numpy',                     
                      'tqdm',       
                      'torch',
                      'yaml',
                      'gensim',
                      'torchmetrics>=0.11',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.9',
    ],
)
