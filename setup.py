from setuptools import setup, find_packages

setup(
    name='pretrain',
    version='0.1.0',
    url='https://github.com/unconst/pretrain',
    author='const',
    author_email='jake@opentensor.dev',
    description='Pretrain subnet for Bittensor',
    packages=find_packages(),    
    install_requires=[],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
    ],
)