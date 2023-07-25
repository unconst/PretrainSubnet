from setuptools import setup, find_packages
import importlib.util

# Function to read the requirements file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')
    # Translate GitHub URLs to PEP 508 form
    for i, requirement in enumerate(requirements):
        if requirement.startswith('git+https://'):
            pkg_name = requirement.split('/')[-1].split('.git')[0]
            requirements[i] = f'{pkg_name} @ {requirement}'
    return requirements

# Function to load the version
def load_version():
    spec = importlib.util.spec_from_file_location('pretrain', 'pretrain/__init__.py')
    pretrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrain)
    return pretrain.__version__

setup(
    name='pretrain',
    version=load_version(),
    url='https://github.com/unconst/pretrain',
    author='const',
    author_email='jake@opentensor.dev',
    description='Pretrain subnet for Bittensor',
    packages=find_packages(),    
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
    ],
)
