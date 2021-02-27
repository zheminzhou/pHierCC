import os, sys
from setuptools import setup, find_packages

__VERSION__ = '1.24'

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pHierCC',
    version= __VERSION__,
    #scripts=['PEPPA.py'] ,
    author="Zhemin Zhou",
    author_email="zhemin.zhou@warwick.ac.uk",
    description="Hierarchical Clustering of cgMLST profiles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zheminzhou/pHierCC",
    packages = ['pHierCC'],
    package_dir = {'pHierCC':'.'},
    keywords=['bioinformatics', 'microbial', 'genomics', 'MLST', 'hierarhical clustering'],
    install_requires=['SharedArray>=3.2.1', 'Click>=7.0', 'matplotlib>=3.2.1', 'scikit-learn>=0.23.1', 'numba>=0.38.0', 'numpy>=1.18.1', 'pandas>=0.24.2', 'scipy>=1.3.2'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pHierCC = pHierCC.pHierCC:phierCC',
            'HCCeval = pHierCC.HCCeval:evalHCC',
    ]},
    package_data={'HierCC': ['LICENSE', 'README.*']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
 )

