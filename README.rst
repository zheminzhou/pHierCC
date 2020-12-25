.. image:: https://img.shields.io/pypi/v/hiercc.svg
   :alt: HierCC on the Python Package Index (PyPI)
   :target: https://pypi.python.org/pypi/hiercc

Hosted by

.. image:: https://warwick.ac.uk/fac/sci/med/research/biomedical/mi/enterobase/enterobase.jpg?maxWidth=300
   :alt: The EnteroBase Website
   :target: https://enterobase.warwick.ac.uk


HierCC (Hierarchical clustering of cgMLST)
==========================================

HierCC is a multi-level clustering scheme for population assignments based on core genome Multi-Locus Sequence Types (cgMLSTs). HierCC as an independent python package works with any cgMLST schemes, and has also been implemented in `EnteroBase <https://enterobase.warwick.ac.uk>`_ since 2018. 

HierCC is open source software mad available under `GPL-3.0 License <https://github.com/zheminzhou/HierCC/blob/master/LICENSE>`_. 

If you use HierCC in work contributing to a scientific publication, we ask that you cite our preprint below: 

Zhou Z, Charlesworth J, Achtman M (2020) 
HierCC: A multi-level clustering scheme for population assignments based on core genome MLST. bioRxiv. 
DOI: https://doi.org/10.1101/2020.11.25.397539

If you use HierCC assignments that are hosted in EnteroBase, we ask that you cite our publication: 

Zhou Z, Alikhan NF, Mohamed K, the Agama Study Group, Achtman M (2020) 
The EnteroBase user's guide, with case studies on *Salmonella* transmissions, *Yersinia pestis* phylogeny and *Escherichia* core genomic diversity. Genome Res. 30:138-152.
DOI: https://dx.doi.org/10.1101%2Fgr.251678.119

Installation
============

Python 3.6 onwards, HierCC can be directly installed and upgraded via PIP, with just one terminal command::

   pip install HierCC
   pip install --upgrade HierCC

Alternatively, you may wish to download the GitHub repo and install the dependencies yourself as shown below. 

Python version
==============

HierCC is currently supported and tested on Python versions: 

- 3.7
- 3.8
- 3.9 (recommended)

Python libraries
================

HierCC requires: 

- `numpy <https://numpy.org/>`_ (>=1.18.1)
- `scipy <https://www.scipy.org/>`_ (>=1.3.2)
- `pandas <https://pandas.pydata.org/>`_ (>=0.24.2)
- `numba <https://numba.pydata.org/>`_ (>=0.38.0)
- `scikit-learn <https://scikit-learn.org/>`_ (>=0.23.1)
- `matplotlib <https://matplotlib.org/>`_ (>=3.2.1)
- `Click <https://click.palletsprojects.com/en/7.x/>`_ (>=7.0)
- `SharedArray <https://pypi.org/project/SharedArray/>`_ (>=3.2.1)

Testing
=======

