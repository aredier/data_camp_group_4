""" package that provides the backend for the hole project of the group number 4
the package is divided into 4 subpackages :
- data provides the tool to build, query and update the data base of the reviews
- preprocessing gives all the preprocessing functions used throughout the package
- supervised gives the supervised analysis functions
- unsupervised gives the unsupervised analysis functions

authors :
-  TO BE COMPLETED

"""

from .data import build_data_base, update
from .preprocessing import build_vocab