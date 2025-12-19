import numpy as np
from HyresBuilder import RNAbuilder

RNAbuilder.build('AUCGCUAGUUUCCGGAA', 'test.pdb')

from HyresBuilder import GenPsf

GenPsf.genpsf()