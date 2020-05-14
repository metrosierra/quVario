#!/usr/bin/env python3
import math
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.odr import *
from scipy import optimize

from natsort import natsorted
from sympy.solvers import solve
from sympy import Symbol
import time
import progressbar

from functions_dict import motherFunc
