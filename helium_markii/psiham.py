#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the hamLet object which handles calls for functions pertaining to hamiltonian generation (supposed to be general, but this case our focus is helium 5 term hamiltonian)

### hamiltonian should serve helium_markii.py which is the higher level script for direct user interface

### This script also provides the psiLet object which handles calls for functions related to creating a trial wavefunction object. It MUST be compatible with hamiltonian such that the objects can interact to give expectation term (quantum chem)

### psifunction should serve helium_markii.py which is the higher level script for direct user interface
