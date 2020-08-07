## Welcome to quVario

This repository contains our attempt at coding a quantum variational method machine with Python3. The work is part of our Imperial College Physics Year 1 Summer Project. Due to the coronavirus, we were forced to choose a more computational project topic.

quVario is essentially a python ecosystem that can mathematically represent multi-electron (valance) atom system and find the system's ground state energy via the variational method. The integration techniques utilised range from deterministic quadrature methods to monte carlo methods. We can describe our iterative progress through two main prongs: Mark I (deterministic) and Mark II (monte carlo).

As of August 2020, the systems tested are the hydrogen and helium atoms.
### Helium_marki

```markdown

- First attempt at quantum method of variation problem
- Uses Sympy for symbolic differentiation
- Used out of the box SciPy integration numerical methods (numerical differentiation too, for hydrogen test case)

```

### Helium_markii

```markdown

- Monte Carlo progression of Mark I, with ecosystem further developed:
- psiham.py contains class objects that do the generation of the integrals (symbolic differentiation, algebra, etc.) in the form of a numba jit decorated function.
- optipack.py contains the actual integration functions: uniform monte carlo, metropolis algorithm, VEGAS algorithm (to be cited)
- psiham and optipack feeds helium_markii.py which does the actual execution.

```
