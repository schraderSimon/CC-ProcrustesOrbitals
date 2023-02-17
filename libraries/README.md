This folder contains the "backbone" of the article, e.g. the methods and classes to implement eigenvector continuation.

- `func_lib.py`, contain much-used convenience functions, such as the Procrustes algorithm, canonical orthonormalization etc.
- `rccsd_gs.py` contains the necessary algorithms & data to implement AMP-CCEVC using RHF determinants and spin-adapted CCSD
- `qs_ref.py` adapts orbitalSystems in Schøyens code, adapted in such a way that Procrustes orbitals are produced.
- `rhs_t.py`, `cc.py` `rccsd.py` contains the code for restricted CCSD amplitude calculations. It also contains code for the specific sums needed in parameter-reduced AMP-CCEVC.
- `machinelearning.py` contains the functions to learn Gaussian processes. It is very basic and makes no use of any machine learning libraries, but is sufficient for the purposes of the article, though when scaling it up to larger molecules and more molecules, improvements might be required there.
