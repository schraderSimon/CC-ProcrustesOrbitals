# AMP-CCEVC

This repo contains all of the code required to reproduce the results from our paper [https://doi.org/10.1063/5.0141145](https://doi.org/10.1063/5.0141145) .

## Prerequisites
In addition to standard python libraries, [pyscf](https://pyscf.org/) is required, as are HyQD's [coupled cluster module](https://github.com/HyQD/coupled-cluster) and [Quantum Systems module](https://github.com/HyQD/quantum-systems). Quantum systems requires [BSE](https://github.com/MolSSI-BSE/basis_set_exchange).

## How to reproduce our data
The folder `coupled_cluster/cc-machinelearning` contains 12 files, 3 of which have a name that begins with `plot`. Running the remaining 9 files produces a one file each, that can be read with [pickle](https://docs.python.org/3/library/pickle.html).

The `plot*.py`-files reproduce all figures (except for figure 1) and the numerical results.

## How to access the results

Depending on the method considered, the pickle-readable files contains information about the energies using the different methods considered, as well as information about the sample cluster operator, the sample geometry, the parameters learned by the machine-learning algorithm etc.

In the files `HF*.py`, we have thoroughly commented what is written to file, such that the data should be easily accessible and verifiable. For other two molecules, we essentially produce the same data. The `plot*.py`-files contain examples how to access the data.

In addition, we made a file `extract_ML_information_HF.py` which exemplifies how to extract the cluster amplitudes, the orthogonalized cluster amplitudes, and the ML-coefficients $\sigma_f$ and l.
