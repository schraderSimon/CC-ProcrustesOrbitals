# AMP-CCEVC

This repo contains all of the code required to reproduce the results from our paper _link follows_. 

## Prerequisites 
In addition to standard python libraries, [pyscf](https://pyscf.org/) is required, as are HyQD's [coupled cluster module](https://github.com/HyQD/coupled-cluster) and [Quantum Systems module](https://github.com/HyQD/quantum-systems). 

## How to reproduce our data
The folder _coupled_cluster/cc-machinelearning_ contains 12 files, 3 of which have a name that begins with _plot_. Running the remaining 9 files produces a one file each, that can be read with [pickle](https://docs.python.org/3/library/pickle.html). 

The _plot_-files reproduce all figures (except for figure 1) and the numerical results.

## How to access the results

Depending on the method considered, the pickle-readable files contains information about the energies using the different methods considered, as well as information about the sample cluster operator, the sample geometry, the parameters learned by the machine-learning algorithm etc.

In the files starting with _HF_, we have thoroughly commented what is written to file, such that the data should be easily accessible and verifiable. The other two molecules essentially produce the same data. The _plot_-files contain examples how to access the data.
