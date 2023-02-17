"""
MIT License

Copyright (c) 2018 Øyvind Sigmundson Schøyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


Based on the QuantumSystem code: https://github.com/HyQD/quantum-systems/
"""
import sys
from func_lib import *
from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)
import pyscf
def construct_pyscf_system_rhf_ref(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_overlap=None,
    reference_state=None,
    mix_states=False,
    return_C=False,
    weights=None,
    givenC=None,
    truncation=1000000,
    **kwargs,
):
    """Construct a spin orbital system with Procrustes orbitals"""
    import pyscf

    if np is None:
        import numpy as np

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = cart
    mol.build(atom=molecule, basis=basis, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    assert (
        n % 2 == 0
    ), "We require closed shell, with an even number of particles"

    l = mol.nao

    hf = pyscf.scf.RHF(mol)
    hf_energy = hf.kernel()

    if not hf.converged:
        warnings.warn("RHF calculation did not converge")

    if verbose:
        print(f"RHF energy: {hf.e_tot}")

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    if reference_state is None and givenC is None:
        C = np.asarray(hf.mo_coeff)
    elif givenC is None:
        C=localize_procrustes_ovlp(mol,hf.mo_coeff,hf.mo_occ,reference_state,mol.intor("int1e_ovlp"),reference_overlap)
    elif reference_state is None:
        C=givenC
    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)
    #system.change_basis(C)
    system.change_basis(C[:,:truncation])
    if return_C:
        return (
            system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
            if add_spin
            else system
        ), C
    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )
