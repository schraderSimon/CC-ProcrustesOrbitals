"""
Copyright (c) 2014-2018, The Psi4NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the Psi4NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modified from the original source code:
    https://github.com/psi4/psi4numpy/blob/cbef6ddcb32ccfbf773befea6dc4aaae2b428776/Coupled-Cluster/RHF/helper_ccenergy.py

Further modified from
    https://github.com/HyQD/coupled-cluster/blob/eb843dcae69106592040a88a4fbe530520c651f6/coupled_cluster/rccsd/rhs_t.py

"""

from opt_einsum import contract
#from numpy import einsum as contract
import sys

def compute_t_1_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates

    Fae = build_Fae(f, u, t1, t2, o, v, np)
    Fmi = build_Fmi(f, u, t1, t2, o, v, np)
    Fme = build_Fme(f, u, t1, o, v, np)

    #### Build residual of T1 equations by spin adaption of  Eqn 1:
    r_T1 = np.zeros((nvirt, nocc), dtype=t1.dtype)
    r_T1 += f[v, o]
    r_T1 += contract("ei,ae->ai", t1, Fae)
    r_T1 -= contract("am,mi->ai", t1, Fmi)
    r_T1 += 2 * contract("aeim,me->ai", t2, Fme)
    r_T1 -= contract("eaim,me->ai", t2, Fme)
    r_T1 += 2 * contract("fn,nafi->ai", t1, u[o, v, v, o])
    r_T1 -= contract("fn,naif->ai", t1, u[o, v, o, v])
    r_T1 += 2 * contract("efmi,maef->ai", t2, u[o, v, v, v])
    r_T1 -= contract("femi,maef->ai", t2, u[o, v, v, v])
    r_T1 -= 2 * contract("aemn,nmei->ai", t2, u[o, o, v, o])
    r_T1 += contract("aemn,nmie->ai", t2, u[o, o, o, v])
    return r_T1


def compute_t_2_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates
    # TODO: This should be handled more smoothly in the sense that
    # they are compute in compute_t1_amplitudes as well

    Fae = build_Fae(f, u, t1, t2, o, v, np)
    Fmi = build_Fmi(f, u, t1, t2, o, v, np)
    Fme = build_Fme(f, u, t1, o, v, np)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    r_T2 += u[v, v, o, o]

    tmp = contract("aeij,be->abij", t2, Fae)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
    tmp = contract("bm,me->be", t1, Fme)
    first = 0.5 * contract("aeij,be->abij", t2, tmp)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
    tmp = contract("abim,mj->abij", t2, Fmi)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
    tmp = contract("ej,me->jm", t1, Fme)
    first = 0.5 * contract("abim,jm->abij", t2, tmp)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # Build TEI Intermediates
    tmp_tau = build_tau(t1, t2, o, v, np)

    Wmnij = build_Wmnij(u, t1, t2, o, v, np)
    Wmbej = build_Wmbej(u, t1, t2, o, v, np)
    Wmbje = build_Wmbje(u, t1, t2, o, v, np)
    Zmbij = build_Zmbij(u, t1, t2, o, v, np)

    # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
    # This also includes the last term in 0.5 * tau_ijef Wabef
    # as Wmnij is modified to include this contribution.
    r_T2 += contract("abmn,mnij->abij", tmp_tau, Wmnij)

    # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
    # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
    # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
    # for in the contraction just above.

    # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
    r_T2 += contract("efij,abef->abij", tmp_tau, u[v, v, v, v])

    # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
    # where Zmbij_mbij = <mb|ef> * tau_ijef
    tmp = contract("am,mbij->abij", t1, Zmbij)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
    # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp -= contract("eaim,mbej->abij", t2, Wmbej)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp += contract("aeim,mbje->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
    tmp = contract("aemj,mbie->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
    #                                                      + t_ie * t_mb * <ma|je>}
    tmp = contract("ei,am->imea", t1, t1)
    tmp1 = contract("imea,mbej->abij", tmp, u[o, v, v, o])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("ei,bm->imeb", t1, t1)
    tmp1 = contract("imeb,maje->abij", tmp, u[o, v, o, v])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
    tmp = contract("ei,abej->abij", t1, u[v, v, v, o])
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
    tmp = contract("am,mbij->abij", t1, u[o, v, o, o])
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r_T2


def compute_t_2_amplitudes_REDUCED(f, u, t1, t2, o, v, np, picks,nos,nvs, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    nof=np.arange(nocc)
    nvf=np.arange(nvirt)
    ### Build OEI intermediates

    Fae = build_Fae(f, u, t1, t2, o, v, np)
    Fmi = build_Fmi(f, u, t1, t2, o, v, np)
    Fme = build_Fme(f, u, t1, o, v, np)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype) #abij format
    r_T2_of_interest=np.zeros((len(nvs), len(nvs), len(nos), len(nos)), dtype=t1.dtype)
    not_nvs=[i for i in nvf if i not in nvs]
    not_nos=[i for i in nof if i not in nos]
    ucopy=u.copy()
    ucopy[v, v, o, o][np.ix_(not_nvs,not_nvs,not_nos,not_nos)]=0
    t2[np.ix_(not_nvs,not_nvs,not_nos,not_nos)]=0
    #np.putmask(ucopy,picks,0)
    r_T2 += ucopy[v, v, o, o]

    tmp = contract("aeij,be->abij", t2, Fae)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    np.putmask(tmp,picks,0)
    np.putmask(tmpcopy,picks,0)
    r_T2 += tmp
    r_T2 += tmpcopy

    # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
    tmp = contract("bm,me->be", t1, Fme)
    first = 0.5 * contract("aeij,be->abij", t2, tmp)
    firstcopy=first.swapaxes(0, 1).swapaxes(2, 3).copy()
    np.putmask(first,picks,0)
    np.putmask(firstcopy,picks,0)
    r_T2 -= first
    r_T2 -= firstcopy


    # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
    tmp = contract("abim,mj->abij", t2, Fmi)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    np.putmask(tmp,picks,0)
    np.putmask(tmpcopy,picks,0)
    r_T2 -= tmp
    r_T2 -= tmpcopy

    # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
    tmp = contract("ej,me->jm", t1, Fme)
    first = 0.5 * contract("abim,jm->abij", t2, tmp)
    firstcopy=first.swapaxes(0, 1).swapaxes(2, 3).copy()
    np.putmask(first,picks,0)
    np.putmask(firstcopy,picks,0)
    r_T2 -= first
    r_T2 -= firstcopy

    # Build TEI Intermediates

    tmp_tau = build_tau(t1, t2, o, v, np)
    tmp_tau_old=tmp_tau.copy()
    tmp_tau[np.ix_(not_nvs,not_nvs,not_nos,not_nos)]=0
    #print(tmp_tau)
    #print(tmp_tau.shape)
    #print(not_nvs)
    #print(not_nos)
    Wmnij = build_Wmnij(u, t1, t2, o, v, np)
    Wmnij[np.ix_([],[],not_nos,not_nos)]=0
    Wmbej = build_Wmbej(u, t1, t2, o, v, np)
    Wmbej[np.ix_([],not_nvs,[],not_nos)]=0
    Wmbje = build_Wmbje(u, t1, t2, o, v, np)
    Wmbje[np.ix_([],not_nvs,not_nos,[])]=0
    Zmbij= build_Zmbij(u, t1, t2, o, v, np)
    Zmbij[np.ix_([],not_nvs,not_nos,not_nos)]=0
    # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
    # This also includes the last term in 0.5 * tau_ijef Wabef
    # as Wmnij is modified to include this contribution.
    tmp=contract("abmn,mnij->abij", tmp_tau, Wmnij)
    r_T2 +=tmp

    # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
    # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
    # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
    # for in the contraction just above.

    # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
    tmp=contract("efij,abef->abij", tmp_tau, u[v, v, v, v])
    r_T2 += tmp

    # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
    # where Zmbij_mbij = <mb|ef> * tau_ijef
    tmp = contract("am,mbij->abij", t1, Zmbij)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 -= tmp
    r_T2 -= tmpcopy

    # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
    # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp -= contract("eaim,mbej->abij", t2, Wmbej)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 += tmp
    r_T2 += tmpcopy

    # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp += contract("aeim,mbje->abij", t2, Wmbje)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 += tmp
    r_T2 += tmpcopy

    # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
    tmp = contract("aemj,mbie->abij", t2, Wmbje)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 += tmp
    r_T2 += tmpcopy

    # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
    #                                                      + t_ie * t_mb * <ma|je>}
    tmp = contract("ei,am->imea", t1, t1)
    tmp1 = contract("imea,mbej->abij", tmp, u[o, v, v, o])
    tmpcopy=tmp1.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 -= tmp1
    r_T2 -= tmpcopy





    tmp = contract("ei,bm->imeb", t1, t1)
    tmp1 = contract("imeb,maje->abij", tmp, u[o, v, o, v])
    tmpcopy=tmp1.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 -= tmp1
    r_T2 -= tmpcopy

    # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
    tmp = contract("ei,abej->abij", t1, u[v, v, v, o])
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 += tmp
    r_T2 += tmpcopy

    # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
    tmp = contract("am,mbij->abij", t1, u[o, v, o, o])
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2 -= tmp
    r_T2 -= tmpcopy
    np.putmask(r_T2,picks,0)
    return r_T2

def compute_t_1_amplitudes_REDUCED_new(f, u, t1, t2, o, v, np, picks,nos,nvs, out=None):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates

    Fae = build_reduced_Fae(f, u, t1, t2, o, v, np,nos,nvs)
    Fmi = build_reduced_Fmi(f, u, t1, t2, o, v, np,nos,nvs)
    Fme = build_reduced_Fme(f, u, t1, o, v, np,nos,nvs)

    #### Build residual of T1 equations by spin adaption of  Eqn 1:
    r_T1 = np.zeros((nvirt, nocc), dtype=t1.dtype)
    r_T1_of_interest=np.zeros((len(nvs),len(nos)), dtype=t1.dtype)
    r_T1_of_interest += f[v, o][np.ix_(nvs,nos)]
    r_T1_of_interest += contract("ei,ae->ai", t1[np.ix_(all_v,nos)], Fae)
    r_T1_of_interest -= contract("am,mi->ai", t1[np.ix_(nvs,all_o)], Fmi)
    r_T1_of_interest += 2 * contract("aeim,me->ai", t2[np.ix_(nvs,all_v,nos,all_o)], Fme)
    r_T1_of_interest -= contract("eaim,me->ai", t2[np.ix_(all_v,nvs,nos,all_o)], Fme)
    r_T1_of_interest += 2 * contract("fn,nafi->ai", t1[np.ix_(all_v,all_o)], u[o, v, v, o][np.ix_(all_o,nvs,all_v,nos)])
    r_T1_of_interest -= contract("fn,naif->ai", t1[np.ix_(all_v,all_o)], u[o, v, o, v][np.ix_(all_o,nvs,nos,all_v)])
    r_T1_of_interest += 2 * contract("efmi,maef->ai", t2[np.ix_(all_v,all_v,all_o,nos)], u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    r_T1_of_interest -= contract("femi,maef->ai", t2[np.ix_(all_v,all_v,all_o,nos)], u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    r_T1_of_interest -= 2 * contract("aemn,nmei->ai", t2[np.ix_(nvs,all_v,all_o,all_o)], u[o, o, v, o][np.ix_(all_o,all_o,all_v,nos)])
    r_T1_of_interest += contract("aemn,nmie->ai", t2[np.ix_(nvs,all_v,all_o,all_o)], u[o, o, o, v][np.ix_(all_o,all_o,nos,all_v)])
    r_T1[np.ix_(nvs,nos)]=r_T1_of_interest
    return r_T1
def compute_t_2_amplitudes_REDUCED_new(f, u, t1, t2, o, v, np, picks,nos,nvs, out=None):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    nof=np.arange(nocc)
    nvf=np.arange(nvirt)
    ### Build OEI intermediates

    Fae = build_reduced_Fae(f, u, t1, t2, o, v, np,nos,nvs)#[np.ix_(nvs,all_v)]
    Fmi = build_reduced_Fmi(f, u, t1, t2, o, v, np,nos,nvs)#[np.ix_(all_o,nos)]
    Fme = build_reduced_Fme(f, u, t1, o, v, np,nos,nvs)#[np.ix_(all_o,all_v)]

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype) #abij format
    r_T2_of_interest=np.zeros((len(nvs), len(nvs), len(nos), len(nos)), dtype=t1.dtype)

    not_nvs=[i for i in nvf if i not in nvs]
    not_nos=[i for i in nof if i not in nos]
    r_T2_of_interest+=u[v, v, o, o][np.ix_(nvs,nvs,nos,nos)]

    tmp = contract("aeij,be->abij", t2[np.ix_(nvs,all_v,nos,nos)], Fae)
    r_T2_of_interest += tmp
    r_T2_of_interest += tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("bm,me->be", t1[np.ix_(nvs,all_o)], Fme)
    first = 0.5 * contract("aeij,be->abij", t2[np.ix_(nvs,all_v,nos,nos)], tmp)
    r_T2_of_interest -= first
    r_T2_of_interest -= first.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("abim,mj->abij", t2[np.ix_(nvs,nvs,nos,all_o)], Fmi)
    r_T2_of_interest -= tmp
    r_T2_of_interest -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("ej,me->jm", t1[np.ix_(all_v,nos)], Fme)
    first = 0.5 * contract("abim,jm->abij", t2[np.ix_(nvs,nvs,nos,all_o)], tmp)
    r_T2_of_interest -= first
    r_T2_of_interest -= first.swapaxes(0, 1).swapaxes(2, 3)

    tmp_tau = build_tau(t1, t2, o, v, np)


    Wmnij = build_reduced_Wmnij(u, t1, t2, o, v, np,nos,nvs)
    Wmbej = build_reduced_Wmbej(u, t1, t2, o, v, np,nos,nvs)
    Wmbje = build_reduced_Wmbje(u, t1, t2, o, v, np,nos,nvs)
    Zmbij = build_reduced_Zmbij(u, t1, t2, o, v, np,nos,nvs)
    tmp=contract("abmn,mnij->abij", tmp_tau[np.ix_(nvs,nvs,all_o,all_o)], Wmnij)
    r_T2_of_interest+=tmp
    tmp=contract("efij,abef->abij", tmp_tau[np.ix_(all_v,all_v,nos,nos)], u[v, v, v, v][np.ix_(nvs,nvs,all_v,all_v)])
    r_T2_of_interest += tmp
    tmp = contract("am,mbij->abij", t1[np.ix_(nvs,all_o)], Zmbij)
    r_T2_of_interest -= tmp
    r_T2_of_interest -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("aeim,mbej->abij", t2[np.ix_(nvs,all_v,nos,all_o)], Wmbej)
    tmp -= contract("eaim,mbej->abij", t2[np.ix_(all_v,nvs,nos,all_o)], Wmbej)
    r_T2_of_interest += tmp
    r_T2_of_interest += tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("aeim,mbej->abij", t2[np.ix_(nvs,all_v,nos,all_o)], Wmbej)
    tmp += contract("aeim,mbje->abij", t2[np.ix_(nvs,all_v,nos,all_o)], Wmbje)
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2_of_interest += tmp
    r_T2_of_interest += tmpcopy
    tmp = contract("aemj,mbie->abij", t2[np.ix_(nvs,all_v,all_o,nos)], Wmbje)
    r_T2_of_interest += tmp
    r_T2_of_interest += tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("ei,am->imea", t1[np.ix_(all_v,nos)], t1[np.ix_(nvs,all_o)])
    tmp1 = contract("imea,mbej->abij", tmp, u[o, v, v, o][np.ix_(all_o,nvs,all_v,nos)])
    r_T2_of_interest -= tmp1
    r_T2_of_interest -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

    tmp = contract("ei,bm->imeb", t1[np.ix_(all_v,nos)], t1[np.ix_(nvs,all_o)])
    tmp1 = contract("imeb,maje->abij", tmp, u[o, v, o, v][np.ix_(all_o,nvs,nos,all_v)])
    r_T2_of_interest -= tmp1
    r_T2_of_interest -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("ei,abej->abij", t1[np.ix_(all_v,nos)], u[v, v, v, o][np.ix_(nvs,nvs,all_v,nos)])
    tmpcopy=tmp.swapaxes(0, 1).swapaxes(2, 3).copy()
    r_T2_of_interest += tmp
    r_T2_of_interest += tmpcopy
    tmp = contract("am,mbij->abij", t1[np.ix_(nvs,all_o)], u[o, v, o, o][np.ix_(all_o,nvs,nos,nos)])
    r_T2_of_interest -= tmp
    r_T2_of_interest -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    r_T2[np.ix_(nvs,nvs,nos,nos)]=r_T2_of_interest
    return r_T2

def build_tilde_tau(t1, t2, o, v, np):
    ttau = t2.copy()
    tmp = 0.5 * contract("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


def build_tau(t1, t2, o, v, np):
    ttau = t2.copy()
    tmp = contract("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


def build_Fae(f, u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fae = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    #print(nvirt)
    #print(v)
    #print(f.shape)
    #sys.exit(1)
    Fae += f[v, v]
    Fae -= 0.5 * contract("me,am->ae", f[o, v], t1)
    Fae += 2 * contract("fm,mafe->ae", t1, u[o, v, v, v])
    Fae -= contract("fm,maef->ae", t1, u[o, v, v, v])
    Fae -= 2 * contract(
        "afmn,mnef->ae", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    Fae += contract(
        "afmn,mnfe->ae", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Fae


def build_Fmi(f, u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fmi = np.zeros((nocc, nocc), dtype=t1.dtype)

    Fmi += f[o, o]
    Fmi += 0.5 * contract("ei,me->mi", t1, f[o, v])
    Fmi += 2 * contract("en,mnie->mi", t1, u[o, o, o, v])
    Fmi -= contract("en,mnei->mi", t1, u[o, o, v, o])
    Fmi += 2 * contract(
        "efin,mnef->mi", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    Fmi -= contract(
        "efin,mnfe->mi", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Fmi


def build_Fme(f, u, t1, o, v, np):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fme = np.zeros((nocc, nvirt), dtype=t1.dtype)
    Fme += f[o, v]
    Fme += 2 * contract("fn,mnef->me", t1, u[o, o, v, v])
    Fme -= contract("fn,mnfe->me", t1, u[o, o, v, v])
    return Fme


def build_Wmnij(u, t1, t2, o, v, np):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmnij = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)

    Wmnij += u[o, o, o, o]
    Wmnij += contract("ej,mnie->mnij", t1, u[o, o, o, v])
    Wmnij += contract("ei,mnej->mnij", t1, u[o, o, v, o])
    # prefactor of 1 instead of 0.5 below to fold the last term of
    # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
    Wmnij += contract(
        "efij,mnef->mnij", build_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Wmnij



def build_Wmbej(u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbej = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t1.dtype)

    Wmbej += u[o, v, v, o]
    Wmbej += contract("fj,mbef->mbej", t1, u[o, v, v, v])
    Wmbej -= contract("bn,mnej->mbej", t1, u[o, o, v, o])
    tmp = 0.5 * t2
    tmp += contract("fj,bn->fbjn", t1, t1)
    Wmbej -= contract("fbjn,mnef->mbej", tmp, u[o, o, v, v])
    Wmbej += contract("fbnj,mnef->mbej", t2, u[o, o, v, v])
    Wmbej -= 0.5 * contract("fbnj,mnfe->mbej", t2, u[o, o, v, v])
    return Wmbej


def build_Wmbje(u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbje = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t1.dtype)

    Wmbje += -1.0 * u[o, v, o, v]
    Wmbje -= contract("fj,mbfe->mbje", t1, u[o, v, v, v])
    Wmbje += contract("bn,mnje->mbje", t1, u[o, o, o, v])
    tmp = 0.5 * t2
    tmp += contract("fj,bn->fbjn", t1, t1)
    Wmbje += contract("fbjn,mnfe->mbje", tmp, u[o, o, v, v])
    return Wmbje


def build_Zmbij(u, t1, t2, o, v, np):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Zmbij = np.zeros((nocc, nvirt, nocc, nocc), dtype=t1.dtype)

    Zmbij += contract(
        "mbef,efij->mbij", u[o, v, v, v], build_tau(t1, t2, o, v, np)
    )
    return Zmbij



def build_reduced_Wmnij(u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmnij = np.zeros((len(all_o), len(all_o), len(nos), len(nos)), dtype=t1.dtype)

    Wmnij += u[o, o, o, o][np.ix_(all_o,all_o,nos,nos)]
    Wmnij += contract("ej,mnie->mnij", t1[np.ix_(all_v,nos)], u[o, o, o, v][np.ix_(all_o,all_o,nos,all_v)])
    Wmnij += contract("ei,mnej->mnij", t1[np.ix_(all_v,nos)], u[o, o, v, o][np.ix_(all_o,all_o,all_v,nos)])
    # prefactor of 1 instead of 0.5 below to fold the last term of
    # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
    Wmnij += contract(
        "efij,mnef->mnij", build_tau(t1, t2, o, v, np)[np.ix_(all_v,all_v,nos,nos)], u[o, o, v, v][np.ix_(all_o,all_o,all_v,all_v)]
    )
    return Wmnij


def build_reduced_Wmbej(u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbej =np.zeros((len(all_o), len(nvs), len(all_v), len(nos)), dtype=t1.dtype)

    Wmbej += u[o, v, v, o][np.ix_(all_o,nvs,all_v,nos)]
    Wmbej += contract("fj,mbef->mbej", t1[np.ix_(all_v,nos)], u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    Wmbej -= contract("bn,mnej->mbej", t1[np.ix_(nvs,all_o)], u[o, o, v, o][np.ix_(all_o,all_o,all_v,nos)])
    tmp = 0.5 * t2[np.ix_(all_v,nvs,nos,all_o)]
    tmp += contract("fj,bn->fbjn", t1[np.ix_(all_v,nos)], t1[np.ix_(nvs,all_o)])
    Wmbej -= contract("fbjn,mnef->mbej", tmp, u[o, o, v, v][np.ix_(all_o,all_o,all_v,all_v)])
    Wmbej += contract("fbnj,mnef->mbej", t2[np.ix_(all_v,nvs,all_o,nos)], u[o, o, v, v][np.ix_(all_o,all_o,all_v,all_v)])
    Wmbej -= 0.5 * contract("fbnj,mnfe->mbej", t2[np.ix_(all_v,nvs,all_o,nos)], u[o, o, v, v][np.ix_(all_o,all_o,all_v,all_v)])
    return Wmbej


def build_reduced_Wmbje(u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbje = np.zeros((len(all_o), len(nvs), len(nos), len(all_v)), dtype=t1.dtype)

    Wmbje += -1.0 * u[o, v, o, v][np.ix_(all_o,nvs,nos,all_v)]
    Wmbje -= contract("fj,mbfe->mbje", t1[np.ix_(all_v,nos)], u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    Wmbje += contract("bn,mnje->mbje", t1[np.ix_(nvs,all_o)], u[o, o, o, v][np.ix_(all_o,all_o,nos,all_v)])
    tmp = 0.5 * t2[np.ix_(all_v,nvs,nos,all_o)]
    tmp += contract("fj,bn->fbjn", t1[np.ix_(all_v,nos)], t1[np.ix_(nvs,all_o)])
    Wmbje += contract("fbjn,mnfe->mbje", tmp, u[o, o, v, v][np.ix_(all_o,all_o,all_v,all_v)])
    return Wmbje


def build_reduced_Zmbij(u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Zmbij = np.zeros((len(all_o), len(nvs), len(nos), len(nos)), dtype=t1.dtype)

    Zmbij += contract("mbef,efij->mbij", u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)], build_tau(t1, t2, o, v, np)[np.ix_(all_v,all_v,nos,nos)])
    return Zmbij


def build_reduced_Fae(f, u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fae = np.zeros((len(nvs), len(all_v)), dtype=t1.dtype)
    Fae += f[v, v][np.ix_(nvs,all_v)]
    Fae -= 0.5 * contract("me,am->ae", f[o, v][np.ix_(all_o,all_v)], t1[np.ix_(nvs,all_o)])
    Fae += 2 * contract("fm,mafe->ae", t1, u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    Fae -= contract("fm,maef->ae", t1, u[o, v, v, v][np.ix_(all_o,nvs,all_v,all_v)])
    Fae -= 2 * contract(
        "afmn,mnef->ae", build_tilde_tau(t1, t2, o, v, np)[np.ix_(nvs,all_v,all_o,all_o)], u[o, o, v, v]
    )
    Fae += contract(
        "afmn,mnfe->ae", build_tilde_tau(t1, t2, o, v, np)[np.ix_(nvs,all_v,all_o,all_o)], u[o, o, v, v]
    )
    return Fae


def build_reduced_Fmi(f, u, t1, t2, o, v, np,nos,nvs):
    all_v=np.arange(t1.shape[0])
    all_o=np.arange(t1.shape[1])
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fmi = np.zeros((len(all_o), len(nos)), dtype=t1.dtype)

    Fmi += f[o, o][np.ix_(all_o,nos)]
    Fmi += 0.5 * contract("ei,me->mi", t1[np.ix_(all_v,nos)], f[o, v][np.ix_(all_o,all_v)])
    Fmi += 2 * contract("en,mnie->mi", t1[np.ix_(all_v,all_o)], u[o, o, o, v][np.ix_(all_o,all_o,nos,all_v)])
    Fmi -= contract("en,mnei->mi", t1[np.ix_(all_v,all_o)], u[o, o, v, o][np.ix_(all_o,all_o,all_v,nos)])
    Fmi += 2 * contract(
        "efin,mnef->mi", build_tilde_tau(t1, t2, o, v, np)[np.ix_(all_v,all_v,nos,all_o)], u[o, o, v, v]
    )
    Fmi -= contract(
        "efin,mnfe->mi", build_tilde_tau(t1, t2, o, v, np)[np.ix_(all_v,all_v,nos,all_o)], u[o, o, v, v]
    )
    return Fmi
def build_reduced_Fme(f, u, t1, o, v, np,nos,nvs):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fme = np.zeros((nocc, nvirt), dtype=t1.dtype)
    Fme += f[o, v]
    Fme += 2 * contract("fn,mnef->me", t1, u[o, o, v, v])
    Fme -= contract("fn,mnfe->me", t1, u[o, o, v, v])
    return Fme
