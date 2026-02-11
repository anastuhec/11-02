# %%
import mpmath, scipy, os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'savefig.dpi':300, 'axes.labelweight':'normal'})
matplotlib.rcParams['axes.linewidth'] = 0.8
from matplotlib import rc
preamble = r'''
\usepackage{physics} \usepackage{upgreek} \usepackage{mhchem}
'''
plt.rc('text.latex', preamble=preamble)
rc('text', usetex=True)
import random
import scipy.ndimage
from numpy import ndarray
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from scipy.optimize import brentq
import numpy.linalg as LA
import time as time
import random, os

from ta2nise5_funkcija import *
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\slike")

U=2.5
V=0.785
a=3.51
b=15.79

# %%
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")
f = open('parametri.txt', 'r')
with open('parametri-kinetic.txt', 'w') as f_new:
   pass 
i=0
with open('parametri-kinetic.txt', 'a') as f_new:
    with open('parametri.txt', 'r') as f:
        for line in f:
            if i == 0 or line == '\n': pass
            else: f_new.write(line)
            i += 1 
a, b = 3.51, 15.79 # angstrom
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\slike")

# %%
''' gostotna matrika pri T=0: zapolnjene nikljeve orbitale '''
def Rho0(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in [4, 5]: rho0[i, i, :, :] = 1
    return rho0

''' gostotna matrika pri T=infty: vse orbitale zasedene z enako verjetnostjo '''
def Rhoinfty(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in range(6): rho0[i, i, :, :] = 2/6
    return rho0

''' naredi osnovno stanje '''
def GS(Nx, Ny, eps, maxiter=1000, N_epsilon=5, eps0=0.1, U=2.5, V=0.785, a=3.51, b=15.79):
    T = 0 
    Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2)/Nx
    Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2)/Ny
    Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
    H_hop = H_hopping(Kymesh, Kxmesh, a, b)
    rho = Rho0(Ny, Nx)
    err, N_iters = 1, 0
    while err > eps and N_iters < maxiter:
        if N_iters < N_epsilon: epsilon = eps0
        else: epsilon = 0
        rho, _, err, _ = F(Kymesh, Kxmesh, rho, T, H_hop, U, V, a, b, epsilon=epsilon, colors='yes')
        N_iters += 1
    print(f'err={err}')
    return rho

''' kineticni clen '''
def H_hopping(Kymesh, Kxmesh, a, b, file='parametri-kinetic.txt'):
    Ny, Nx = Kymesh.shape
    hop = np.zeros((6, 6, Ny, Nx), dtype='complex')
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            ad = t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
            hop[orb1 - 1, orb2 - 1] += ad
            if orb1 != orb2: hop[orb2 - 1, orb1 - 1] += ad.conjugate()
    return hop

''' Hartree interakcijski clen '''
def H_hartree(rho, U, V, file='parametri-interaction.txt'):
    (I, J, Ny, Nx) = rho.shape
    Nk = Nx*Ny
    hartree_k = np.zeros((I,J), dtype='complex')
    with open(file) as f:
        for line in f:
            [orb1, orb2] = list(map(float, line.split()))[-2:]
            orb1, orb2 = int(orb1-1), int(orb2-1)
            if orb1 == orb2: 
                hartree_k[orb1, orb1] += U * np.sum(rho[orb1,orb1])
            else:
                hartree_k[orb1, orb1] += 2 * V * np.sum(rho[orb2, orb2])
                hartree_k[orb2, orb2] += 2 * V * np.sum(rho[orb1, orb1])
    return hartree_k / Nk

''' Fockov interakcijski clen '''
def H_fock2(Kymesh, Kxmesh, rho, V, a, b, file='parametri-interaction.txt'):
    Ny, Nx = Kymesh.shape
    Nk = Nx*Ny
    fock = np.zeros(rho.shape, dtype='complex')
    with open(file) as f:
        for line in f:
            [x, y, orb1, orb2] = list(map(float, line.split()))
            if orb1 == orb2: pass # these are hartree contributions
            else:
                Vq = -V*np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
                konv = np.roll(scipy.signal.convolve2d(rho[int(orb1 - 1), int(orb2 - 1)], np.block([[Vq, Vq, Vq], [Vq, Vq, Vq], [Vq, Vq, Vq]]), mode='same', ), (-1,-1), axis=(0,1)) / Nk
                # opomba: ne uporabljaj scipy.ndimage.convolve!! tisto dela počasneje, idk zakaj
                #konv = scipy.ndimage.convolve(rho[int(orb1-1),int(orb2-1)], Vq, mode='wrap') / Nk
                fock[int(orb1 - 1), int(orb2 - 1)] += konv
                fock[int(orb2 - 1), int(orb1 - 1)] += konv.conjugate()
    return fock

''' tu upoštevamo kar se da izpeljat za Fockov člen v momentnem prostoru
    naredi isto kot H_fock2, vendar mnogo hitreje '''
def H_fock(Kymesh, Kxmesh, rho, V, a, b):
    fock = np.zeros(rho.shape, dtype='complex')
    phi = Phi(Kxmesh, rho, a)
    fock[0,4] = - V * Delta(Kxmesh, rho, 0, 4, 0)*(1 - np.exp(-1j*Kxmesh*a)) - V * phi[0] * np.exp(-1j*Kxmesh*a)
    fock[4,0] = fock[0,4].conjugate()
    fock[1,4] = - V * Delta(Kxmesh, rho, 1, 4, 0)*(1 - np.exp(-1j*Kxmesh*a)) - V * phi[1] * np.exp(-1j*Kxmesh*a)
    fock[4,1] = fock[1,4].conjugate()
    fock[2,5] = - V * Delta(Kxmesh, rho, 2, 5, 0)*(1 - np.exp(1j*Kxmesh*a)) - V * phi[2] * np.exp(1j*Kxmesh*a)
    fock[5,2] = fock[2,5].conjugate()
    fock[3,5] = - V * Delta(Kxmesh, rho, 3, 5, 0)*(1 - np.exp(1j*Kxmesh*a)) - V * phi[3] * np.exp(1j*Kxmesh*a)
    fock[5,3] = fock[3,5].conjugate()
    return fock

''' motnja, ki zlomi simetrijo. eps je majhen parameter '''
def H_perturb(Kymesh, Kxmesh, a, b, eps, file='perturbacija.txt'):
    return eps * H_hopping(Kymesh, Kxmesh, a, b, file=file)

''' hibridizacija Delta_ij(x) '''
def Delta(Kxmesh, rho, i, j, x): 
    Ny, Nx = Kxmesh.shape
    Nk = Nx*Ny
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(1j * Kxmesh * x1)) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(1j * Kxmesh * x)) / Nk

''' Phi: order parameter (4-komponentni vektor) '''
def Phi(Kxmesh, rho, a):
    pos12 = np.array([0, a])
    pos34 = np.array([0, -a])
    phi = np.zeros(4, dtype='complex')
    phi[0] = np.sum(Delta(Kxmesh, rho, 0, 4, pos12))
    phi[1] = np.sum(Delta(Kxmesh, rho, 1, 4, pos12))
    phi[2] = np.sum(Delta(Kxmesh, rho, 2, 5, pos34))
    phi[3] = np.sum(Delta(Kxmesh, rho, 3, 5, pos34))
    return phi

''' H and diagonalize it. fs are FD distributions for each m in mu (can be either float or an array) '''
def H_diagonalize(Kymesh, Kxmesh, rho, H_hop, U, V, a, b, T, mu=0., epsilon=0):
    Ny, Nx = Kxmesh.shape
    H = H_hop + H_fock(Kymesh, Kxmesh, rho, V, a, b)
    if epsilon != 0: H += H_perturb(Kymesh, Kxmesh, a, b, epsilon, file='perturbacija.txt')
    h_hartree = H_hartree(rho, U, V)
    energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
    fs = np.zeros((6, 6, Ny, Nx))
    for m in range(Ny):
        for n in range(Nx):
            en, v = LA.eigh(H[:, :, m, n] + h_hartree)
            energije[:, m, n] = en
            vecs[:, :, m, n] = v
            if T == 0: np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty': np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n], 1/(1 + np.exp((en - mu)/T)))
    return energije, vecs, fs

def Rho_new(vecs, fs):
    return np.einsum('ijkl,jmkl,mnkl-> inkl', vecs, fs, np.swapaxes(vecs.conj(), 0, 1))

def F(Kymesh, Kxmesh, rho, T, H_hop, U, V, a, b, epsilon=0, mu=0., colors='no', occupation='no', vectors='no'):
    energije, vecs, fs = H_diagonalize(Kymesh, Kxmesh, rho, H_hop, U, V, a, b, T, mu, epsilon)
    rho_new = Rho_new(vecs, fs)
    if colors == 'yes':
        barve = np.einsum('ijkl->jkl', np.abs(vecs[:4, :, :, :])**2)
        return rho_new, energije, np.max(np.abs(rho - rho_new)), barve
    if occupation == 'no': return rho_new, energije, np.max(np.abs(rho - rho_new))
    elif occupation == 'yes' and vectors == 'yes': return rho_new, energije, np.linalg.norm(rho - rho_new, ord='fro'), fs, vecs #np.max(np.abs(rho - rho_new)), fs, vecs

''' zasedenost '''
def Zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijkl->ij', rho)))/(np.prod(rho.shape[-2:]))).real

'''to make Fermi-Dirac distribution on entire array (is it possible to avoide for loops here?) '''
def fil(fs, energije, T, mu):
    Ny, Nx = energije.shape[1:]
    for m in range(Ny):
        for n in range(Nx):
            np.fill_diagonal(fs[:,:,m,n], 1/(1 + np.exp((energije[:,m,n] - mu)/T)))
    return fs


# %%



def remove_nonhermiticity(rho):
    (I, J, Ny, Nx) = rho.shape
    rho_new = np.copy(rho)
    for i in range(I):
        for j in range(i):
            for k in range(Ny//2):
                rho_new[i,j,-(k+1),Nx//2:] = rho[i,j,k+1,1:Nx//2+1][::-1].conjugate()
    return rho_new


# %%
def Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, T, mu, eps, maxiter, mix):
    err, err_new = 1, 1
    iters = 0
    orders = []
    while err > eps and iters < maxiter:
        if iters < 5: eps0 = 0.1
        else: eps0 = 0
        err = err_new
        rho_new, _, err_new = F(Kymesh, Kxmesh, rho, T, H_hop, U, V, a, b, mu=mu, epsilon=eps0)
        #rho_new = remove_nonhermiticity(rho_new)
        rho = rho_new * mix + rho * (1 - mix)
        iters += 1
        #if err_new > err and err < 1e-6: break
        orders.append(Phi(Kxmesh, rho, a)[0].real)
        #print(err_new)
        #print(Phi(Kxmesh, rho, a)[0].real)
        #if iters > 101 and np.std(orders[-100:]) < 0.0001: #and err < 1e-5:
        #    print('yes')
        #    break
    return rho, err, Zasedenost(rho), iters

def newMu_highT(Kymesh, Kxmesh, rho, mu, beta, H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix, chi, mix3, n_pass, first='last', fix_faktor='no'):
    if first=='last':
        rho_a, err_a, n_a, iters_a = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, 1/beta, mu, eps_last, maxiter_last, mix)
    elif first =='first':
        rho_a, err_a, n_a, iters_a = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, 1/beta, mu, eps, maxiter, mix)
    print(f'{n_a, err_a}')
    if np.abs(n_a - 2) < n_pass and err_a < eps_last:
        #print(f'final: {beta}, {n_a}, {err_a}, {iters_a}')
        return rho_a, err_a, n_a, mu, iters_a
    pogoj = False
    koraki = 0
    if fix_faktor == 'no':
        if np.abs(chi) > 0: faktor = (n_a - 2)/chi * mix3
        else: faktor = faktor
    elif fix_faktor == 'yes': faktor = faktor
    if chi >= 0:
        if n_a > 2:
            sign = -1
        elif n_a < 2: sign = +1
    elif chi < 0:
        if n_a > 2: sign = +1
        elif n_a < 2: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - 2)
    ns = np.array([0, n_a])
    mus = [0, mu]
    while sgns[0] == sgns[1]:
        #print('f')
        print(f'mu = {mu + faktor*koraki*sign}')
        rho_b, err_b, n_b, iters_b = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, 1/beta, mu + faktor*koraki*sign, eps, maxiter, mix)
        if np.abs(n_b - 2) < n_pass and err_b < eps_last: return rho_b, err_b, n_b, mu + faktor*koraki*sign, iters_b
        print(f'{n_b, err_b}')
        #print(f' {n_b}, {err_b}')
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - 2)
        if sgns[0] != sgns[1]: break
        if n_b < 2 and n_b < ns[1]: sign *= -1
        if n_b > 2 and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - 2)
        koraki +=1
        
    mus = np.sort(np.array([mu + faktor*(koraki-0)*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    
    while pogoj == False:
        mu_mid = (mus[0] + mus[1])/2
        rho_mid, err_mid, n_mid, iters_mid = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, 1/beta, mu_mid, eps, maxiter, mix)
        print('bisection')
        print(f'{n_mid, err_mid}')
        if n_mid > 2: mus[1] = mu_mid
        elif n_mid < 2: mus[0] = mu_mid
        #print(f' {n_mid}, {err_mid}')
        if np.abs(n_mid - 2) < n_pass: break
        trials +=1 
        if trials > 10: break
    rho_mid, err_mid, n_mid, iters_mid = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho, 1/beta, mu_mid, eps_last, maxiter_last, mix)
    #print(f'final: {beta}, {n_mid}, {err_mid}, {iters_mid}')
    return rho_mid, err_mid, n_mid, mu_mid, iters_mid

def F_beta(Kymesh, Kxmesh, rho, mu, betas, H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='last'):
    rho0 = np.copy(rho)
    N = len(betas)  
    Mus, ns, errs, orders, chis, iters, mixs = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    for i, beta in enumerate(betas):

        if i == 0:
            mu_first = mu
            chi = 1
        else:
            _, _, n_a, _ = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho0, 1/beta, mu, eps, maxiter, mix)
            _, _, n_b, _ = Rho_next(Kymesh, Kxmesh, H_hop, U, V, a, b, rho0, 1/beta, mu + dmu, eps, maxiter, mix)
            chi = (n_b - n_a)/dmu
            chis[i] = chi
            if np.abs(chi) > 0: mu_first = Mus[i-1] - mix2 * (n_a- 2)/np.abs(chi)
            else: mu_first = Mus[i-1]
            if i > 2:
                dmu = Mus[i-2] - Mus[i-1]
                if dmu == 0: dmu = np.max(np.diff(Mus))
        rho, err1, n1, mu1, iters1 = newMu_highT(Kymesh, Kxmesh, rho0, mu_first, beta, H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix, chi, mix3, n_pass, first=first)
        if i == N-2: np.save('rhotrial.npy', rho)
        print(f'beta={beta}, phi={Phi(Kxmesh, rho, a)[0].real}, err={err1}, n1={n1}')
        Mus[i], ns[i], errs[i], orders[i], iters[i], mixs[i] = mu1, n1, err1, Phi(Kxmesh, rho, a)[0].real, iters1, mix
        print('---')

    return Mus, ns, errs, orders, chis, rho, iters, mixs

def F_shoot(Kymesh, Kxmesh, rho, mu, betas, H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix2, mix, mix3, n_pass, first='last', fix_faktor='no'):
    Ny, Nx = Kymesh.shape
    N = len(betas)
    rho0 = np.copy(rho)

    for i, beta in enumerate(betas):
        print(i)
        energije, vecs, fs = H_diagonalize(Kymesh, Kxmesh, rho0, H_hop, U, V, a, b, 1/beta, mu)

        H = H_hop + H_fock(Kymesh, Kxmesh, rho, V, a, b)
        h_hartree = H_hartree(rho, U, V)
        energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
        fs = np.zeros((2, 6, 6, Ny, Nx))
        for m in range(Ny):
            for n in range(Nx):
                en, v = LA.eigh(H[:, :, m, n] + h_hartree)
                energije[:, m, n] = en
                vecs[:, :, m, n] = v
                np.fill_diagonal(fs[0, :, :, m, n], 1/(1 + np.exp((en - mu)*beta)))
                np.fill_diagonal(fs[1, :, :, m, n], 1/(1 + np.exp((en - (mu + dmu))*beta)))
        rho1, rho2 = Rho_new(vecs, fs[0]), Rho_new(vecs, fs[1])
        n1, n2 = Zasedenost(rho1), Zasedenost(rho2)
        chi = (n2 - n1)/dmu
        if np.abs(chi) > 0: mu = mu - mix2 * (n1 - 2)/np.abs(chi)
        else: mu = mu

        if i != N-1:
            rho, err, n, mu, iters = newMu_highT(Kymesh, Kxmesh, rho0, mu, beta, H_hop, U, V, a, b, dmu, eps1, eps1, maxiter1, maxiter1, faktor, mix, chi, mix3, n_pass, first='first', fix_faktor=fix_faktor)
        else: 
            rho, err, n, mu, iters = newMu_highT(Kymesh, Kxmesh, rho0, mu, beta, H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix, chi, mix3, n_pass, first='first', fix_faktor=fix_faktor)

    return Phi(Kxmesh, rho, a)[0].real, mu, n, err, iters, rho


''' z natancnostjo eps1 doloci vmesne, z eps2 koncnega'''
def Shoot_beta(Kymesh, Kxmesh, rho, mu, beta0, scale, N, H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix2, mix, mix3, n_pass, first='first'):
    betas = beta0/scale**np.arange(1,N + 2)
    values = {'betas' : np.empty(N), 'orders' : np.empty(N), 'mus' : np.empty(N), 'ns' : np.empty(N), 'errs' : np.empty(N), 'iters' : np.empty(N)}
    for i in range(N):
        print(betas[i+1])
        res = F_shoot(Kymesh, Kxmesh, rho, mu, betas[1:i+2], H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix2, mix, mix3, n_pass, first=first)
        values['orders'][i], values['mus'][i], values['ns'][i], values['errs'][i], values['iters'][i] = res[:-1]
        rho_new = res[-1]
        values['betas'][i] = betas[i+1]
        print(values['orders'][i], values['errs'][i])
    return values, rho_new

''' z natancnostjo eps1 doloci vmesne, z eps2 koncnega'''
def Shoot_beta2(Kymesh, Kxmesh, rho, mu, beta0, scale, N, k, H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix2, mix, mix3, n_pass, first='last'):
    betas = beta0/scale**np.arange(1)
    values = {'betas' : np.empty(N//k+1), 'orders' : np.empty(N//k+1), 'mus' : np.empty(N//k+1), 'ns' : np.empty(N//k+1), 'errs' : np.empty(N//k+1), 'iters' : np.empty(N//k+1)}
    for i in range(N+k):
        if i % k == 0:
            print(betas[-1])
            values['betas'][i//k] = betas[-1]
            res = F_shoot(Kymesh, Kxmesh, rho, mu, betas[:], H_hop, U, V, a, b, dmu, eps1, eps2, maxiter1, maxiter2, faktor, mix2, mix, mix3, n_pass, first=first)
            values['orders'][i//k], values['mus'][i//k], values['ns'][i//k], values['errs'][i//k], values['iters'][i//k] = res[:-1]
            rho_new = res[-1]
            betas = values['betas'][i//k]/scale**np.arange(1+k)
            mu, rho = values['mus'][i//k], rho_new
            print(values['orders'][i//k])
    return values, rho_new

class system:
  def __init__(self, Ny, Nx, a, b):
    Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2)/Nx
    Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2)/Ny
    Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
    self.kx = Kxmesh
    self.ky = Kymesh
    self.gs = GS(Nx, Ny, 1e-10)
    self.H_hop = H_hopping(Kymesh, Kxmesh, a, b)

# %%
Ny, Nx = 40, 40
s = system(Ny, Nx, a, b)


# %%

Betas = np.hstack([50/1.03**np.arange(1,11), 37.20469574483625/1.03**np.arange(1,25), betas3, betas4, betas5, betas6, betas7])
Mus = np.hstack([mus, mus2, mus3, mus4, mus5, mus6, mus7])
Orders = np.hstack([orders, orders2, orders3, orders4, orders5, orders6, orders7])
Ns = np.hstack([ns, ns2, ns3, ns4, ns5, ns6, ns7])
Errs = np.hstack([errs, errs2, errs3, errs4, errs5, errs6, errs7])
plt.plot(Betas, Errs)

np.save('Ns-0504-40.npy', Ns)

# %%
beta0 = 50
mu = 2.84
betas = beta0/1.03**np.arange(1,11)

print(f'started calculation with initial mu={mu}, beta={beta0}. lowest beta is {betas[-1]}' + '\n' + '---')

maxiter = 200
maxiter_last = 1000
eps = 1e-4
eps_last = 1e-9
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-4

mus, ns, errs, orders, chis, Rho, iters, mixs = F_beta(s.ky, s.kx, s.gs, mu, betas, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
mu = mus[-1]
betas = 37.20469574483625/1.03**np.arange(1,25)

print(f'started calculation with initial mu={mu}, beta={beta0}. lowest beta is {betas[-1]}' + '\n' + '---')

maxiter = 200
maxiter_last = 1000
eps = 1e-4
eps_last = 1e-9
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-4

mus2, ns2, errs2, orders2, chis2, Rho2, iters2, mixs2 = F_beta(s.ky, s.kx, Rho, mu, betas, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
mu = mus2[-1]
beta0 = 18.302244987131925
scale = 1.005
N = 20

print(beta0/scale**N)

maxiter = 100
maxiter_last = 2000
eps = 1e-5
eps_last = 1e-8

dd, mix2 = 0.0001, 0.1
faktor = 0.01
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-4

betas3 = beta0/scale**np.arange(20)
mus3, ns3, errs3, orders3, chis3, Rho3, iters3, mixs3 = F_beta(s.ky, s.kx, Rho2, mus2[-1], betas3, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
beta0 = betas3[-1]
scale = 1.01
N = 10

maxiter = 100
maxiter_last = 2500
eps = 1e-3
eps_last = 1e-8
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-3

betas4 = beta0/scale**np.arange(20)
mus4, ns4, errs4, orders4, chis4, Rho4, iters4, mixs4 = F_beta(s.ky, s.kx, Rho3, mus3[-1], betas4, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
beta0 = betas4[-1]
scale = 1.01
N = 10

maxiter = 100
maxiter_last = 2500
eps = 1e-3
eps_last = 1e-8
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-3

betas5 = beta0/scale**np.arange(20)
mus5, ns5, errs5, orders5, chis5, Rho5, iters5, mixs5 = F_beta(s.ky, s.kx, Rho4, mus4[-1], betas5, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
beta0= betas5[-1]
scale = 1.008
N = 10

maxiter = 100
maxiter_last = 2500
eps = 1e-3
eps_last = 1e-8
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-4

betas6 = beta0/scale**np.arange(1,10)
mus6, ns6, errs6, orders6, chis6, Rho6, iters6, mixs6 = F_beta(s.ky, s.kx, Rho5, mus5[-1], betas6, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
beta0= betas6[-1]
scale = 1.002
N = 10

maxiter = 100
maxiter_last = 3000
eps = 1e-3
eps_last = 1e-8
print(betas[-1])

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-4

betas7 = beta0/scale**np.arange(1,7)
mus7, ns7, errs7, orders7, chis7, Rho7, iters7, mixs7 = F_beta(s.ky, s.kx, Rho6, mus6[-1], betas7, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='first')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 100
eps = 1e-3
eps_last = 1e-6

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 0.02

mu =  2.5
betas1 = 7.5*1.01**np.arange(0,2)
phi1, mu1, n1, err1, iters1, rho1 = F_shoot(s.ky, s.kx, Rhoinfty(Ny, Nx), mu, betas1, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas2 = 8*1.01**np.arange(0,2)
phi2, mu2, n2, err2, iters2, rho2 = F_shoot(s.ky, s.kx, rho1, mu1, betas2, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas3 = 8.5*1.01**np.arange(0,2)
phi3, mu3, n3, err3, iters3, rho3 = F_shoot(s.ky, s.kx, rho2, mu2, betas3, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas4 = 9*1.01**np.arange(0,2)
phi4, mu4, n4, err4, iters4, rho4 = F_shoot(s.ky, s.kx, rho3, mu3, betas4, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas5 = 9.5*1.01**np.arange(0,2)
phi5, mu5, n5, err5, iters5, rho5 = F_shoot(s.ky, s.kx, rho4, mu4, betas5, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas6 = 9.65*1.01**np.arange(0,2)
phi6, mu6, n6, err6, iters6, rho6 = F_shoot(s.ky, s.kx, rho5, mu5, betas6, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas7 = 9.75*1.01**np.arange(0,2)
phi7, mu7, n7, err7, iters7, rho7 = F_shoot(s.ky, s.kx, rho6, mu6, betas7, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas8 = 9.8*1.01**np.arange(0,2)
phi8, mu8, n8, err8, iters8, rho8 = F_shoot(s.ky, s.kx, rho7, mu7, betas8, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas9 = 9.9*1.01**np.arange(0,2)
phi9, mu9, n9, err9, iters9, rho9 = F_shoot(s.ky, s.kx, rho8, mu8, betas9, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas10 = 9.95*1.03**np.arange(0,2)
phi10, mu10, n10, err10, iters10, rho10 = F_shoot(s.ky, s.kx, rho9, mu9, betas10, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas11 = 9.98*1.03**np.arange(0,2)
phi11, mu11, n11, err11, iters11, rho11 = F_shoot(s.ky, s.kx, rho10, mu10, betas11, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas12 = 10.1*1.03**np.arange(0,2)
phi12, mu12, n12, err12, iters12, rho12 = F_shoot(s.ky, s.kx, rho11, mu11, betas12, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas13 = 10.2*1.03**np.arange(0,2)
phi13, mu13, n13, err13, iters13, rho13 = F_shoot(s.ky, s.kx, rho12, mu12, betas13, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas14 = 10.3*1.03**np.arange(0,2)
phi14, mu14, n14, err14, iters14, rho14 = F_shoot(s.ky, s.kx, rho13, mu13, betas13, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas14 = 10.4*1.03**np.arange(0,2)
phi14, mu14, n14, err14, iters14, rho14 = F_shoot(s.ky, s.kx, rho13, mu13, betas14, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Ny, Nx = 40, 40
maxiter = 100
maxiter_last = 2000
eps = 1e-3
eps_last = 1e-8

faktor = 0.005
dmu = 0.01
mix2 = 0.05
mix = 0.3
mix3 = 0.5
n_pass = 1e-3

betas15 = 10.55*1.03**np.arange(0,2)
phi15, mu15, n15, err15, iters15, rho15 = F_shoot(s.ky, s.kx, rho14, mu14, betas15, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, fix_faktor ='yes')

# %%
Betas_single = np.hstack([u[-1] for u in [betas1, betas2, betas3, betas4, betas5, betas6, betas7, betas8,
                                          betas9, betas10, betas11, betas12, betas13, betas14, betas15]])
Ns_single = np.hstack([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15])
Orders_single = np.hstack([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10,phi11,phi12,phi13,phi14,phi15])
Errs_single = np.hstack([err1,err2,err3,err4,err5,err6,err7,err8,err9,err10,err11,err12,err13,err14,err15])
Mus_single = np.hstack([mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8,mu9,mu10,mu11,mu12,mu13,mu14,mu15])

np.save('Betas-single-0504-40.npy', Betas_single)
np.save('Ns-single-0504-40.npy', Ns_single)
np.save('Orders-single-0504-40.npy', Orders_single)
np.save('Mus-single-0504-40.npy', Mus_single)
np.save('Errs-single-0504-40.npy', Errs_single)

# %%
betas = np.load('Betas-0504-40.npy')
orders = np.load('Orders-0504-40.npy')
ns = np.load('Ns-0504-40.npy')
errs = np.load('Errs-0504-40.npy')
mus = np.load('Mus-0504-40.npy')

betas_s = np.load('Betas-single-0504-40.npy')
orders_s = np.load('Orders-single-0504-40.npy')
ns_s = np.load('Ns-single-0504-40.npy')
errs_s = np.load('Errs-single-0504-40.npy')
mus_s = np.load('Mus-single-0504-40.npy')

fig, ax = plt.subplots(ncols=2,nrows=2, figsize=(8,6))
ins = ax[0,0].inset_axes([0.45,0.3,0.45,0.45])

ax[0,0].plot(betas, orders, lw=1, color='grey')
ax[0,0].plot(betas, orders, '.', ms=1, color='blue')
ins.plot(betas, orders, color='grey', lw=1)
ins.plot(betas, orders, '.', ms=1, color='blue')
ax[0,1].plot(betas, mus, lw=1, color='grey')
ax[0,1].plot(betas, mus, '.', color='blue', ms=1)
ax[1,0].plot(betas, np.abs(ns-2), lw=1, color='grey')
ax[1,0].plot(betas, np.abs(ns-2), '.', color='blue', ms=1)
ax[1,1].plot(betas, errs, lw=1, color='grey')
ax[1,1].plot(betas, errs, '.', color='blue', ms=1)

ax[0,0].plot(betas_s, orders_s, lw=1, color='grey')
ax[0,0].plot(betas_s, orders_s, '.', ms=1, color='red')
ins.plot(betas_s, orders_s, lw=1, color='grey')
ins.plot(betas_s, orders_s, '.', ms=1, color='red')
ax[0,1].plot(betas_s, mus_s, color='grey', lw=1)
ax[0,1].plot(betas_s, mus_s, '.', ms=1, color='red')
ax[1,0].plot(betas_s, np.abs(ns_s-2), color='grey', lw=1)
ax[1,0].plot(betas_s, np.abs(ns_s-2), '.', color='red', ms=1)
ax[1,1].plot(betas_s, errs_s, color='grey', lw=1)
ax[1,1].plot(betas_s, errs_s, '.', ms=1, color='red')

for i in range(2):
    for j in range(2):
        ax[i,j].set_xlabel(r'$\beta$', fontsize=13)
        ax[i,j].set_xticks([5*i for i in range(11)])
        ax[i,j].set_xticklabels([rf'${5*i}$' for i in range(11)], fontsize=10)
ax[1,0].set_yscale('log'), ax[1,1].set_yscale('log')
ax[0,0].set_ylabel(r'$\phi$', fontsize=13)
ax[0,1].set_ylabel(r'$\mu$', fontsize=13)
ax[1,0].set_ylabel(r'$|n-2|$', fontsize=13)
ax[1,1].set_ylabel(r'$||\rho - F(\rho)||$', fontsize=13)
ax[0,0].set_title(r'Order parameter', fontsize=13)
ax[0,1].set_title(r'Chemical potential', fontsize=13)
ax[1,0].set_title(r'Deviation from occupation $n=2$', fontsize=13)
ax[1,1].set_title(r'Error in last iteration', fontsize=13)
fig.suptitle(r'$\ce{Ta2NiSe5}$ crystal with $40\times 40$ unit cells, ' + rf'$U={U}, V={V}$,' + '\n' + r'124 $\beta$ points, 2.5 hours', fontsize=16)


ax[0,1].scatter(60, 2.8, s=10, label=r'from high to low $\beta$', color='blue')
ax[0,1].scatter(60, 2.8, s=10, label=r'from low to high $\beta$', color='red')
ax[0,1].legend(facecolor='lightyellow', fontsize=10).set_title(r'computed:', prop={'size':11})
for i in range(2):
    for j in range(2):
        ax[i,j].set_xlim(5,50)
ins.set_xlabel(r'$\beta$'), ins.set_ylabel(r'$\phi$')
ins.set_xlim(8,15), ins.set_yticks([0,0.025,0.05,0.075,0.10,0.125]), ins.set_yticklabels([rf'${u}$' for u in [0,0.025,0.05,0.075,0.1,0.125]], fontsize=8)
ins.set_xticks([8,9,10,11,12,13,14,15]), ins.set_xticklabels([rf'${u}$' for u in range(8,16)], fontsize=8)
plt.tight_layout()
plt.savefig('05-04-40crystal.png')

# %%
Betas, Orders, Mus, Errs, Ns = np.load('Betas-40.npy'), np.load('Orders-40.npy'), np.load('Mus-40.npy'), np.load('Errs-40.npy'), np.load('Ns-40.npy')
fig, ax = plt.subplots(ncols=4, figsize=(12,3))
ins = ax[0].inset_axes([0.5,0.3,0.4,0.4])
ax[0].plot(Betas[:-40], Orders[:-40], '.', ms=1)
ins.plot(Betas[:-40], Orders[:-40], '.', ms=1), ins.set_xlabel(r'$\beta$'), ins.set_ylabel(r'$\phi$')
ins.set_xlim(5,20)
ax[1].plot(Betas[:-40], Mus[:-40], '.', ms=1)
ax[2].plot(Betas[:-40], np.abs(Ns-2)[:-40], '.', ms=1)
ax[3].plot(Betas[:-40], Errs[:-40], '.', ms=1)
for i in range(4):
    ax[i].set_xlabel(r'$\beta$', fontsize=13)
    ax[i].set_xticks([5*i for i in range(11)])
    ax[i].set_xticklabels([rf'${5*i}$' for i in range(11)], rotation=45, fontsize=8)
ax[2].set_yscale('log'), ax[3].set_yscale('log')
ax[0].set_ylim(-0.001,0.15)
ax[0].set_ylabel(r'$\phi$', fontsize=13)
ax[1].set_ylabel(r'$\mu$', fontsize=13)
ax[2].set_ylabel(r'$|n-2|$', fontsize=13)
ax[3].set_ylabel(r'$||\rho - F(\rho)||$', fontsize=13)
ax[0].set_title(r'order parameter', fontsize=13)
ax[1].set_title(r'chemical potential', fontsize=13)
ax[2].set_title(r'deviation from occupation $n=2$', fontsize=13)
ax[3].set_title(r'error in last iteration', fontsize=13)
fig.suptitle(r'$\ce{Ta2NiSe5}$ crystal with $40\times 40$ unit cells, ' + rf'$U={U}, V={V}$', fontsize=16)
betas_low = np.array([betas1[-1], betas2[-1], betas3[-1], betas4[-1], betas5[-1], betas6[-1], betas7[-1], betas8[-1], betas9[-1], betas10[-1], betas11[-1], betas12[-1], betas13[-1], betas14[-1], betas15[-1]])
phis_low = np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12, phi13, phi14,  phi15])
mus_low = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15])
ns_low = np.array([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15])
errs_low = np.array([err1, err2, err3, err4, err5, err6, err7, err8, err9, err10, err11, err12, err13, err14, err15])

ins.plot(betas_low, phis_low, '.', ms=1)
ax[0].plot(betas_low, phis_low, '.', ms=1)
ax[1].plot(betas_low, mus_low, '.', ms=1)
ax[2].plot(betas_low, np.abs(ns_low - 2), '.', ms=1)
ax[3].plot(betas_low, errs_low, '.', ms=1)

ax[1].scatter(60, 2.8, s=10, label=r'ground state')
ax[1].scatter(60, 2.8, s=10, label=r'$T=\infty$ state')
ax[1].legend(frameon=False).set_title(r'initial guess:')
for i in range(4): ax[i].set_xlim(5,50)
ins.set_xlim(5,15)
plt.tight_layout()
#plt.savefig('05-04.png')

# %%
plt.plot(Betas[:-40], Orders[:-40], '.')
plt.plot(betas_low[:], phis_low[:], '.')
plt.xlim(0,15)

phis_low
plt.axvline(11.4)

# %%

print(f'started calculation with initial mu={mu}, beta={beta0}. lowest beta is {betas[-1]}' + '\n' + '---')
beta0 = values4['betas'][-1]
N = 10
scale = 1.001
mu = values4['mus'][-1]
maxiter = 50
maxiter_last = 3000
eps = 1e-3
eps_last = 1e-9

dd, mix2 = 0.0001, 0.1
faktor = 0.001
dmu = 0.001
mix2 = 0.001
mix = 0.5
mix3 = 1.2
n_pass = 1e-2

values5, Rho5 = Shoot_beta(s.ky, s.kx, Rho4, mu, beta0, scale, N, s.H_hop, U, V, a, b, dmu, eps, eps_last, maxiter, maxiter_last, faktor, mix2, mix, mix3, n_pass, first='last')

# %%
np.max(np.abs(Rho2 - Rho5))

# %%
fig, ax = plt.subplots(ncols=5, figsize=(16,4.5))
col1, col2 = 'firebrick', 'steelblue'
jx, jy = Nx//2, Ny//2

mu=scipy.optimize.brentq(f,0,5, args=(Kymesh, Kxmesh, rho1, 1/beta, H_hop, U, V, a, b), xtol=1e-1)
rho1, energije = F(Kymesh, Kxmesh, rho1, 1/beta, H_hop, U, V, a, b, mu=mu)[:2]
zas1 = Zasedenost(rho1)

barve[np.where(np.abs(barve-1) < 1e-5)] = 1
print('done')
# risanje disperzije v posebnih smereh Brillouinove cone
# Gamma - X
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + (Ny//2-1)*2*np.pi/b + np.arange(Nx//2)*2*np.pi/a
    E = energije[i, jy, jx:]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, jy, jx:][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# Z - Gamma
E_c, E_v = np.max(energije[:jy, jx]), np.min(energije[:jy, jx])
gap = E_c - E_v
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + np.arange(Ny//2)*2*np.pi/b
    E = energije[i, :jy, jx]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, :jy, jx][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# M - Z
for i in range(6):
    K = np.arange(Nx//2)*2*np.pi/a
    E = energije[i, 0, jx:][::-1]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, 0, jx:][::-1][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))
xs = np.arange(-3*a,3*a+a/12,a/12)
deltas = np.zeros((4, len(xs)), dtype='complex')

ax[0].axhline(mu)

deltas[0] = Delta(Kxmesh, rho, 0, 4, xs + a/2 )
deltas[1] = Delta(Kxmesh, rho, 1, 4, xs + a/2)
deltas[2] = Delta(Kxmesh, rho, 2, 5, xs - a/2)
deltas[3] = Delta(Kxmesh, rho, 3, 5, xs - a/2)

ax[2].set_xticks([0,1,2,3])
ax[2].set_xticklabels([r'$15$', r'$25$', r'$36$', r'$46$'])
ax[1].axvline(0, lw=1, color='black')
ax[1].axhline(0, lw=1, color='black')
ax[1].axvline(a, lw=1, color='magenta', ls='dashed')
#ax[0,j].set_yticks(range(-1,4))

ins = ax[0].inset_axes([0.25,0.8,0.5,0.1])
for n in range(101):
    ins.axvspan(n, n+1, color=colorFader(col1, col2, n/100))
ax[0].set_xticks([0, (Nx//2 -1)*2*np.pi/a, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])
ax[0].set_xlim([0, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_ylabel(r'$E_k$', fontsize=13)
ins.axis('off')
ins.text(-3, -0.5, r'Ni'), ins.text(95, -0.5, r'Ta')
ins.set_title(r'značaj $\psi_{k}$', fontsize=12)
ax[0].set_xlabel(r'$k$', fontsize=13)
ax[0].set_title(r'Disperzija: rešitve $\hat{H}_k\psi_k = E_k \psi_k $', fontsize=13)

# %%
a, b = 3.51, 15.79 # angstrom

            
T = 0
Nx, Ny = 30, 30
Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2)/Nx
Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2)/Ny
Kxmesh, Kymesh = np.meshgrid(Kx, Ky)

rho0 = Rho0(Ny, Nx)

fig, ax = plt.subplots(ncols=5, figsize=(16,4.5))
col1, col2 = 'firebrick', 'steelblue'
jx, jy = Nx//2, Ny//2

H_hop = H_hopping(Kymesh, Kxmesh, a, b)
V = 0.785
U = 2.5

rho = Rho0(Ny, Nx)

err, N_iters = 1, 0
eps, maxiter = 1e-10, 20
errs = []
eps0, N_epsilon = 0.1, 5
epsilons = []
order_params = []
while err > eps and N_iters < maxiter:

    if N_iters < N_epsilon: epsilon = eps0
    else: epsilon = 0
    rho, energije, err, barve = F(Kymesh, Kxmesh, rho, T, H_hop, U, V, a, b, epsilon=epsilon, colors='yes')
    order_params.append(Phi(Kxmesh, rho, a))
    epsilons.append(epsilon)
    errs.append(err)
    N_iters +=1


print(Phi(Kxmesh, rho, a)[0].real)
barve[np.where(np.abs(barve-1) < 1e-5)] = 1
print('done')
# risanje disperzije v posebnih smereh Brillouinove cone
# Gamma - X
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + (Ny//2-1)*2*np.pi/b + np.arange(Nx//2)*2*np.pi/a
    E = energije[i, jy, jx:]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, jy, jx:][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# Z - Gamma
E_c, E_v = np.max(energije[:jy, jx]), np.min(energije[:jy, jx])
gap = E_c - E_v
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + np.arange(Ny//2)*2*np.pi/b
    E = energije[i, :jy, jx]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, :jy, jx][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# M - Z
for i in range(6):
    K = np.arange(Nx//2)*2*np.pi/a
    E = energije[i, 0, jx:][::-1]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, 0, jx:][::-1][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))
xs = np.arange(-3*a,3*a+a/12,a/12)
deltas = np.zeros((4, len(xs)), dtype='complex')


deltas[0] = Delta(Kxmesh, rho, 0, 4, xs + a/2 )
deltas[1] = Delta(Kxmesh, rho, 1, 4, xs + a/2)
deltas[2] = Delta(Kxmesh, rho, 2, 5, xs - a/2)
deltas[3] = Delta(Kxmesh, rho, 3, 5, xs - a/2)

ax[2].set_xticks([0,1,2,3])
ax[2].set_xticklabels([r'$15$', r'$25$', r'$36$', r'$46$'])
ax[1].axvline(0, lw=1, color='black')
ax[1].axhline(0, lw=1, color='black')
ax[1].axvline(a, lw=1, color='magenta', ls='dashed')
#ax[0,j].set_yticks(range(-1,4))

ins = ax[0].inset_axes([0.25,0.8,0.5,0.1])
for n in range(101):
    ins.axvspan(n, n+1, color=colorFader(col1, col2, n/100))
ax[0].set_xticks([0, (Nx//2 -1)*2*np.pi/a, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])
ax[0].set_xlim([0, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_ylabel(r'$E_k$', fontsize=13)
ins.axis('off')
ins.text(-3, -0.5, r'Ni'), ins.text(95, -0.5, r'Ta')
ins.set_title(r'značaj $\psi_{k}$', fontsize=12)
ax[0].set_xlabel(r'$k$', fontsize=13)
ax[0].set_title(r'Disperzija: rešitve $\hat{H}_k\psi_k = E_k \psi_k $', fontsize=13)
ax[1].set_xlabel(r'$x$ \,[\AA]', fontsize=13)
ax[1].set_ylabel(r'$\Delta_{ij}(x)$', fontsize=13)

ax[1].plot(xs, deltas[0].real, 'x-', label=r'$15$', ms=3, color='steelblue')
ax[1].plot(xs, deltas[1].real, 'o-', label=r'$25$', ms=1, color='lightblue')
ax[1].plot(xs, deltas[2].real, 'x-', label=r'$36$', ms=3, color='purple')
ax[1].plot(xs, deltas[3].real, 'o-', label=r'$46$', ms=1, color='pink')
ax[1].legend(fontsize=10, loc='upper left').set_title(r'$ij$', prop={'size':10})
ax[1].set_title(r'Hibridizacija', fontsize=13)

order_parameters = Phi(Kxmesh, rho, a)

ax[2].scatter(range(4), order_parameters.real, marker='x', c=['steelblue', 'lightblue', 'purple', 'pink'])
ax[2].scatter(range(4), order_parameters.imag, c=['steelblue', 'lightblue', 'purple', 'pink'])


ax[2].set_xlabel(r'$ij$', fontsize=13)
ax[2].set_ylabel(r'$\phi_{ij}$', fontsize=13)
ax[2].scatter(10, 1e-18, color='black', marker='x', label=r'Re$(\phi_{ij})$')
ax[2].scatter(10, 1e-18, color='black', label=r'Im$(\phi_{ij})$')
ax[2].legend()
ax[2].set_xlim([-0.5,3.5])

ax[4].plot(range(len(errs)), errs, '.-', color='black')
ax[4].set_xlabel(r'iteracija $n$',fontsize=13)
ax[4].set_ylabel(r'max$\left[\rho^{(n+1)} - \rho^{(n)}\right]$', fontsize=13)
ax[4].set_title(r'napaka')
ax[4].set_yscale('log')

mid = 0.5 * ((Nx//2 -1)*2*np.pi/a + (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b)
#ax[0].set_xlim([mid - (Nx//2 -1)*2*np.pi/a/2, mid + (Nx//2 -1)*2*np.pi/a/2])
ax[4].axvspan(-1, N_epsilon, zorder=-100, color='red', alpha=0.3)
ax[4].set_xlim(-1, len(epsilons)+1)

order_params = np.array(order_params)
ax[3].plot(range(len(errs)), order_params[:,0].real, '.-', color='steelblue', label=r'$15$')
ax[3].plot(range(len(errs)), order_params[:,1].real, '.-', color='lightblue', label=r'$25$')
ax[3].plot(range(len(errs)), order_params[:,2].real, '.', color='purple', label=r'$36$')
ax[3].plot(range(len(errs)), order_params[:,3].real, '.', color='pink', label=r'$46$')
ax[3].axhline(order_params[-1,0].real, color='limegreen', ls='dashed')
ax[3].text(N_iters//3, order_params[-1,0].real*0.75, rf'$\phi\approx{np.round(order_params[-1,0].real, 3)}$', color='forestgreen', fontsize=12)
ax[3].legend()

ax[3].axvspan(-1, N_epsilon, zorder=-100, color='red', alpha=0.3)
ax[3].set_xlim(-1, len(epsilons)+1)
ax[3].legend(fontsize=10, loc='center right').set_title(r'$ij$', prop={'size':10})
ax[3].set_xlabel(r'iteracija $n$', fontsize=13)
ax[3].set_ylabel(r'$\phi_{ij}^{(n)}$', fontsize=13)
ax[2].set_title(r'Ureditveni parameter', fontsize=13)
ax[3].set_title(r'Razvoj ureditvenega parametra', fontsize=13)

#ax[3].text(0.2, 0, 'motnja' + '\n' + 'vklopljena', fontsize=10, color='red')

ax[1].text(a, ax[1].get_ylim()[1]*.95, '$a$', color='magenta', fontsize=10)
fig.suptitle(rf'$N_k={Nx*Ny}$ točk v 1. BZ, $U={U}, V={V}, \varepsilon={eps0}$ (prvih ${N_epsilon}$ iteracij)', fontsize=15)
plt.tight_layout()
#plt.savefig('razvoj_0.pdf')


# %%
a, b = 3.51, 15.79 # angstrom

            
T = 'infty'
Nx, Ny = 30, 30
Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2)/Nx
Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2)/Ny
Kxmesh, Kymesh = np.meshgrid(Kx, Ky)

rho0 = Rho0(Ny, Nx)

fig, ax = plt.subplots(ncols=5, figsize=(16,4.5))
col1, col2 = 'firebrick', 'steelblue'
jx, jy = Nx//2, Ny//2

H_hop = H_hopping(Kymesh, Kxmesh, a, b)
V = 0.785
U = 2.5

rho = Rhoinfty(Ny, Nx)

err, N_iters = 1, 0
eps, maxiter = 1e-10, 20
errs = []
eps0, N_epsilon = 0.1, 5
epsilons = []
order_params = []
while err > eps and N_iters < maxiter:

    if N_iters < N_epsilon: epsilon = eps0
    else: epsilon = 0
    rho, energije, err, barve = F(Kymesh, Kxmesh, rho, T, H_hop, U, V, a, b, epsilon=epsilon, colors='yes')
    order_params.append(Phi(Kxmesh, rho, a))
    epsilons.append(epsilon)
    errs.append(err)
    N_iters +=1
barve[np.where(np.abs(barve-1) < 1e-5)] = 1
print('done')
# risanje disperzije v posebnih smereh Brillouinove cone
# Gamma - X
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + (Ny//2-1)*2*np.pi/b + np.arange(Nx//2)*2*np.pi/a
    E = energije[i, jy, jx:]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, jy, jx:][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# Z - Gamma
E_c, E_v = np.max(energije[:jy, jx]), np.min(energije[:jy, jx])
gap = E_c - E_v
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + np.arange(Ny//2)*2*np.pi/b
    E = energije[i, :jy, jx]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, :jy, jx][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# M - Z
for i in range(6):
    K = np.arange(Nx//2)*2*np.pi/a
    E = energije[i, 0, jx:][::-1]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, 0, jx:][::-1][g:g+2])
        ax[0].plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))
xs = np.arange(-3*a,3*a+a/12,a/12)
deltas = np.zeros((4, len(xs)), dtype='complex')


deltas[0] = Delta(Kxmesh, rho, 0, 4, xs + a/2 )
deltas[1] = Delta(Kxmesh, rho, 1, 4, xs + a/2)
deltas[2] = Delta(Kxmesh, rho, 2, 5, xs - a/2)
deltas[3] = Delta(Kxmesh, rho, 3, 5, xs - a/2)

ax[2].set_xticks([0,1,2,3])
ax[2].set_xticklabels([r'$15$', r'$25$', r'$36$', r'$46$'])
ax[1].axvline(0, lw=1, color='black')
ax[1].axhline(0, lw=1, color='black')
ax[1].axvline(a, lw=1, color='magenta', ls='dashed')
#ax[0,j].set_yticks(range(-1,4))

ins = ax[0].inset_axes([0.25,0.8,0.5,0.1])
for n in range(101):
    ins.axvspan(n, n+1, color=colorFader(col1, col2, n/100))
ax[0].set_xticks([0, (Nx//2 -1)*2*np.pi/a, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])
ax[0].set_xlim([0, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])
ax[0].set_ylabel(r'$E_k$', fontsize=13)
ins.axis('off')
ins.text(-3, -0.5, r'Ni'), ins.text(95, -0.5, r'Ta')
ins.set_title(r'značaj $\psi_{k}$', fontsize=12)
ax[0].set_xlabel(r'$k$', fontsize=13)
ax[0].set_title(r'Disperzija: rešitve $\hat{H}_k\psi_k = E_k \psi_k $', fontsize=13)
ax[1].set_xlabel(r'$x$ \,[\AA]', fontsize=13)
ax[1].set_ylabel(r'$\Delta_{ij}(x)$', fontsize=13)

ax[1].plot(xs, deltas[0].real, 'x-', label=r'$15$', ms=3, color='steelblue')
ax[1].plot(xs, deltas[1].real, 'o-', label=r'$25$', ms=1, color='lightblue')
ax[1].plot(xs, deltas[2].real, 'x-', label=r'$36$', ms=3, color='purple')
ax[1].plot(xs, deltas[3].real, 'o-', label=r'$46$', ms=1, color='pink')
ax[1].legend(fontsize=10, loc='upper left').set_title(r'$ij$', prop={'size':10})
ax[1].set_title(r'Hibridizacija', fontsize=13)

order_parameters = Phi(Kxmesh, rho, a)

ax[2].scatter(range(4), order_parameters.real, marker='x', c=['steelblue', 'lightblue', 'purple', 'pink'])
ax[2].scatter(range(4), order_parameters.imag, c=['steelblue', 'lightblue', 'purple', 'pink'])


ax[2].set_xlabel(r'$ij$', fontsize=13)
ax[2].set_ylabel(r'$\phi_{ij}$', fontsize=13)
ax[2].scatter(10, 1e-18, color='black', marker='x', label=r'Re$(\phi_{ij})$')
ax[2].scatter(10, 1e-18, color='black', label=r'Im$(\phi_{ij})$')
ax[2].legend()
ax[2].set_xlim([-0.5,3.5])

ax[4].plot(range(len(errs)), errs, '.-', color='black')
ax[4].set_xlabel(r'iteracija $n$',fontsize=13)
ax[4].set_ylabel(r'max$\left[\rho^{(n+1)} - \rho^{(n)}\right]$', fontsize=13)
ax[4].set_title(r'napaka')
ax[4].set_yscale('log')

mid = 0.5 * ((Nx//2 -1)*2*np.pi/a + (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b)
#ax[0].set_xlim([mid - (Nx//2 -1)*2*np.pi/a/2, mid + (Nx//2 -1)*2*np.pi/a/2])
ax[4].axvspan(-1, N_epsilon, zorder=-100, color='red', alpha=0.3)
ax[4].set_xlim(-1, len(epsilons)+1)

order_params = np.array(order_params)
ax[3].plot(range(len(errs)), order_params[:,0].real, '.-', color='steelblue', label=r'$15$')
ax[3].plot(range(len(errs)), order_params[:,1].real, '.-', color='lightblue', label=r'$25$')
ax[3].plot(range(len(errs)), order_params[:,2].real, '.', color='purple', label=r'$36$')
ax[3].plot(range(len(errs)), order_params[:,3].real, '.', color='pink', label=r'$46$')
ax[3].axhline(order_params[-1,0].real, color='limegreen', ls='dashed')
ax[3].text(N_iters//3, order_params[-1,0].real*0.75, rf'$\phi\approx{np.round(order_params[-1,0].real, 3)}$', color='forestgreen', fontsize=12)
ax[3].legend()

ax[3].axvspan(-1, N_epsilon, zorder=-100, color='red', alpha=0.3)
ax[3].set_xlim(-1, len(epsilons)+1)
ax[3].legend(fontsize=10, loc='center right').set_title(r'$ij$', prop={'size':10})
ax[3].set_xlabel(r'iteracija $n$', fontsize=13)
ax[3].set_ylabel(r'$\phi_{ij}^{(n)}$', fontsize=13)
ax[2].set_title(r'Ureditveni parameter', fontsize=13)
ax[3].set_title(r'Razvoj ureditvenega parametra', fontsize=13)

#ax[3].text(0.2, 0, 'motnja' + '\n' + 'vklopljena', fontsize=10, color='red')

ax[1].text(a, ax[1].get_ylim()[1]*.95, '$a$', color='magenta', fontsize=10)
fig.suptitle(rf'$N_k={Nx*Ny}$ točk v 1. BZ, $U={U}, V={V}, \varepsilon={eps0}$ (prvih ${N_epsilon}$ iteracij)', fontsize=15)
plt.tight_layout()
#plt.savefig('razvoj_0.pdf')


# %%
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(10,5))

c = ax[0,0].pcolormesh(Kxmesh, Kymesh, barve[:,:,0], cmap='RdBu', vmin=0, vmax=1)
c = ax[0,1].pcolor(Kxmesh, Kymesh, barve[:,:,1].T, cmap='RdBu', vmin=0, vmax=1)
c = ax[0,2].pcolor(Kxmesh, Kymesh, barve[:,:,2].T, cmap='RdBu', vmin=0, vmax=1)
c = ax[1,0].pcolor(Kxmesh, Kymesh, barve[:,:,3].T, cmap='RdBu', vmin=0, vmax=1)
c = ax[1,1].pcolor(Kxmesh, Kymesh, barve[:,:,4].T, cmap='RdBu', vmin=0, vmax=1)
c = ax[1,2].pcolor(Kxmesh, Kymesh, barve[:,:,4].T, cmap='RdBu', vmin=0, vmax=1)

for i in range(2):
    for j in range(3):
            ax[i,j].text(0, 0, r'$\Gamma$', color='white')
        #ax[i,j].scatter(Kx[-1],0, color='white')
            ax[i,j].text(Kx[-1]-0.1, 0, '$X$', color='white')
        #ax[i,j].scatter(Kx[-1],Ky[0], color='white')
            ax[i,j].text(Kx[-1]-0.1, Ky[0], '$M$', color='white')
        #ax[i,j].scatter(0, Ky[-1], color='white')
            ax[i,j].text(0, Ky[0], '$Z$', color='white')
            ax[i,j].set_xlabel(r'$k_x [A^{-1}]$')
            ax[i,j].set_ylabel(r'$k_y [A^{-1}]$')
clb = fig.colorbar(c, ax=ax[0,3], orientation='vertical')

ax[0,3].axis('off'), ax[1,3].axis('off')
plt.tight_layout()
plt.savefig('barve1.pdf')




