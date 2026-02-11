import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'savefig.dpi':300, 'axes.labelweight':'normal'})
matplotlib.rcParams['axes.linewidth'] = 0.8
from matplotlib import rc
preamble = r'''
\usepackage{physics} \usepackage{upgreek} \usepackage{mhchem} \usepackage{bm}
'''
plt.rc('text.latex', preamble=preamble)
rc('text', usetex=True)
import numpy as np
import scipy, os
import seaborn as sns
import scipy.linalg as LA
import time

os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5")
import tokovi_drugic
import helpers


U = 2.5
V = 0.785
a = 3.51
b = 15.79
b2 = b/10

os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")

''' df/domega, f je Fermi-Diracova porazdelitvena funkcija '''
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

''' create TNS class '''
class TNS:
    def __init__(self, a, b, b2, Ny, Nx, U, V, mu, parameters1, parameters2):
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, a, b)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.mu = mu

        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)
        j = tokovi_drugic.j_tok(self.kymesh, self.kxmesh, a, b, b2)
        j1 = tokovi_drugic.j_1(self.kymesh, self.kxmesh, a, b, b2)
        j2 = tokovi_drugic.j_2(self.kymesh, self.kxmesh, V, a, b, b2)

        j2_matrix_ = np.empty((2,6,6,6,Ny,Nx,Ny,Nx), dtype='complex')
        j2_matrix_[0] = j2[0]
        j2_matrix_[1] = j2[1]
        j_matrix_ = np.empty((2,6,6,Ny,Nx), dtype='complex')
        j_matrix_[0] = j[0]
        j_matrix_[1] = j[1]
        j1_matrix_ = np.empty((2,6,6,Ny,Nx), dtype='complex')
        j1_matrix_[0] = j1[0]
        j1_matrix_[1] = j1[1]
        self.j_matrix = j_matrix_
        self.j1_matrix = j1_matrix_
        self.j2_matrix = j2_matrix_

        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, 0.1, a, U, V, 1e-10, maxiter=1000, N_epsilon=5)
        self.rho0 = self.rho
        self.fock0 = self.fock
        self.hartree0 = self.hartree
        self.phi = helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real

        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []
        self.times_boltzmann = []
        self.times_kubo = []
        self.Ts = []
        self.betas = []
        
        self.boltzmann_L1_xx, self.boltzmann_L0_xx = [], []
        self.boltzmann_L1_yy, self.boltzmann_L0_yy = [], []
        self.boltzmann_L1_xy, self.boltzmann_L0_xy = [], []

        self.kubo_L1_xx, self.kubo_L0_xx = [], []
        self.kubo_L1_yy, self.kubo_L0_yy = [], []
        self.kubo_L1_xy, self.kubo_L0_xy = [], []
        self.kubo_L1_yx, self.kubo_L0_yx = [], []

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        rho, energije, fs, vecs, fock, hartree, err, mu, n = helpers.NewMu(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, self.mu,
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n
        self.times_rho.append(time.time() - start)
        print(1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def boltzmann_koef(self, T) -> None:
        start = time.time()
        L1_xx, L0_xx, L1_yy, L0_yy, L1_xy, L0_xy = tokovi_drugic.Ln_Boltzmann(self.kymesh, self.kxmesh, self.energije, T, self.mu)
        self.boltzmann_L1_xx.append(L1_xx)
        self.boltzmann_L0_xx.append(L0_xx)
        self.boltzmann_L1_yy.append(L1_yy)
        self.boltzmann_L0_yy.append(L0_yy)
        self.boltzmann_L1_xy.append(L1_xy)
        self.boltzmann_L0_xy.append(L0_xy)  
        self.times_boltzmann.append(time.time() - start) 

    def kubo_koef(self, T) -> None:
        start = time.time()
        L1_xx, L1_yy, L1_xy, L1_yx = tokovi_drugic.L_K(self.kymesh, self.vecs, self.energije, self.j1_matrix, self.j_matrix, T, self.mu) + \
                                    tokovi_drugic.L_I_1(self.kymesh, self.vecs, self.energije, self.fs, self.j2_matrix, self.j_matrix, T, self.mu)
        
        res = tokovi_drugic.L_I_1_alternative(self.kymesh, self.vecs, self.energije, self.fs, self.j2_matrix, self.j_matrix, T, self.mu)
        L0_xx, L0_yy, L0_xy, L0_yx = tokovi_drugic.L_11(self.kymesh, self.vecs, self.energije, self.j_matrix, T, self.mu)
        self.kubo_L1_xx.append(L1_xx)
        self.kubo_L0_xx.append(L0_xx)
        self.kubo_L1_yy.append(L1_yy)
        self.kubo_L0_yy.append(L0_yy)
        self.kubo_L1_xy.append(L1_xy)
        self.kubo_L0_xy.append(L0_xy)
        self.kubo_L1_yx.append(L1_yx)
        self.kubo_L0_yx.append(L0_yx)
        self.times_kubo.append(time.time() - start)

    def run(self, Ts):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1)
                self.boltzmann_koef(T)
                self.kubo_koef(T)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

            else:
                self.next_T(T, 2)

    def S_kubo(self):
        return - np.array(self.kubo_L1_xx) / np.array(self.kubo_L0_xx.real),\
                - np.array(self.kubo_L1_yy) / np.array(self.kubo_L0_yy.real),\
                - np.array(self.kubo_L1_xy) / np.array(self.kubo_L0_xy.real),\
                - np.array(self.kubo_L1_yx) / np.array(self.kubo_L0_yx.real)
                
    def S_boltzmann(self):
        return - np.array(self.boltzmann_L1_xx) / np.array(self.boltzmann_L0_xx),\
                - np.array(self.boltzmann_L1_yy) / np.array(self.boltzmann_L0_yy),\
                - np.array(self.boltzmann_L1_xy) / np.array(self.boltzmann_L0_xy)

    def reset(self):
        self.rho = self.rho0
        self.hartree = self.hartree0
        self.fock = self.fock0

    def reset_infty(self):
        self.rho = helpers.Rhoinfty(self.Ny, self.Nx)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)

    def collect(self):
        self.phis = np.array(self.phis)
        self.mus = np.array(self.mus)
        self.errors = np.array(self.errors)
        self.occupations = np.array(self.occupations)
        self.times_rho = np.array(self.times_rho)
        self.times_boltzmann = np.array(self.times_boltzmann)
        self.times_kubo = np.array(self.times_kubo)
        self.Ts = np.array(self.Ts)
        self.betas = np.array(self.betas)
        
        self.boltzmann_L1_xx, self.boltzmann_L0_xx = np.array(self.boltzmann_L1_xx), np.array(self.boltzmann_L0_xx)
        self.boltzmann_L1_yy, self.boltzmann_L0_yy = np.array(self.boltzmann_L1_yy), np.array(self.boltzmann_L0_yy)
        self.boltzmann_L1_xy, self.boltzmann_L0_xy = np.array(self.boltzmann_L1_xy), np.array(self.boltzmann_L0_xy)

        self.kubo_L1_xx, self.kubo_L0_xx = np.array(self.kubo_L1_xx), np.array(self.kubo_L0_xx)
        self.kubo_L1_yy, self.kubo_L0_yy = np.array(self.kubo_L1_yy), np.array(self.kubo_L0_yy)
        self.kubo_L1_xy, self.kubo_L0_xy = np.array(self.kubo_L1_xy), np.array(self.kubo_L0_xy)
        self.kubo_L1_yx, self.kubo_L0_yx = np.array(self.kubo_L1_yx), np.array(self.kubo_L0_yx)

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 100, 2000, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters1 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 10, 10, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters2 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

Ny, Nx = 10,10
mu = 2.84
print('here')
s = TNS(a, b, b2, Ny, Nx, U, V, mu, parameters1, parameters2)
Ny, Nx = s.kymesh.shape

fs = s.fs
vecs = s.vecs
j_matrix = s.j_matrix
j2_matrix = s.j2_matrix
energije = s.energije

barve = np.einsum('ijkl->jkl', np.abs(vecs[:4, :, :, :])**2)

fig, ax = plt.subplots()

col1, col2 = 'firebrick', 'steelblue'
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

# Gamma - X
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + (Ny//2-1)*2*np.pi/b + np.arange(Nx//2)*2*np.pi/a
    E = energije[i, Ny//2, Nx//2:]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, Ny//2, Nx//2:][g:g+2])
        plt.plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# Z - Gamma
E_c, E_v = np.max(energije[:Ny//2, Nx//2]), np.min(energije[:Ny//2, Nx//2])
gap = E_c - E_v
for i in range(6):
    K = (Nx//2-1)*2*np.pi/a + np.arange(Ny//2)*2*np.pi/b
    E = energije[i, :Ny//2, Nx//2]
    for g in range(0,len(K)-1):#)len(K)-1):
        col = np.mean(barve[i, :Ny//2, Nx//2][g:g+2])
        plt.plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

# Z - M
for i in range(6):
    K = np.arange(Nx//2)*2*np.pi/a
    E = energije[i, 0, Nx//2:][::-1]
    for g in range(0,len(K)-1):
        col = np.mean(barve[i, 0, Nx//2:][::-1][g:g+2])
        plt.plot(K[g:g+2], E[g:g+2], color=colorFader(col1, col2, col))

ax.set_xticklabels(['$M$', '$Z$', r'$\Gamma$', '$X$'])
ax.set_xticks([0, (Nx//2 -1)*2*np.pi/a, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a], color='magenta')

ax.set_xlim([0, (Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b + (Nx//2-1)*2*np.pi/a])

plt.axvline((Nx//2 -1)*2*np.pi/a, ls='dashed', color='grey')
plt.axvline((Nx//2 -1)*2*np.pi/a + (Ny//2 - 1)*2*np.pi/b, ls='dashed', color='grey')

plt.show()
