
import numpy as np
import time

#os.chdir('/Users/ana/Desktop/ta2nise5')

import tokovi_drugic
import helpers


#os.chdir('/Users/ana/Desktop/ta2nise5/parameters')
import os 

os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")


U = 2.5 # eV
V = 0.785 # eV
a = 3.51 # A
b = 15.79 # A
b2 = 1.927 # A
mu0 = 2.84
dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 100, 2000, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters1 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 10, 10, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters2 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]



''' create TNS class '''
class TNS:
    def __init__(self, a, b, b2, Ny, Nx, U, V, mu0, parameters1, parameters2, eps0, n_target, faktor=1.):
        self.faktor = faktor
        self.n_target = n_target
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, a, b, faktor=faktor)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.mu = mu0
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)

        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, a, U, V, 1e-10, maxiter=1000, N_epsilon=5)
    
        self.velocities()

        self.rho0 = self.rho
        self.fock0 = self.fock
        self.hartree0 = self.hartree
        self.phi = helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real
        self.Phi_x = []
        self.Phi_y = []
        
        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []

        self.Ts = []
        self.betas = []

        self.RhoB_x = []
        self.RhoB_y = []
        self.RhoK_x = []
        self.RhoK_y = []

        self.Seebeck_Kx = []
        self.Seebeck_Ky = []
        self.Seebeck_Bx = []
        self.Seebeck_By = []

    def velocities(self) -> None:
        # Boltzmann's group velocities using Hellmann-Feynmann
        self.dH_dk = tokovi_drugic.velocity_HF(self.kymesh, self.kxmesh, a, b, tokovi_drugic.kinetic(), self.faktor)
        self.dfock_dk = tokovi_drugic.velocity_fock_HF(self.rho, self.kymesh, self.kxmesh, a, V)
        self.velocity_x = np.einsum('jixy, jlxy, lixy -> ixy', self.vecs.conj(), self.dH_dk[0] + self.dfock_dk[0], self.vecs).real
        self.velocity_y = np.einsum('jixy, jlxy, lixy -> ixy', self.vecs.conj(), self.dH_dk[1] + self.dfock_dk[1], self.vecs).real

        # Kubo's velocities: expectation values of current operator
        self.tok = tokovi_drugic.j_tok(self.kymesh, self.kxmesh, a, b, b2, tokovi_drugic.kinetic(), self.faktor)
        self.tok_fock = tokovi_drugic.j_Fock(self.rho, self.kymesh, self.kxmesh, a, b, b2, V)
        self.tok_x = np.einsum('jixy, jlxy, lkxy -> ikxy', self.vecs.conj(), self.tok[0] + self.tok_fock[0], self.vecs).real
        self.tok_y = np.einsum('jixy, jlxy, lkxy -> ikxy', self.vecs.conj(), self.tok[1] + self.tok_fock[1], self.vecs).real

    def transport_Kubo(self, omegas, Gamma):
        phiKx, phiKy = tokovi_drugic.phi_kubo(self.kymesh, self.kxmesh, self.tok_x, self.tok_y, self.energije, omegas, self.mu, Gamma)
        return np.pi * phiKx, np.pi * phiKy
    
    def transport_Boltzmann(self, omegas, Gamma, faktor):
        phiBx, phiBy = tokovi_drugic.phi_boltzmann(self.kymesh, self.kxmesh, self.velocity_x, self.velocity_y, self.energije, omegas, self.mu, faktor=faktor)
        return phiBx/(2*Gamma), phiBy/(2*Gamma)

    def density_of_states(self, omegas, faktor):
        return helpers.DoS(self.kymesh, self.kxmesh, self.energije, omegas, self.mu, self.velocity_x, self.velocity_y, faktor=faktor)

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        if len(self.mus) > 1:
            mu_candidate = self.mu #self.mus[-1] + (self.mus[-1] - self.mus[-2])/(self.Ts[-1] - self.Ts[-2]) * (T - self.Ts[-1])
        else:
            mu_candidate = self.mu
        rho, energije, fs, vecs, fock, hartree, err, n, mu = helpers.NewMu(self.n_target, self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, mu_candidate, 
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
        #print(1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def run2(self, betas, stops, Gamma, eps, Nomega):
        #hbar = 6.582119569 * 1e-16 # eV s
        #e0 = 1.602176634 * 1e-19 # A s
        kb = 8.6173303 * 1e-5 # eV/K
        a = 3.51 # A
        b = 15.79 # A
        c = 13.42 # A

        predfaktor = 2 * np.pi / (a * b * c * 1e-10) / (25.8 * 1e3)  # 1/(Ohm * m^3)

        for i, beta in enumerate(betas):
            T = 1/beta
            if i not in stops:
                print(f'started evaluating at beta={beta}')
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    fs_save = self.fs
                    vecs_save = self.vecs
                    fock_save = self.fock
                    hartree_save = self.hartree
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                self.next_T(T, 2)
            else:
                print(f'started evaluating at beta={beta}')
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                print(f'occupation error is {np.abs(self.n - self.n_target)}')

                omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
                omegas = np.linspace(-omega_max, omega_max, Nomega)

                self.velocities()

                # Boltzmann coefficients -> Seebeck and resistivity
                K0b_x, K0b_y, K1b_x, K1b_y = tokovi_drugic.K0_boltzmann(self.kymesh, self.kxmesh, self.velocity_x, self.velocity_y, self.energije, self.mu, T)
                sigmaB_x = K0b_x  / (2*Gamma) * predfaktor # 1/(Ohm m)
                sigmaB_y = K0b_y / (2*Gamma) * predfaktor # 1/(Ohm m)
                rhoB_x = 1/sigmaB_x * 1e2 # Ohm cm
                rhoB_y = 1/sigmaB_y * 1e2 # Ohm cm
                seebeck_Bx = - K1b_x/K0b_x/T * kb * 1e6 # muV / K
                seebeck_By = - K1b_y/K0b_y/T * kb * 1e6 # muV / K

                # Kubo coefficients -> Seebeck and resistivity
                phiKx, phiKy = tokovi_drugic.phi_kubo(self.kymesh, self.kxmesh, self.tok_x, self.tok_y, self.energije, omegas, self.mu, Gamma)
                phiKx, phiKy = phiKx * np.pi, phiKy * np.pi
                K0k_x = np.sum(phiKx.real * (-tokovi_drugic.fd_1(omegas, T))) * (omegas[1] - omegas[0])
                K0k_y = np.sum(phiKy.real * (-tokovi_drugic.fd_1(omegas, T))) * (omegas[1] - omegas[0])
                K1k_x = np.sum(omegas * phiKx.real * (-tokovi_drugic.fd_1(omegas, T))) * (omegas[1] - omegas[0])
                K1k_y = np.sum(omegas * phiKy.real * (-tokovi_drugic.fd_1(omegas, T))) * (omegas[1] - omegas[0])
                sigmaK_x = K0k_x * predfaktor # 1/(Ohm m)
                sigmaK_y = K0k_y * predfaktor # 1/(Ohm m)
                rhoK_x = 1/sigmaK_x * 1e2 # Ohm cm
                rhoK_y = 1/sigmaK_y * 1e2 # Ohm cm
                seebeck_Kx = -K1k_x/K0k_x/T * kb * 1e6 # muV / K
                seebeck_Ky = -K1k_y/K0k_y/T * kb * 1e6 # muV / K

                self.RhoB_x.append(rhoB_x) # Ohm cm
                self.RhoB_y.append(rhoB_y)
                self.RhoK_x.append(rhoK_x)
                self.RhoK_y.append(rhoK_y)

                self.Seebeck_Kx.append(seebeck_Kx) # muV / K
                self.Seebeck_Ky.append(seebeck_Ky)
                self.Seebeck_Bx.append(seebeck_Bx)
                self.Seebeck_By.append(seebeck_By)

                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.fs = fs_save
                    self.vecs = vecs_save
                    self.fock = fock_save
                    self.hartree = hartree_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

    def reset(self, mu0):
        self.rho = self.rho0
        self.hartree = self.hartree0
        self.fock = self.fock0
        self.mu = mu0

    def reset_infty(self):
        self.rho = helpers.Rhoinfty(self.Ny, self.Nx)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        
        _, energije, fs, vecs, _, _, _, _ = helpers.Rho_next(self.kxmesh, self.kymesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, a, b, b2, U, V, 0, self.mu, 50, 0.5, 1e-10, eps0=0.0, N_epsilon=5)
        self.energije = energije
        self.fs = fs
        self.vecs = vecs


