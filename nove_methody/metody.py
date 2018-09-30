#!/usr/bin/env python

import numpy as np
from scipy.special import erf


# Constants used to convert to atomic units
angstrom = 1.889726133921252
electronvolt = 0.036749325919680595

# Nicer format for the arrays
np.set_printoptions(precision=5, linewidth=200, suppress=True)


class System(object):
    def __init__(self, atoms, bonds):
        if not isinstance(atoms, np.ndarray):
            raise TypeError('atoms argument must be a numpy array.')
        if not isinstance(bonds, np.ndarray):
            raise TypeError('bonds argument must be a numpy array.')
        self.atoms = atoms
        self.bonds = bonds
        self.natom = len(atoms)
        self.nbond = len(bonds)

    @classmethod
    def from_file(cls, fn_inp):
        '''Alternative constructor that loads a system from a file

           See water.inp for an example input file.
        '''
        # Load all non-empty lines
        lines = []
        with open(fn_inp) as f:
            for line in f:
                line = line[:line.find('#')].strip()
                if len(line) > 0:
                    lines.append(line)

        # Load title
        title = lines.pop(0)

        # Load atom part
        natom = int(lines.pop(0))
        atoms = np.zeros(natom, dtype=[
            ('symbol','a2'), ('x','f8') , ('y','f8'), ('z','f8'), ('chi','f8'),
            ('eta','f8'), ('width','f8'), ('q0','f8')
        ])
        for i in range(natom):
            words = lines.pop(0).split()
            atoms[i] = (
                words[0],                     # symbol
                float(words[1])*angstrom,     # x
                float(words[2])*angstrom,     # y
                float(words[3])*angstrom,     # z
                float(words[4])*electronvolt, # chi
                float(words[5])*electronvolt, # eta
                float(words[6])*angstrom,     # width
                float(words[7]),              # q0
            )
        # Load bond part
        nbond = int(lines.pop(0))
        bonds = np.zeros(nbond, dtype=[('i','i4'), ('j','i4'), ('kappa','f8')])
        for i in range(nbond):
            words = lines.pop(0).split()
            bonds[i] = (
                int(words[0]),                 # i
                int(words[1]),                 # j
                float(words[2])*electronvolt,  # kappa
            )

        return System(atoms, bonds)


def setup_eem(s, A_eem, B_eem):
    # Fill in the hardness matrix and the chemical potentials
    for i in range(s.natom):
        for j in range(s.natom):
            if i == j:
                # check if the atomic hardness is large enough to guarantee a
                # positive definnite hardness matrix.
                eta_min = 1.0/(np.sqrt(np.pi)*s.atoms['width'][i])
                eta = s.atoms['eta'][i]
                if eta < eta_min:
                    raise ValueError('The hardness parameter of atom %i should at least be %.5f eV.' % (i, eta_min/electronvolt))
                # fill in parameters
                A_eem[i,i] = eta
                B_eem[i] = -s.atoms['chi'][i]
            else:
                # fill in the electrostatics
                d = np.sqrt(
                    (s.atoms['x'][i] - s.atoms['x'][j])**2 +
                    (s.atoms['y'][i] - s.atoms['y'][j])**2 +
                    (s.atoms['z'][i] - s.atoms['z'][j])**2
                )
                d0 = np.sqrt(2*s.atoms['width'][i]**2 + 2*s.atoms['width'][j]**2)
                if d0 == 0:
                    A_eem[i,j] = 1.0/d
                else:
                    A_eem[i,j] = erf(d/d0)/d

    evals = np.linalg.eigvalsh(A_eem)
    assert (evals > 0).all()



def setup_pert(s, pert):
    pert[:,0] = s.atoms['x']
    pert[:,1] = s.atoms['y']
    pert[:,2] = s.atoms['z']


def solve_sqe(s):
    if s.nbond == 0:
        return s.atoms['q0'], np.zeros((3,3))
    else:
        # Setup transfer matrix T
        T = np.zeros((s.nbond, s.natom))
        for ibond in range(s.nbond):
            print(ibond, s.bonds['i'][ibond], s.bonds['j'][ibond])
            T[ibond,s.bonds['i'][ibond]] = +1
            T[ibond,s.bonds['j'][ibond]] = -1
        print(T)
        from sys import exit ; exit()
        # Setup EEM part
        A_eem = np.zeros((s.natom, s.natom))
        B_eem = np.zeros(s.natom)
        setup_eem(s, A_eem, B_eem)

        # Add contribution due to reference charges (only electrostatic interactions)
        B_eem -= np.dot(A_eem, s.atoms['q0'])
        B_eem += s.atoms['eta']*s.atoms['q0']

        # Convert to split-charge basis
        A_sqe = np.dot(T, np.dot(A_eem, T.T))
        B_sqe = np.dot(T, B_eem)

        # Add bond hardness terms
        for ibond in range(s.nbond):
            A_sqe[ibond,ibond] += s.bonds['kappa'][ibond]

        # Solve the system
        A_inv = np.linalg.inv(A_sqe)
        solution = np.dot(A_inv, B_sqe)

        # Compute the dipole polarizability tensor
        pert = np.zeros((s.natom, 3))
        setup_pert(s, pert)
        pert_sqe = np.dot(T, pert)
        ptens = np.dot(pert_sqe.T, np.dot(A_inv, pert_sqe))

        # Get the charges
        charges = np.dot(solution, T) + s.atoms['q0']
        Tp = np.linalg.pinv(T)
        return charges, ptens


def solve_acks2(s):
    A_acks2 = np.zeros((2*s.natom+2, 2*s.natom+2))
    B_acks2 = np.zeros(2*s.natom+2)

    # Setup EEM part
    setup_eem(s, A_acks2[:s.natom,:s.natom], B_acks2[:s.natom])

    # Add contribution from reference charges
    B_acks2[:s.natom] += s.atoms['eta']*s.atoms['q0']

    # Fill in the constraints
    A_acks2[s.natom,:s.natom] = 1
    A_acks2[:s.natom,s.natom] = 1
    A_acks2[-1,s.natom+1:2*s.natom+1] = 1
    A_acks2[s.natom+1:2*s.natom+1,-1] = 1
    B_acks2[s.natom] = s.atoms['q0'].sum()
    B_acks2[s.natom+1:2*s.natom+1] = s.atoms['q0']

    # Fill in off-diagonal identity matrix blocks
    for i in range(s.natom):
        A_acks2[i,s.natom+1+i] = 1.0
        A_acks2[s.natom+1+i,i] = 1.0

    # Add the bond hardness terms as off-diagonal bond-softness parameters
    for ibond in range(s.nbond):
        i = s.natom+1+s.bonds['i'][ibond]
        j = s.natom+1+s.bonds['j'][ibond]
        bsoft = 1/s.bonds['kappa'][ibond]
        A_acks2[i,j] += bsoft
        A_acks2[j,i] += bsoft
        A_acks2[i,i] -= bsoft
        A_acks2[j,j] -= bsoft

    # Solve the equations to obtain the ground state charge distribution
    A_inv = np.linalg.inv(A_acks2)
    solution = np.dot(A_inv, B_acks2)

    # Compute the dipole polarizability tensor
    pert = np.zeros((2*s.natom+2, 3))
    setup_pert(s, pert[:s.natom])
    ptens = np.dot(pert.T, np.dot(A_inv, pert))

    # Get the charges
    charges = solution[:s.natom]
    return charges, ptens


def main(fn_inp):
    print(' .. Loading system')
    s = System.from_file(fn_inp)
    print(' .. Computing SQE results')
    charges_sqe, ptens_sqe = solve_sqe(s)
    print(' .. Computing ACKS2 results')
    charges_acks2, ptens_acks2 = solve_acks2(s)
    print('Charges')
    print('-------')
    print('Index  Symbol       q_SQE     q_ACKS2')
    for i in range(s.natom):
        print('%5i      %2s  %10.5f  %10.5f' % (i, s.atoms['symbol'][i], charges_sqe[i], charges_acks2[i]))
    error_charges = abs(charges_sqe - charges_acks2).max()/abs(charges_sqe).max()
    print('Relative maximum deviation between SQE and ACKS2 charges: %10.3e' % error_charges)
    print()
    print('Dipole polarizability tensor')
    print('----------------------------')
    print('(printed in atomic units)')
    print('SQE')
    print(ptens_sqe)
    print('ACKS2')
    print(ptens_acks2)
    error_ptens = abs(ptens_sqe - ptens_acks2).max()/abs(ptens_sqe).max()
    print('Relative maximum deviation between SQE and ACKS2 tensors: %10.3e' % error_ptens)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 1:
        fn_inp = args[0]
        main(fn_inp)
    else:
        print('Expecting one input file as argument')
        sys.exit(-1)
