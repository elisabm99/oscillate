# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:40:33 2015

@author: elisa
# """
# gl = globals().copy()
# for var in gl:
#        if var[0] == '_': continue
#        if 'func' in str(globals()[var]): continue
#        if 'module' in str(globals()[var]): continue
#        del globals()[var]
 
import numpy as np
import scipy as sp
from scipy import integrate
import itertools as it
import sys
import os



#pi = -0.5 * sp.pi
               # number of quanta
M = 5                  # number of oscillators
lamb = 0.1        # coupling constant
T = float(sys.argv[6])            # integration time
deltat = float(sys.argv[7])            # integrator returns data every deltat
dt =0.0001           # time step for integrate_gao()
energy_levels = (1,2,3,4,5)

# Parameters for initial state
# 0 -> number eigenstate, 
# 1 -> coherent state, 
# 2 -> load eigenvector from file
initial_state_mode = int(sys.argv[1])


t1=0
v_i = tuple((int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])))

E = sum(np.multiply(v_i, energy_levels))
v_i_str = str(v_i).replace('(', '_').replace(')', '').replace(', ', '_')
v_i_lamb_str = v_i_str + '_lamb=%.2f'%lamb


N = sum(v_i)
print_mode=0
b = N+1

data_dir = './tests/test'+ v_i_str+'/data/'
if not os.path.exists('./tests/test'+ v_i_str):
	os.mkdir('./tests/test'+ v_i_str)
	os.mkdir('./tests/test'+ v_i_str+'/data/')

plot_dir = './tests/test'+ v_i_str+'/plots/'
if not os.path.exists(plot_dir):
	os.mkdir(plot_dir)


hilbert_space_str = '_M=%d_N=%d_E=%d' % (M, N, E)

filename_psi = data_dir + 'psi'+hilbert_space_str+v_i_lamb_str+'_t=%.3f.npz' 
filename_data = data_dir + 'data'+hilbert_space_str+v_i_lamb_str+'_t=%.3f.npz' 
filename_H = './H'+hilbert_space_str+'.npz'
filename_H_dic = './Hd'+hilbert_space_str+'.npz'
filename_Mx0 = data_dir + 'Mx0'+hilbert_space_str+'.npz' 
filename_basis_vectors = './basis_vectors'+hilbert_space_str+'.npz' 
filename_dict1 = './dict1'+hilbert_space_str+'.npz' 
filename_eigvals = 'eigvals'+hilbert_space_str+'.npz'
filename_eigvects = 'eigvects'+hilbert_space_str+'.npz'

np.set_printoptions(precision=3, linewidth=150)

############################################################################ 
############################################################################ 
############################################################################ 
def check_parameters():
    
    # check length of eigenvector_initial and energies
    if len(energy_levels) != M:
        sys.exit("Error. Wrong number of energy levels.") 
    if len(v_i) != M:
        sys.exit("Error. Wrong length of eigenvector_initial.") 

    # check number of quanta
    if sum(v_i) != N:
        sys.exit("Error. Wrong number of quanta in eigenvector_initial.")
  
def initialize_couplings(lamb, M):  
    # Initialize coupling constants
    # Rules hard coded in this initialization
    # - Conservation of energy with the energy of a level j being w_j = j*w_1
    # - *NOT* force lambda to be zero if i=j or k=l           
    lambdas = {}
    for i in range(M):
        for j in range(i, M):
            for k in range(M):
                for l in range(k, M):
                    if (i,j) == (k,l):
                        continue
                    if i + j == k + l:
                        if (k, l, i, j) in lambdas.keys():
                            continue
                        lambdas[(i, j, k, l)] = lamb
    return lambdas

def generate_basis_vectors():
    # Generates basis of restricted Hilbert space given the intial state vector    

    basis_vectors = []
    # dict1:  keys = basis vectors in base 10, values = index of the corresponding amplitude in psi
    dict1 = {}
    i = 0
    # it.product(...) calculates all the tuples of M elements with values from 0 to N
    for indices1 in it.product(np.arange(N + 1), repeat=(M-1)):
        # select the tuples that have number of quanta = N     
        last_n = N - sum(indices1)
        if last_n < 0:
            continue
        else:
            indices = indices1+(last_n,)
            
            # select the tuples that have energy = E
            e = calculate_energy_per_oscillator(indices)
            if sum(e) == E:
                basis_vectors.append(indices)
                dict1[convert_to_base_10(indices, b)] = i
                i += 1
    return basis_vectors, dict1

def convert_to_base_10(n_occ, base):
# Takes a vector of  occupation numbers (in base b=N+1) and converts
# it to a number in base 10

    x = 0    
    p = 0
    for n in n_occ:
        x += n * pow(base, p)
        p += 1
    return int(x)

def get_occupation_numbers(x, base, M):

    n = np.empty(M)    
    for p in reversed(range(M)):
        #print p, math.pow(b, p), x
        n[p] = int(x / pow(base, p))
        x = x - n[p] * pow(base, p)
        
    return n   
    
def calculate_energy_per_oscillator(n_occ):
  # Calculates the energy in each oscillator given the occupation numbers  
    return np.multiply(n_occ, energy_levels)
    
def calculate_probability(psi):    
    return np.real(psi.dot(np.conj(psi)))
    
def calculate_occupation_numbers(psi, basis_vectors):
    
    bv = np.array(basis_vectors)
    psi_squared = np.real(np.multiply(psi, np.conj(psi)))
    prob = sum(psi_squared)
    n = np.empty(M)
    for m in range(M):
        v = bv[:,m]
        n[m] = sum(np.multiply(v, psi_squared))/prob
    return n
    
def initialize_wave_function(dict1, basis_vectors):
    
    psi_i = np.zeros(len(basis_vectors), dtype=complex)
    v10_i = convert_to_base_10(v_i, b)
    index = dict1[v10_i]
    psi_i[index] = 1. + 0.0j    
    return psi_i        
  
def calculate_hamiltonian_dic(lambdas, basis_vectors, dict1):
    
    # row index = final state, column index = initial state
    keys = []
    values = []
    element_indeces = []
    
    for (i,j,k,l), lamb in lambdas.items():
        for vector10, index in dict1.items():
                        
            vector = basis_vectors[index]
            
            if (vector[i] == 0 or vector[j] == 0):
                continue

            if k == l:
                coefficient = lamb * np.sqrt(
                     vector[i] * vector[j] * (vector[k] + 1) * (vector[k] + 2))         
                     
            else:
                coefficient = lamb * np.sqrt(
                     vector[i] * vector[j] * (vector[k] + 1) * (vector[l] + 1))         
            
            vector_final10 = vector10 - pow(b, i) - pow(b, j) + pow(b, k) + pow(b, l)

            i_row = dict1[vector_final10]
                    
            key = convert_to_base_10([i_row, index], len(basis_vectors)+1)
            if key in keys:
                values[keys.index(key)] += coefficient
            else:
                keys.append(key)
                values.append(coefficient)
                element_indeces.append([i_row, index])
                
    return element_indeces, values

def f_dic(t, psi):
    
    psi1 = np.zeros(len(psi), dtype=complex)
    for [alpha, beta], coeff in zip(element_indeces, values):
        psi1[alpha] += -1j * coeff * psi[beta]
        psi1[beta] += -1j * coeff * psi[alpha]
    return psi1


############################################################################ 
############################################################################ 
############################################################################   
if __name__ == "__main__":       
    
    # Consistency check on parameters
    check_parameters()
    counter = 0

    print("\n\n\n\n\n\n\n###############################################################") 
    print ("\nParameters:\nN = %d \nM = %d \nT = %f \ndeltat = %f" %(N, M, T, deltat))
    
    print ('\nInitial state: number eigenstate. v_i = ', v_i)
        
    print ('\nTotal energy:', E )


    # Create basis for Hilbert space (restrict to states that are accessible
    # given the initial state vector)
    if os.path.isfile(filename_basis_vectors) and os.path.isfile(filename_dict1):
        basis_vectors = np.load(filename_basis_vectors)['arr_0']
        dict1 = np.load(filename_dict1, allow_pickle=True)['arr_0'].item()
    else:
        basis_vectors, dict1 = generate_basis_vectors()               
        dict1 = {}
        for i in range(len(basis_vectors)):
            dict1[convert_to_base_10(basis_vectors[i],b)] = i
            i += 1
    
        np.savez(filename_basis_vectors, basis_vectors)
        np.savez(filename_dict1, dict1)    
    
       
    print ('\nNumber of basis vectors compatible with N and E:', len(basis_vectors))
    if print_mode == 1:
        print ('\nBasis vectors:\n%r' % basis_vectors)
        print ('\nUseful dictionary:\n', dict1)

    # Initialize couplings 
    lambdas = initialize_couplings(lamb, M)
    print ('\nCouplings:\n', lambdas)

    # Calculate Hamiltonian H
    if os.path.isfile(filename_H_dic):
        element_indeces = np.load(filename_H_dic)['elements']
        values = np.load(filename_H_dic)['values']
    else:
        element_indeces, values = calculate_hamiltonian_dic(lambdas, basis_vectors, dict1)
        np.savez(filename_H_dic, elements=element_indeces, values=values)
    
    if print_mode == 1:
        print ('\nHamiltonian dictionary:\n', zip(element_indeces, values) )
    
    # Initialize wave function
    psi_i = initialize_wave_function(dict1, basis_vectors)

        
    if print_mode == 1:
        print ('\nInitial wave function:\n', psi_i)
    
    n0 = calculate_occupation_numbers(psi_i, basis_vectors)
    e0 = calculate_energy_per_oscillator(n0)
    np.savez(filename_psi % t1, psi=psi_i)         
    np.savez(filename_data % t1, n=n0, e=e0)

    
    
    n_steps = int(round((T - t1)/deltat)) # number of steps not counting t=0
    print ('\nNumber of integration steps: %d' %n_steps)
    
        
    # Call the ODE solver
    print ("\nCalling ODE solver")
    psi = sp.integrate.ode(f_dic).set_integrator(
                'zvode', method='bdf', with_jacobian=False)
    psi.set_initial_value(psi_i)
    
    n = 1
    while psi.successful() and n < n_steps + 1:
     
        psi.integrate(psi.t + deltat)
        
        np.savez(filename_psi % (psi.t + t1), psi=psi.y)
        
        nocc = calculate_occupation_numbers(psi.y, basis_vectors)
        e = calculate_energy_per_oscillator(nocc)        
        np.savez(filename_data % (n*deltat + t1), n=nocc, e=e)

        n += 1

    np.savez(filename_psi % (psi.t), psi=psi.y)
    print ('total probability at time %f: %f' %(psi.t, calculate_probability(psi.y)))

    




    
     
