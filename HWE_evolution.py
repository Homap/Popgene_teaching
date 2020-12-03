#!/usr/bin/python
import numpy as np
import random
import matplotlib.pyplot as plt

# Hardy-Weinberg simulation written by Homa Papoli
# Translated to Python from R from the code:
# https://gist.github.com/cjbayesian/468725f4bd7d6f3f027d#file-hw_sim-r

# Define a function for crossing to create a offspring
def cross(parents):
	offspring = ['d', 'd']
	offspring[0] = random.sample(list(parents[0,:]),1)[0]
	offspring[1] = random.sample(list(parents[1,:]),1)[0]
	return(offspring)

# Define a function to find two random numbers for random mating
def random_mating(pop):
	tmp_pop = pop
	for n in range(0, N):
		parents = random.sample([i for i in range(0, N)], 2) 
		tmp_pop[n,:] = cross(pop[parents,:])
	pop = tmp_pop

# Two allele
genotypes = ['A','a']

p = 0.5 # Frequency of allele A
q = 1-p # Frequency of allele a
N = 200 # Population size
a_freq = [p, q] # list of allele frequencies

# Random sampling with replacement
g_list = random.choices(genotypes, weights = [0.5, 0.5], k = 2 * N)
# Turn the 1d array into a 2d array to represent genotypes
pop = np.array(g_list).reshape(-1, 2) # 20 diploids


I = 100 # number of iterations
num_generations = 1 # number of generations
g_freq = np.zeros((I, 3)) # Define an array, rows are the number of simulations and columns are the number of genotypes
p_vec = np.zeros((I)) # Define an array with length being equal to the number of simulations

for i in range(0, I):
	p = np.random.uniform(low = 0.0, high = 1.0, size = 1)
	q = 1 - p
	a_freq = [p, q]
	
	pop[:,0] = random.choices(genotypes, weights=a_freq, k = N)
	pop[:,1] = random.choices(genotypes, weights=a_freq, k = N)

	for g in range(0, num_generations):
		random_mating(pop)

	f_aa = 0 
	f_Aa = 0
	f_AA = 0

	for n in range(0, N):
		if "".join(pop[n,:]) == 'AA':
			f_AA = f_AA+1
		if "".join(pop[n,:]) == 'Aa' or "".join(pop[n,:]) == 'aA':
			f_Aa = f_Aa+1
		if "".join(pop[n,:]) == 'aa':
			f_aa = f_aa+1
	
	f_aa = f_aa/N
	f_Aa = f_Aa/N
	f_AA = f_AA/N
	
	g_freq[i,:] = [f_AA,f_Aa,f_aa]
	p_vec[i] = p	


## Plot the sims
plt.figure()
plt.ylabel("Genotype frequency")
plt.xlabel("Frequency of A")
plt.scatter(p_vec,g_freq[:,0], c = "blue", lw = 0, alpha = 0.5)
plt.scatter(p_vec,g_freq[:,1], c = "red", lw = 0, alpha = 0.5)
plt.scatter(p_vec,g_freq[:,2], c = "green", lw= 0, alpha = 0.5)

# Plot the theoretical
# HW proportions
AA_l = []
Aa_l = []
aa_l = []

for A in np.arange(0, 1, 0.01):
	a = 1 - A
	AA = A**2
	Aa = A * a * 2
	aa = a**2
	AA_l.append(AA)
	Aa_l.append(Aa)
	aa_l.append(aa)

plt.plot(np.arange(0, 1, 0.01), AA_l, "-b", label="Freq-AA")
plt.plot(np.arange(0, 1, 0.01), Aa_l, "-r", label="Freq-Aa")
plt.plot(np.arange(0, 1, 0.01), aa_l, "-g", label="Freq-aa")
plt.legend(loc="upper center")
plt.show()
