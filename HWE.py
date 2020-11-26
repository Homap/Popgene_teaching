#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

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

plt.figure()
plt.ylabel("Genotype frequency")
plt.xlabel("Frequency of A")
plt.plot(np.arange(0, 1, 0.01), AA_l, "-b", label="Freq-AA")
plt.plot(np.arange(0, 1, 0.01), aa_l, "-r", label="Freq-aa")
plt.plot(np.arange(0, 1, 0.01), Aa_l, "-g", label="Freq-Aa")
plt.legend(loc="upper left")
plt.show()

