import numpy as np
import sys
import time
from shutil import copyfile
pop_size = int(sys.argv[1])
num_var = int(sys.argv[2])
np.random.seed(int(time.time()))
pop = np.random.uniform(-5, 5, size=(pop_size,num_var))
pop = np.around(pop, decimals=2)
with open('pop.txt', 'w') as file:
    for i in range(pop_size):
        for j in range(num_var):
            file.write(str(pop[i, j])+' ')
        file.write('\n')
copyfile('pop.txt', 'g3pcx_cpp/pop.txt')
