# !/usr/bin/python
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import random
import os


# --------------------------------------------- Basic Neural Network Class ---------------------------------------------

class Network(object):

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer
        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer

    @staticmethod
    def sigmoid(x):
        x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    def sample_er(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def calculate_rmse(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def sample_ad(self, actualout):
        error = np.subtract(self.out, actualout)
        mod_error = np.sum(np.abs(error)) / self.topology[2]
        return mod_error

    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def backward_pass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1_size = self.topology[0] * self.topology[1]
        w_layer2_size = self.topology[1] * self.topology[2]
        w_layer1 = w[0:w_layer1_size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        w_layer2 = w[w_layer1_size: w_layer1_size + w_layer2_size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1_size + w_layer2_size :w_layer1_size + w_layer2_size + self.topology[1]]
        self.B2 = w[w_layer1_size + w_layer2_size + self.topology[1] :w_layer1_size + w_layer2_size + self.topology[1] + self.topology[2]]

    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    @staticmethod
    def scale_data(data, maxout=1, minout=0, maxin=1, minin=0):
        attribute = data[:]
        attribute = minout + (attribute - minin)*((maxout - minout)/(maxin - minin))
        return attribute

    @staticmethod
    def denormalize(data, indices, maxval, minval):
        for i in range(len(indices)):
            index = indices[i]
            attribute = data[:, index]
            attribute = Network.scale_data(attribute, maxout=maxval[i], minout=minval[i], maxin=1, minin=0)
            data[:, index] = attribute
        return data

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability

    def generate_output(self, data, w):  # BP with SGD (Stocastic BP)
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))  # temp hold input
        fx = np.zeros((size,self.topology[2]))
        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.forward_pass(Input)
            fx[i] = self.out
        return fx

    def evaluate_fitness(self, w):  # BP with SGD (Stocastic BP
        data = self.train_data
        y = data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        fx = self.generate_output(data, w)
        return self.calculate_rmse(fx, y)

class G3PCX(object):
    def __init__(self, population_size, num_variables, max_limits, min_limits, max_evals=500000):
        self.initialize_parameters()
        self.sp_size = self.children + self.family
        self.population = np.random.uniform(min_limits[0], max_limits[0], size=(population_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(population_size)]
        self.sub_pop = np.random.uniform(min_limits[0], max_limits[0], size=(self.sp_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(NPSize)]
        self.fitness = np.zeros(population_size)
        self.sp_fit  = np.zeros(self.sp_size)
        self.best_index = 0
        self.best_fit = 0
        self.worst_index = 0
        self.worst_fit = 0
        self.rand_parents =  self.num_parents
        self.temp_index =  np.arange(0, population_size)
        self.rank =  np.arange(0, population_size)
        self.list = np.arange(0, self.sp_size)
        self.parents = np.arange(0, population_size)
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_evals = 0
        self.max_evals = max_evals
        self.problem = 1

    def fitness_function(self, x):    #  function  (can be any other function, model or even a neural network)
		fit = 0.0
		if self.problem == 1: # rosenbrock
			for j in range(x.size -1):
				fit += (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))
		elif self.problem ==2:  # ellipsoidal - sphere function
			for j in range(x.size):
				fit = fit + ((j+1)*(x[j]*x[j]))
		return fit # note we will maximize fitness, hence minimize error

    def initialize_parameters(self):
        self.epsilon = 1e-40  # convergence
        self.sigma_eta = 0.1
        self.sigma_zeta = 0.1
        self.children = 2
        self.num_parents = 3
        self.family = 2

    def rand_normal(self, mean, stddev):
        if (not G3PCX.n2_cached):
            #choose a point x,y in the unit circle uniformly at random
            x = np.random.uniform(-1,1,1)
            y = np.random.uniform(-1,1,1)
            r = x*x + y*y
            while (r == 0 or r > 1):
                x = np.random.uniform(-1,1,1)
                y = np.random.uniform(-1,1,1)
                r = x*x + y*y
            # Apply Box-Muller transform on x, y
            d = np.sqrt(-2.0*np.log(r)/r)
            n1 = x*d
            G3PCX.n2 = y*d
            # scale and translate to get desired mean and standard deviation
            result = n1*stddev + mean
            G3PCX.n2_cached = True
            return result
        else:
            G3PCX.n2_cached = False
            return G3PCX.n2*stddev + mean

    def evaluate(self):
        self.fitness[0] = self.fitness_function(self.population[0,:])
        self.best_fit = self.fitness[0]
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i,:])
            if (self.best_fit> self.fitness[i]):
                self.best_fit =  self.fitness[i]
                self.best_index = i
        self.num_evals += 1

    # calculates the magnitude of a vector
    def mod(self, List):
        sum = 0
        for i in range(self.num_variables):
            sum += (List[i] * List[i] )
        return np.sqrt(sum)

    def parent_centric_xover(self, current):
        centroid = np.zeros(self.num_variables)
        tempar1 = np.zeros(self.num_variables)
        tempar2 = np.zeros(self.num_variables)
        temp_rand = np.zeros(self.num_variables)
        d = np.zeros(self.num_variables)
        D = np.zeros(self.num_parents)
        temp1, temp2, temp3 = (0,0,0)
        diff = np.zeros((self.num_parents, self.num_variables))
        for i in range(self.num_variables):
            for u in range(self.num_parents):
                centroid[i]  = centroid[i] +  self.population[self.temp_index[u],i]
        centroid   = centroid / self.num_parents
        # calculate the distace (d) from centroid to the index parent self.temp_index[0]
        # also distance (diff) between index and other parents are computed
        for j in range(1, self.num_parents):
            for i in range(self.num_variables):
                if j == 1:
                    d[i]= centroid[i]  - self.population[self.temp_index[0],i]
                diff[j, i] = self.population[self.temp_index[j], i] - self.population[self.temp_index[0],i]
            if (self.mod(diff[j,:]) < self.epsilon):
                print 'Points are very close to each other. Quitting this run'
                return 0
        dist = self.mod(d)
        if (dist < self.epsilon):
            print " Error -  points are very close to each other. Quitting this run   "
            return 0
        # orthogonal directions are computed
        for j in range(1, self.num_parents):
            temp1 = self.inner(diff[j,:] , d )
            if ((self.mod(diff[j,:]) * dist) == 0):
                print "Division by zero"
                temp2 = temp1 / (1)
            else:
                temp2 = temp1 / (self.mod(diff[j,:]) * dist)
            temp3 = 1.0 - np.power(temp2, 2)
            D[j] = self.mod(diff[j]) * np.sqrt(np.abs(temp3))
        D_not = 0.0
        for i in range(1, self.num_parents):
            D_not += D[i]
        D_not /= (self.num_parents - 1) # this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector
        G3PCX.n2 = 0.0
        G3PCX.n2_cached = False
        for i in range(self.num_variables):
            tempar1[i] = self.rand_normal(0,  self.sigma_eta * D_not) #rand_normal(0, D_not * sigma_eta);
            tempar2[i] = tempar1[i]
        if(np.power(dist, 2) == 0):
            print " division by zero: part 2"
            tempar2  = tempar1
        else:
            tempar2  = tempar1  - (    np.multiply(self.inner(tempar1, d) , d )  ) / np.power(dist, 2.0)
        tempar1 = tempar2
        self.sub_pop[current,:] = self.population[self.temp_index[0],:] + tempar1
        rand_var = self.rand_normal(0, self.sigma_zeta)
        for j in range(self.num_variables):
            temp_rand[j] =  rand_var
        self.sub_pop[current,:] += np.multiply(temp_rand ,  d )
        self.sp_fit[current] = self.fitness_function(self.sub_pop[current,:])
        self.num_evals += 1
        return 1

    def inner(self, ind1, ind2):
        sum = 0.0
        for i in range(self.num_variables):
            sum += (ind1[i] * ind2[i])
        return  sum

    def sort_population(self):
        dbest = 99
        for i in range(self.children + self.family):
            self.list[i] = i
        for i in range(self.children + self.family - 1):
            dbest = self.sp_fit[self.list[i]]
            for j in range(i + 1, self.children + self.family):
                if(self.sp_fit[self.list[j]]  < dbest):
                    dbest = self.sp_fit[self.list[j]]
                    temp = self.list[j]
                    self.list[j] = self.list[i]
                    self.list[i] = temp

    def replace_parents(self): #here the best (1 or 2) individuals replace the family of parents
        for j in range(self.family):
            self.population[ self.parents[j],:]  =  self.sub_pop[ self.list[j],:] # Update population with new species
            fx = self.fitness_function(self.population[ self.parents[j],:])
            self.fitness[self.parents[j]]   =  fx
            self.num_evals += 1

    def family_members(self): #//here a random family (1 or 2) of parents is created who would be replaced by good individuals
        swp = 0
        for i in range(self.population_size):
            self.parents[i] = i
        for i in range(self.family):
            randomIndex = random.randint(0, self.population_size - 1) + i # Get random index in population
            if randomIndex > (self.population_size-1):
                randomIndex = self.population_size-1
            swp = self.parents[randomIndex]
            self.parents[randomIndex] = self.parents[i]
            self.parents[i] = swp

    def find_parents(self): #here the parents to be replaced are added to the temporary subpopulation to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
        self.family_members()
        for j in range(self.family):
            self.sub_pop[self.children + j, :] = self.population[self.parents[j],:]
            fx = self.fitness_function(self.sub_pop[self.children + j, :])
            self.sp_fit[self.children + j]  = fx
            self.num_evals += 1

    def random_parents(self ):
        for i in range(self.population_size):
            self.temp_index[i] = i
        swp=self.temp_index[0]
        self.temp_index[0]=self.temp_index[self.best_index]
        self.temp_index[self.best_index]  = swp
        # best is always included as a parent and is the index parent
        # this can be changed for solving a generic problem
        for i in range(1, self.rand_parents):
            index= np.random.randint(self.population_size)+i
            if index > (self.population_size-1):
                index = self.population_size-1
            swp=self.temp_index[index]
            self.temp_index[index]=self.temp_index[i]
            self.temp_index[i]=swp

    def evolve(self, outfile):
        tempfit = 0
        prevfitness = 99
        self.evaluate()
        tempfit= self.fitness[self.best_index]
        while(self.num_evals < self.max_evals):
            tempfit = self.best_fit
            self.random_parents()
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break
            if tag == 0:
                break
            self.find_parents()
            self.sort_population()
            self.replace_parents()
            self.best_index = 0
            tempfit = self.fitness[0]
            for x in range(1, self.population_size):
                if(self.fitness[x] < tempfit):
                    self.best_index = x
                    tempfit  =  self.fitness[x]
            if self.num_evals % 197 == 0:
                print self.fitness[self.best_index]
                print self.num_evals, 'num of evals\n\n\n'
            np.savetxt(outfile, [ self.num_evals, self.best_index, self.best_fit], fmt='%1.5f', newline="\n")
        print self.sub_pop, '  sub_pop'
        print self.population[self.best_index], ' best sol                                         '
        print self.fitness[self.best_index], ' fitness'


class MCMC(G3PCX):
    def __init__(self, num_samples, population_size, topology, train_data, test_data, directory, problem_type='regression', max_limit=(-5), min_limit=5):
        self.num_samples = num_samples
        self.topology = topology
        self.train_data = train_data
        self.test_data = test_data
        self.problem_type = problem_type
        self.directory = directory
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.neural_network = Network(topology, train_data, test_data)
        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
        self.initialize_sampling_parameters()
        self.create_directory(directory)
        G3PCX.__init__(self, population_size, self.w_size, self.max_limits, self.min_limits)

    def fitness_function(self, x):
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

    def initialize_sampling_parameters(self):
        self.eta_stepsize = 0.02
        self.sigma_squared = 25
        self.nu_1 = 0
        self.nu_2 = 0
        self.start = time.time()

    @staticmethod
    def convert_time(secs):
        if secs >= 60:
            mins = str(int(secs/60))
            secs = str(int(secs%60))
        else:
            secs = str(int(secs))
            mins = str(00)
        if len(mins) == 1:
            mins = '0'+mins
        if len(secs) == 1:
            secs = '0'+secs
        return [mins, secs]

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    @staticmethod
    def calculate_rmse(actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    @staticmethod
    def multinomial_likelihood(neural_network, data, weights):
        y = data[:, neural_network.topology[0]: neural_network.top[2]]
        fx = neural_network.generate_output(data, weights)
        rmse = self.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        probability = neural_network.softmax(fx)
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2])
        out = np.argmax(fx, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = float(count)/y_out.shape[0] * 100
        return [loss, rmse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = MCMC.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss), rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights, tau):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau)
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
        return likelihood, rmse

    def prior_function(self, weights, tau):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current):
        accept = False
        likelihood_ignore, rmse_test_proposal = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal)
        likelihood_proposal, rmse_train_proposal = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        mh_ratio = min(1, np.exp(min(709, difference_likelihood + difference_prior)))
        u = np.random.uniform(0,1)
        if u < mh_ratio:
            accept = True
            likelihood_current = likelihood_proposal
            prior_proposal = prior_current
        return accept, rmse_train_proposal, rmse_test_proposal, likelihood_current, prior_current


    def mcmc_sampler(self, save_knowledge=True):
        train_rmse_file = open(self.directory+'/train_rmse.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse.csv', 'w')
        weights_initial = np.random.uniform(-5, 5, self.w_size)

        # ------------------- initialize MCMC
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = np.log(np.var(prediction_train - y_train))
        tau_proposal = np.exp(eta)
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        rmse_test = self.calculate_rmse(prediction_test, y_test)

        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        num_accept = 0

        tempfit = 0
        self.evaluate()
        tempfit = self.fitness[self.best_index]

        # start sampling
        for sample in range(self.num_samples - 1):
            tempfit = self.best_fit
            self.random_parents()
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break
            if tag == 0:
                break
            self.find_parents()
            self.sort_population()
            self.replace_parents()
            self.best_index = 0
            tempfit = self.fitness[0]
            for x in range(1, self.population_size):
                if(self.fitness[x] < tempfit):
                    self.best_index = x
                    tempfit  =  self.fitness[x]
            # Evaluate population proposal
            # for index in range(self.population_size):
            weights_proposal = self.population[self.best_index]
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)

            if accept:
                num_accept += 1
                weights_current = weights_proposal
                eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test

            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])

            elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))

            print("Sample: {}, Best Fitness: {}, Proposal: {}, Time Elapsed: {}".format(sample, rmse_train_current, rmse_train, elapsed_time))

        elapsed_time = time.time() - self.start
        accept_ratio = num_accept/num_samples

        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()

        return accept_ratio

if __name__ == '__main__':
    num_samples = 10000
    population_size = 100
    problem_type = 'regression'
    topology = [4, 25, 1]
    problem_name = 'synthetic'

    train_data_file = 'synthetic_data/target_train.csv'
    test_data_file = 'synthetic_data/target_test.csv'

    train_data = np.genfromtxt(train_data_file, delimiter=',')
    test_data = np.genfromtxt(test_data_file, delimiter=',')

    model = MCMC(num_samples, population_size, topology, train_data, test_data, directory=problem_name)
    accept_ratio = model.mcmc_sampler()
    print("accept ratio: {}".format(accept_ratio))
