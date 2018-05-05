# !/usr/bin/python

 
# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2018 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra
# rohitash-chandra.github.

#EVolutionary MCMC

#built on: https://github.com/rohitash-chandra/real-coded-genetic-alg
#https://github.com/rohitash-chandra/mcmc-randomwalk

# Evol alg uses Real coded genetic algoritm with Wright's heuristic crossover operator, roulette wheel selection, uniform mutation 

#Choose the crossover operator. Choose population size

### Output

#Posterior distribution plots 


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math 
import random  


class Network:
	def __init__(self, Topo, Train, Test):
		self.Top = Topo  # NN topology [input, hidden, output]
		self.TrainData = Train
		self.TestData = Test
		np.random.seed()

		self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
		self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
		self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
		self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

		self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
		self.out = np.zeros((1, self.Top[2]))  # output last layer

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def RMSE_Er(self, targets):
		return np.sqrt((np.square( np.subtract(np.absolute(self.out),np.absolute(targets)))).mean())

	def RMSE(self, actual, targets):
		return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())
 

	def sampleEr(self, actualout):
		error = np.subtract(self.out, actualout)
		sqerror = np.sum(np.square(error)) / self.Top[2] 
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) - self.B1
		self.hidout = self.sigmoid(z1)  # output of first hidden layer
		z2 = self.hidout.dot(self.W2) - self.B2
		self.out = self.sigmoid(z2)  # output second hidden layer

	def BackwardPass(self, Input, desired, vanilla):
		out_delta = (desired - self.out) * (self.out * (1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

		self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
		self.B2 += (-1 * self.lrate * out_delta)
		self.W1 += (Input.T.dot(hid_delta) * self.lrate)
		self.B1 += (-1 * self.lrate * hid_delta)

	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
		self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


	def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.

		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for pat in xrange(0, size):
			Input[:] = data[pat, 0:self.Top[0]]
			Desired[:] = data[pat, self.Top[0]:]

			self.ForwardPass(Input)
			try:
				fx[pat] = self.out
			except:
			   print 'Error'



		return fx

	def ForwardFitnessPass(self, data, w):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.

		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)
		actual = np.zeros(size)

		for pat in xrange(0, size):
			Input[:] = data[pat, 0:self.Top[0]]
			Desired[:] = data[pat, self.Top[0]:]

			actual[pat] = data[pat, self.Top[0]:]

			self.ForwardPass(Input)
			try:
				fx[pat] = self.out
			except:
			   print 'Error'


		# FX holds prediction

		return self.RMSE(actual, fx)



class Evolution:
	def __init__(self, prob, pop_size,   max_evals,  xover_rate, mu_rate, min_fitness, chose_xover, topology, traindata, testdata):
 
		self.pop   = [] 
		self.new_pop =  []

		self.pop_size = pop_size
		self.num_variables = 0 # will get updated later
		self.max_evals = max_evals 
		self.best_index = 0
		self.best_fit = 0
		self.worst_index = 0
		self.worst_fit = 0

		self.xover_rate = xover_rate 
		self.mu_rate = mu_rate 
		self.fit_list = np.zeros(pop_size)

		self.fit_ratio = np.zeros(pop_size)

		self.max_limits = [] # defines limits in your parameter space - will get updated in initilize() function
		self.min_limits = []
		self.stepsize_vec  =  []
		self.step_ratio = 0.1 # determines the extent of noise you add when you mutate

		self.num_eval = 0 # begin with

		self.problem = prob  # 1 rosen, 2 ellipsoidal 

		self.min_error = min_fitness

		self.xover = chose_xover  # 1. Wrights Heuristic Xover, 2. Simulated Binary Xover

		self.topology = topology  # max epocs
		self.traindata = traindata  #
		self.testdata = testdata
		# ----------------

	def rmse(self, predictions, targets):
		return np.sqrt(((predictions - targets) ** 2).mean())


	def neuralnet_fit(self, neuralnet, data, w):
		y = data[:, self.topology[0]]

		fx = neuralnet.evaluate_proposal(data, w)
		rmse = self.rmse(fx, y)

		return 1/rmse  # inverse as evo alg will maximise a minimization problem

	def likelihood_func(self, neuralnet, data, w, tausq):
		y = data[:, self.topology[0]]
		fx = neuralnet.evaluate_proposal(data, w)
		rmse = self.rmse(fx, y)
		loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
		return [np.sum(loss), fx, rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
		h = self.topology[1]  # number hidden neurons
		d = self.topology[0]  # number input neurons
		part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss
 


	def initialize(self):

		self.pop   =np.random.rand(self.pop_size, self.num_variables) * 5
		self.new_pop   = self.pop 



		#max_limits = [2, 2, 2, 2, 2]  # can handle different limits for different variables  ( not really needed for neural networks )
		self.max_limits = np.repeat(50, self.num_variables)
		#min_limits = [0, 0, 0, 0, 0] 

		self.min_limits = np.repeat(-50, self.num_variables)

		span = np.subtract(self.max_limits,self.min_limits)/self.max_limits

		self.stepsize_vec  =  np.zeros(self.num_variables)


		for i in range(self.num_variables): # calculate the step size of each of the parameters. just for mutation by random-walk
			self.stepsize_vec[i] = self.step_ratio  #* span[i] 

	def print_pop(self ):

		print(self.pop, ' self.pop')
 


	'''def fit_func(self, x):    #   these are optimisation problems. for evo-MCMC we will use model_func()
		fit = 0

		if self.problem == 1: # rosenbrock
			for j in range(x.size -1): 
				fit  = fit +  (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))

		elif self.problem ==2:  # ellipsoidal - sphere function
			for j in range(x.size):
				fit= fit + ((j+1)*(x[j]*x[j]))

		if fit ==0:
			fit = 1e-20 # to be safe with division by 0
				  

		return 1/fit # note we will maximize fitness, hence minimize error  
		'''
 

	def evaluate_population(self,  neuralnet, data): 
		w = self.pop[0,:] 
		self.fit_list[0] =   self.neuralnet_fit( neuralnet, data, w)
		self.best_fit = self.fit_list[0]
		self.best_index = 0

		sum = 0

		for i in range(self.pop_size):
			w = self.pop[i,:]
			self.fit_list[i] = self.neuralnet_fit( neuralnet, data, w)
			sum = sum + self.fit_list[i]

			if self.best_fit > self.fit_list[i]:
				self.best_fit = self.fit_list[i]
				self.best_index = i  

		self.num_eval = self.num_eval + self.pop_size
 
		 
		for j in range(self.pop_size):
			self.fit_ratio[j] = (self.fit_list[j]/sum)* 100

 

	def roullete_wheel(self):

		wheel = np.zeros(self.pop_size+1) 
		wheel[0] = 0

		u = np.random.randint(100)

		if u == 0:
			u = 1  

		for i in range(1, wheel.size):
			wheel[i] = self.fit_ratio[i-1] + wheel[i-1]  
		for j in range( wheel.size-1):
			if((u> wheel[j]) and (u < wheel[j+1])):
				return j   
	 
		return 0

 

	def sim_binary_xover(self,ind1, ind2): # Simulated Binary Crossover https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

	#:param ind1: The first individual participating in the crossover.
	#:param ind2: The second individual participating in the crossover.
	#:param eta: Crowding degree of the crossover. A high eta will produce
	 #           children resembling to their parents, while a small eta will
	  #          produce solutions much more different.
		x1 = ind1
		x2  = ind2 

		eta = 2

		for i, (x1, x2) in enumerate(zip(ind1, ind2)):
			rand = random.random()
			if rand <= 0.5:
				beta = 2. * rand
			else:
				beta = 1. / (2. * (1. - rand))

			beta **= 1. / (eta + 1.)

			ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
			ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))


		return ind1, ind2

 


	def blend_xover(self,ind1, ind2):
		'''Executes a blend crossover that modify in-place the input individuals. 
		:param ind1: The first individual participating in the crossover.
		:param ind2: The second individual participating in the crossover.
		:param alpha: Extent of the interval in which the new values can be drawn
				  for each attribute on both side of the parents' attributes. '''
		alpha = 0.1
		x1 = ind1
		x2 = ind2


		for i in range(ind1.size): #(x1, x2) in enumerate(zip(ind1, ind2)):
			gamma = (1. + 2. * alpha) * random.random() - alpha
			ind1[i] = (1. - gamma) * x1[i] + gamma * x2[i]
			ind2[i] = gamma * x1[i] + (1. - gamma) * x2[i]

		return ind1, ind2




	def wrights_xover(self,left_fit, right_fit, left, right):   #Wrights Heuristic Crossover

		alpha = random.uniform(0,1) 
		if ( left_fit > right_fit): 
			child = (alpha *(left-right))+ left
		else: 
			child = (alpha *  (right-left))+ right

		return child

	def impose_limits(self, child, prev):

		for j in range(child.size):
			if child[j] > self.max_limits[j]:
					child[j] = prev[j]
			elif child[j] < self.min_limits[j]:
					child[j] = prev[j] 
		return child 

	def xover_mutate(self, leftpair,rightpair):  # xover and mutate
 
		left = self.pop[leftpair,:]
		right = self.pop[rightpair,:]

		left_fit = self.fit_list[leftpair]
		right_fit =  self.fit_list[rightpair] 

		u = random.uniform(0,1)

		if u < self.xover_rate:  # implement xover  

			if self.xover == 1:
				child_one = self.wrights_xover(left_fit, right_fit, left, right ) 
				child_two = self.wrights_xover(left_fit, right_fit, left, right ) 

			elif self.xover == 2:
				child_one, child_two = self.sim_binary_xover( left, right)

			elif self.xover == 3:
				child_one, child_two = self.blend_xover( left, right)

		else: 
			child_one = left
			child_two = right

		if u < self.mu_rate: # implement mutation 
			child_one =  np.random.normal(child_one, self.stepsize_vec) 
			child_two =  np.random.normal(child_two, self.stepsize_vec)
 
			# check if the limits satisfy - keep in range 
		child_one = self.impose_limits(child_one, left) 
		child_two = self.impose_limits(child_two, right)

		return child_one, child_two



	def view_posterior(self, list, title  ): 

		list_points =  list
 
		width = 9 

		font = 9

		fig = plt.figure(figsize=(10, 12))
		ax = fig.add_subplot(111) 
		slen = np.arange(0,len(list),1) 
		 
		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)
	
		ax1 = fig.add_subplot(211) 

		n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)	
 
    

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212) 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig('results/posterior/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf() 



	def evo_MCMC(self):

		samples = self.max_evals # max number of samples or function evals 

		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]


		netw = self.topology  # [input, hidden, output]
		 
		
		w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

		self.num_variables = w_size 

		pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
		pos_tau = np.ones((samples, 1))

		fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
		fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
		
		rmse_train = np.zeros(samples)
		rmse_test = np.zeros(samples)

		 
		w = np.random.randn(w_size)
		w_proposal = np.random.randn(w_size)
 
		step_eta = 0.01;
		# --------------------- Declare FNN and initialize

		neuralnet = Network(self.topology, self.traindata, self.testdata)
		 
		pred_train = neuralnet.evaluate_proposal(self.traindata, w)
		 
		eta = np.log(np.var(pred_train - self.traindata[:, netw[0]]))
		tau_pro = np.exp(eta) # initial 

		sigma_squared = 25 # detemined by limits in weight space
		nu_1 = 0
		nu_2 = 0

		prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

		[likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
		[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

		print likelihood

 


		self.initialize()
 
		global_bestfit = 0
		global_best = []
		global_bestindex = 0

		naccept = 0 # number of samples accepted

		i =0
 

		while(i< self.max_evals-1):



			# --------------------------------------------------------------- This is evolutionary phase

			self.evaluate_population( neuralnet, self.traindata)  
 
			for p in range(0,  self.pop_size , 2):
 
				leftpair =  self.roullete_wheel() #np.random.randint(self.pop_size) 
				rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size)  

				while (leftpair == rightpair): 
					leftpair =  self.roullete_wheel()  # np.random.randint(self.pop_size) 
					rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size) 

				first_child, second_child = self.xover_mutate(leftpair,rightpair) 
		
				self.new_pop[p,:] = first_child 
				self.new_pop[p+1 ,:] = second_child
				

			best = self.pop[self.best_index, :]

			if self.best_fit > global_bestfit:
				global_bestfit = self.best_fit 
				global_best = self.pop[self.best_index, :]
 

			self.pop = self.new_pop


			self.pop[0,:] = global_best # ensure that you retain the best 

			#print(best, ' is best so far')
			print(self.num_eval, 1/self.best_fit, ' is local best fit')
			print(self.num_eval, 1/global_bestfit, ' is global best fit')
			#print(self.pop, ' self.pop')
			#print(self.fit_list, ' self.fit_list')


  
 
			#if  (1/self.best_fit) < self.min_error:
				#print(' reached min error')
				#break 

				# after evolution of the population, accept or reject each population member by MH criteria  -----------------------------------------------

			#k = 0

			#for i in range(self.num_eval  , (self.pop_size + self.num_eval  )   ):
			for k in range(0  ,   self.pop_size     ):





				eta_pro = eta + np.random.normal(0, step_eta, 1)
				tau_pro = math.exp(eta_pro)

				w_proposal = self.pop[k,:]
				#print w_proposal, ' w_proposal'

 

				[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,tau_pro)
				[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,tau_pro)

				#print  likelihood_proposal,  rmsetrain, rmsetest,  ' ------------------- '

				#print(i, k,    self.num_eval, rmsetrain, rmsetest,  '   i rmsetrain rmsetest ****     ')


				prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro)  # takes care of the gradients

				diff_likelihood = likelihood_proposal - likelihood
				diff_priorliklihood = prior_prop - prior_likelihood

				mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

				u = random.uniform(0, 1)


				#k = k +1


				if u < mh_prob:
				# Update position 
					naccept += 1
					likelihood = likelihood_proposal
					prior_likelihood = prior_prop
					w = w_proposal
					eta = eta_pro

					print  naccept, i,    likelihood, prior_likelihood, rmsetrain, rmsetest,  'accepted'

					pos_w[i + 1,] = w_proposal # weights 
					pos_tau[i + 1,] = tau_pro  # noise in predictions 
					fxtrain_samples[i + 1,] = pred_train # predictions
					fxtest_samples[i + 1,] = pred_test
					rmse_train[i + 1,] = rmsetrain
					rmse_test[i + 1,] = rmsetest
					  


				else:
					#print  naccept, i, i+1,  likelihood, prior_likelihood, rmsetrain, rmsetest,  ' NOT accepted'

					pos_w[i + 1,] = pos_w[i,] # weights 
					pos_tau[i + 1,] = pos_tau[i,] # noise in predictions 
					fxtrain_samples[i + 1,] = fxtrain_samples[i,] # predictions
					fxtest_samples[i + 1,] = fxtest_samples[i,]
					rmse_train[i + 1,] = rmse_train[i,]
					rmse_test[i + 1,] = rmse_test[i,]


				

				i = i +1  
				if i == self.max_evals -1: 
					break


		print naccept, ' n accepted                    ******  '

		accept_per= naccept/(self.max_evals *1.0) * 100


		for s in range(w_size):  
			self.view_posterior(pos_w[s,:], 'pos_w_'+str(s)  ) 
					 






  

		return  global_best, 1/global_bestfit, pos_w, pos_tau, fxtrain_samples, fxtest_samples, accept_per, rmse_train,  rmse_test


 

def main():




	file=open('results/results_fitness.txt','a')

	file_solution=open('results/results_solution.txt','a')

	#-------- set up neural network training data and topology




	random.seed(time.time()) 

	min_fitness = 0.005  # stop when fitness reaches this value. not implemented - can be implemented later


	max_evals = 5000  # need to decide yourself - function evaluations 

	pop_size = 10  # to be adjusted for the problem 

	xover_rate = 0.8 # ideal, but you can adjust further 
	mu_rate = 0.1

 



	prob = 1 # 1 is rosenbrock, 2 is ellipsoidal

	num_experiments = 1 # number of experiments with diffirent initial population

	num_problem = 1 # number of problems in total

	# lets evaluate which one is the best xover operator, 

	#pre results show that for 10 dimensions - blend xover is best, then wrights, then sim binary xover - for Rosenbrock problem
	#wrights heuristic works best for ellipsoidal problem

	global_fit = np.zeros(num_experiments)


	hidden = 5
	input = 4  #
	output = 1

	topology = [input, hidden, output]

	choose_xover  = 1 # 1. Wrights Heuristic Xover, 2. Simulated Binary Xover, 3. Blend xover, 4. 


	for problem in range(num_problem): 
 
		if problem == 1:
			traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
		elif problem == 0:
			traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
		elif problem == 2:
			traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #

		print(traindata)
 

		for j in range(num_experiments): 

			evo = Evolution(prob, pop_size,   max_evals,   xover_rate, mu_rate, min_fitness, choose_xover, topology, traindata, testdata)
 
			global_best, global_bestfit, pos_w, pos_tau, fx_train, fx_test, per_accept, rmse_train, rmse_test = evo.evo_MCMC()
			#np.savetxt(file_solution, np.transpose([i,j, global_bestfit, best_fit]), fmt='%1.1f')
			#np.savetxt(file_solution, np.transpose(global_best) , fmt='%1.4f') 
			#np.savetxt(file_solution, np.transpose(best) , fmt='%1.4f') 


			global_fit[j] = global_bestfit 

			print(j,  global_bestfit, ' best and global best fit')

			print(j,   global_best, ' best and global best sol')

			print(j, per_accept , ' is percentage accepted')

			burnin = int(0.1 * max_evals)  # use post burn in samples

			pos_w = pos_w[int(burnin):, ]
			pos_tau = pos_tau[int(burnin):, ]

			fx_mu = fx_test.mean(axis=0)
			fx_high = np.percentile(fx_test, 95, axis=0)
			fx_low = np.percentile(fx_test, 5, axis=0)

			fx_mu_tr = fx_train.mean(axis=0)
			fx_high_tr = np.percentile(fx_train, 95, axis=0)
			fx_low_tr = np.percentile(fx_train, 5, axis=0)

			print(rmse_train )

			rmse_tr = np.mean(rmse_train[int(burnin):])
 
			rmsetr_std = np.std(rmse_train[int(burnin):])
			rmsetest= np.mean(rmse_test[int(burnin):])
 

			rmsetest_std = np.std(rmse_test[int(burnin):])
 
			print rmse_tr, rmsetr_std, rmsetest, rmsetest_std, '  rmse_tr, rmsetr_std, rmse_test, rmsetest_std '
			np.savetxt(file, [rmse_tr, rmsetr_std, rmsetest, rmsetest_std, per_accept], fmt='%1.5f')



			# for plotting
			x_test = np.linspace(0, 1, num=testdata.shape[0])
			x_train = np.linspace(0, 1, num=traindata.shape[0])  

			ytestdata = testdata[:, input]
			ytraindata = traindata[:, input]

			plt.plot(x_test, ytestdata, label='actual')
			plt.plot(x_test, fx_mu, label='pred. (mean)')
			plt.plot(x_test, fx_low, label='pred.(5th percen.)')
			plt.plot(x_test, fx_high, label='pred.(95th percen.)')
			plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
			plt.legend(loc='upper right')

			plt.title("Plot of Test Prediction with Uncertainty ")
			plt.savefig('results/test.png') 
			plt.clf()
		# -----------------------------------------
			plt.plot(x_train, ytraindata, label='actual')
			plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
			plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
			plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
			plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
			plt.legend(loc='upper right')

			plt.title("Plot of Train Prediction with Uncertainty")
			plt.savefig('results/train.png') 
			plt.clf()

			mpl_fig = plt.figure()
			ax = mpl_fig.add_subplot(111)

			ax.boxplot(pos_w)

			ax.set_xlabel('[W1] [B1] [W2] [B2]')
			ax.set_ylabel('Posterior')

			plt.legend(loc='upper right')

			plt.title("Boxplot of Posterior W (weights and biases)")
			plt.savefig('results/w_pos.png') 

			plt.clf()




		#print(global_fit, '    global bestfit')  
		#print(i, np.mean(global_fit), '  mean  global bestfit') 
		#np.savetxt(file, global_fit)
 
		#np.savetxt(file, [problem,np.mean(global_fit) ], fmt='%1.2f')
 



 

if __name__ == "__main__": main()
