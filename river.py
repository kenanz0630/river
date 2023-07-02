import numpy as np
import copy
import pandas as pd
from scipy.integrate import quad
import time



class Market:
	'''
	Spatiotemporal market with single platform
	'''


	def __init__(self, net, net_prmt, mode, hyper=None):
		'''
		Init market


		args:
			- net (str): 			filename of network
			- net_prmt (str): 		filename of network parameters
			- mode (str): 			ride-hail mode
			- hyper (dict): 		hyper parameters


		'''

		self.network = Network(net, net_prmt)

		dict_meeting_prob = {
			'shail': self._meeting_prob_shail,
			'ehail': self._meeting_prob_ehail,
		}


		dict_meeting_prob_deriv = {
			'shail': self._meeting_prob_deriv_shail,
			'ehail': self._meeting_prob_deriv_ehail,
		}

		self._meeting_prob = dict_meeting_prob[mode]
		self._meeting_prob_deriv = dict_meeting_prob_deriv[mode]

		self._init_hyper(hyper, mode)



	def _init_hyper(self, hyper, mode):
		'''
		Init hyper parameters

		args:
			- hyper (dict): 		hyper parameters
			- mode (str): 			ride-hail mode


		'''
		dict_hyper = {
			'shail': {
				'Delta': 0.25,			# duration of each time period (hr)
				'beta_0': 0.1959, 				# constant coeff
				'beta_v': 0.2949, 				# coeff of cruising speed 
				'beta_rho': -0.3732, 			# coeff of road density
			},
			'ehail': {
				'Delta': 0.25,			# duration of each time period (hr)
				'beta_0': (-0.0751, -0.5384),				# constant coeff
				'beta_v': (0.2789, 0.1223), 				# coeff of cruising speed 
				'beta_rho': (0.0767, 0.0551),			# coeff of road density
				'ratio_threshold': 0.36, 		# threshold of demand-supply ratio
			}
		}

		if hyper is None:
			# defaul hyper parameters
			self.hyper = dict_hyper[mode]
		else:
			self.hyper = hyper






	def _meeting_prob_shail(self, y, q, v, net_prmt, eps=1e-6):
		'''
		Meeting probability for street-hail

		args:
			- y (np.array): 			number of vacant vehicles in each zone
			- q (np.array): 			passenger arrival rate in each zone (pax/hr)
			- v (np.array): 			cruising speed 
			- net_prmt (pd.DataFrame)	parameter of zones

		returns:
			- m (np.array): 		meeting probabilities
		'''	

		hyper = self.hyper

		x = q*hyper['Delta']*net_prmt.A.values/(y+eps)
		k = np.exp(hyper['beta_0'] + hyper['beta_v']*np.log(v) + hyper['beta_rho']*np.log(net_prmt.rho.values))

		m = 1-np.exp(-k*x)

		return m



	def _meeting_prob_deriv_shail(self, y, q, v, net_prmt, eps=1e-6):
		'''
		Derivative of meeting probability wrt y for street-hail

		args:
			- y (np.array): 			number of vacant vehicles in each zone
			- q (np.array): 			passenger arrival rate in each zone (pax/hr)
			- v (np.array): 			cruising speed 
			- net_prmt (pd.DataFrame)	parameter of zones

		returns:
			- m_deriv (np.array): 		derivatives of meeting probabilities wrt y
		'''

		hyper = self.hyper

		m = self._meeting_prob_shail(y, q, v, net_prmt)

		x = q*hyper['Delta']*net_prmt.A.values/(y+eps)
		k = np.exp(hyper['beta_0'] + hyper['beta_v']*np.log(v) + hyper['beta_rho']*np.log(net_prmt.rho.values))

		m_deriv = (m-1)*(k*x/(y+eps))

		return m_deriv




	def _meeting_prob_ehail(self, y, q, v, net_prmt, eps=1e-6):
		'''
		Meeting probability for street-hail

		args:
			- y (np.array): 			number of vacant vehicles in each zone
			- q (np.array): 			passenger arrival rate in each zone (pax/hr)
			- v (np.array): 			cruising speed 
			- net_prmt (pd.DataFrame)	parameter of zones
		
		returns:
			- m (np.array): 			meeting probabilities
		'''

		hyper = self.hyper

		x = q*hyper['Delta']*net_prmt.A.values/(y+eps)
		k = np.exp(hyper['beta_0'][0] + hyper['beta_v'][0]*np.log(v) + hyper['beta_rho'][0]*np.log(net_prmt.rho.values))
		m = 1-np.exp(-k*(x**2))

		idx = x < hyper['ratio_threshold']
		k = np.exp(hyper['beta_0'][1] + hyper['beta_v'][1]*np.log(v[idx]) + hyper['beta_rho'][1]*np.log(net_prmt.rho.values[idx]))
		m[idx] = 1-np.exp(-k*x[idx])


		return m




	def _meeting_prob_deriv_ehail(self, y, q, v, net_prmt, eps=1e-6):
		'''
		Derivative of meeting probability wrt y for street-hail

		args:
			- y (np.array): 			number of vacant vehicles in each zone
			- q (np.array): 			passenger arrival rate in each zone (pax/hr)
			- v (np.array): 			cruising speed 
			- net_prmt (pd.DataFrame)	parameter of zones

		returns:
			- m_deriv (np.array): 		derivatives of meeting probabilities wrt y
		'''

		hyper = self.hyper

		m = self._meeting_prob_ehail(y, q, v, net_prmt)

		x = q*hyper['Delta']*net_prmt.A.values/(y+eps)
		

		k = np.exp(hyper['beta_0'][0] + hyper['beta_v'][0]*np.log(v) + hyper['beta_rho'][0]*np.log(net_prmt.rho.values))
		m_deriv = (m-1)*(2*k*(x**2)/(y+eps))

		idx = x < hyper['ratio_threshold']
		k = np.exp(hyper['beta_0'][1] + hyper['beta_v'][1]*np.log(v[idx]) + hyper['beta_rho'][1]*np.log(net_prmt.rho.values[idx]))
		m_deriv[idx] = (m[idx]-1)*(k*x[idx]/(y[idx]+eps))


		return m_deriv











	def equilibrium(self, fleet, demand, od, prc, reward_type='ue', 
					tau=None, v=None, gamma=1., 
					tol=1e-4, max_iter=1000, print_fnl=True, print_iter=False):
		'''
		Solve equilibrium of vehicle routing

		args:
			- fleet (int): 			fleet size
			- demand (np.array): 	demand pattern (n_time, n_zone)
			- od (np.array): 		od pattern (n_time, n_zone, n_zone)
			- prc (np.array): 		trip surge fare (n_org, n_dest)
			- reward_type (str): 	type of reward function
			- tau (np.array): 		travel time (n_time, n_zone, n_zone) (if None, use network default values)
			- v (np.array): 		travel speed (n_time, n_zone) (if None, use network default values)
			- gamma (float): 		discount factor
			- tol (float): 			gap tolerance
			- max_iterm (int): 		max iteration
			- print_fnl (boolean): 	whether to print convergence info
			- print_iter (boolean): whether to print by iteration


		returns:
			- ls_gap (list): 		list of gaps over iterations
			_ ls_obj (list): 		list of objective values over iterations

		'''

		self.q = demand
		self.H, self.N = demand.shape
		self.od = od
		self.prc = prc
		self.gamma = gamma

		

		# set travel time and speed
		if tau is None:
			self.tau = np.repeat(self.network.tau.reshape(1,self.N,self.N), self.H, axis=0)
		else:
			self.tau = tau

		if v is None:
			self.v = np.repeat(self.network.prmt.v.values.reshape(1,self.N), self.H, axis=0)
		else:
			self.v = v





		dict_reward = {
			'ue': self._reward_ue,
			'so': self._reward_so,
		}

		dict_objective = {
			'ue': self._objective_ue,
			'so': self._objective_so,
		}

		# dict_objective_gradient = {
		# 	'ue': self._objective_gradient_diag_ue
		# }

		self._reward = dict_reward[reward_type]
		self._objective = dict_objective[reward_type]
		# self._objective_gradient = dict_objective_gradient[reward_type]



		self._init(fleet)


		gap = np.inf
		ls_gap = []
		ls_obj = []


		for i in range(max_iter):
			# forward induction
			self._forward()

			# backward induction
			pi = self._backward()


			# line search
			mu = 1/(i+1)
			self.pi = (1-mu)*self.pi + mu*pi


			# compute gap
			gap = self._compute_gap()
			ls_gap.append(gap)


			if reward_type == 'so':
				obj = self._objective(self.x)
				ls_obj.append(obj)


			if print_iter:
				if reward_type == 'ue':
					if i == 0:
						cols = ['iter','gap','elp']
						line = '|'.join([f'{col:<12}' for col in cols])
						print(line)

						start = time.time()

				
					elif i%print_iter == 0:
						elp = time.time() - start
						line = f'{i:<12d}|{gap:<12.4f}|{elp:<12.4f}'
						print(line)

						start = time.time()


				if reward_type == 'so':
					if i == 0:
						cols = ['iter','gap','obj','elp']
						line = '|'.join([f'{col:<12}' for col in cols])
						print(line)

						start = time.time()

					
					elif i%print_iter == 0:
						elp = time.time() - start
						line = f'{i:<12d}|{gap:<12.4f}|{obj:<12.4f}|{elp:<12.4f}'
						print(line)

						start = time.time()

		
			if gap < tol:
				break



		if print_fnl:
			if reward_type == 'ue':
				print(f'Terminate at {i} iteration with gap {gap:<10.4e}')
			if reward_type == 'so':
				print(f'Terminate at {i} iteration with obj {obj:<10.4e}, gap {gap:<10.4e}')


		return ls_gap, ls_obj




	def _compute_gap(self):
		'''
		Compute gap as the difference between Q-values and values

		returns:
			- gap (float):			gap 

		'''

		V = np.max(self.Q, axis=2)
		
		gap = self.Q - V.reshape(self.H, self.N, 1)
		idx = self.x == 0
		gap[idx] = 0
		gap = np.sum(np.abs(gap))/np.sum(np.abs(self.Q))
		# gap = np.sum(np.abs(gap))

		return gap










	def _init(self, fleet):
		'''
		Init vehicle flows, relocation strategies and Q values


		args:
			- fleet (int): 		fleet size


		'''
		
		
		self.y0 = np.ones(self.N, dtype=float)/self.N*fleet	# initial vehicle distribution
		self.Q = np.zeros((self.H, self.N, self.N)) # Q-values
		self.V_fnl = 0 # final value

		self.x = np.zeros((self.H, self.N, self.N)) # relocation flow (n_time, n_zone, n_zone)
		self.m = np.zeros((self.H, self.N)) # meeting probability (n_time, n_zone)
		self.y = np.zeros((self.H, self.N)) # idle vehicle distribution (n_time, n_zone)
		self.d = np.zeros((self.H, self.N)) # future vehicle arrivals (n_time, n_zone)



		self._init_strategy()
		



	def _init_strategy(self):
		'''
		Init search strategy 
			- rangdomly select among current zone and nearby zones

		'''


		pi = np.zeros((self.N, self.N))
		for i in range(self.N):
			idx = self.network.neighbors[i]
			rand = np.random.rand(len(idx))
			pi[i,idx] = rand/np.sum(rand)

		self.pi = np.tile(pi, (self.H,1,1))













	def _forward(self):
		'''
		Forward induction	
			

		'''

		x = np.zeros((self.H, self.N, self.N)) # relocation flow (n_time, n_zone, n_zone)
		m = np.zeros((self.H, self.N)) # meeting probability (n_time, n_zone)
		y = np.zeros((self.H, self.N)) # idle vehicle distribution (n_time, n_zone)
		d = np.zeros((self.H, self.N)) # future vehicle arrivals (n_time, n_zone)
		


		x[0] = self.pi[0]*self.y0.reshape((-1,1))
		y[0] = np.sum(x[0], axis=0)
		m[0] = self._meeting_prob(y[0], self.q[0], self.v[0], self.network.prmt)
		r = (1-m[0])*y[0] # vehicle remaining in current zone

		d = self._update_future_vehicle_arrival(d, 0, m[0]*y[0])

		for t in range(1, self.H):
			# update relocation flow
			x[t] = self.pi[t]*(r+d[t]).reshape((-1,1))

			# update vacant vehicles
			y[t] = np.sum(x[t], axis=0)

			# compute meeting probability
			m[t] = self._meeting_prob(y[t], self.q[t], self.v[t], self.network.prmt)
			r = (1-m[t])*y[t]

			d = self._update_future_vehicle_arrival(d, t, m[t]*y[t])



		self.x = x
		self.m = m
		self.y = y
		self.d = d



	def _update_future_vehicle_arrival(self, d, t, o):
		'''
		Update vehicle arrivals in the future (d)

		args:
			- d (np.array): 	future vehicle arrivals (n_time, n_zone)
			- t (int): 			current time period
			- o (np.array): 	occupied vehicles (n_zone,)

		returns:
			- d (np.array): 	updated future vehicle arrivals (n_time, n_zone)

		'''



		for i in range(self.N):
			# time arriving in zone i
			arrvl_time = t + self.tau[t,:,i]
			arrvl_veh = o*self.od[t,:,i]

			for j in range(self.N):
				if 1+arrvl_time[j] < self.H:
					d[1+arrvl_time[j],i] += arrvl_veh[j]


		return d














	def _backward(self):
		'''
		Backward induction
			- update Q, pi


		'''
		pi = np.zeros((self.H, self.N, self.N))
		for t in range(self.H-1,-1,-1):
			for i in range(self.N):
				idx = self.network.neighbors[i] # list of feasible relocation zones: neighbors (including itself)

				# update Q value
				self.Q[t,i,idx] = np.sum(self._transition(t, idx)*(self._reward(t, idx) + self._discount(t, idx)*self._next_value(t, idx)), axis=1)
		
				# update search strategy
				pi = self._update_strategy_by_Q(pi, t, i, idx)


		return pi
				




	def _transition(self, t, idx):
		'''
		Transition matrix

		args:
			- t (int): 			time period
			- idx (np.array): 	index of search zones


		returns:
			- P (np.array): 	transition matrix (n_search_zone, 1+n_zone)

		'''

		P = np.zeros((len(idx), self.N+1))
		P[:,0] = 1-self.m[t,idx] # unmatched vehicles
		P[:,1:] = self.od[t,idx]*self.m[t,idx].reshape((-1,1)) # matched vehicles
		


		return P



	def _reward_ue(self, t, idx):
		'''
		Reward matrix under UE 

		args:
			- t (int): 			time period
			- idx (np.array): 	index of search zones


		returns:
			- R (np.array): 	reward matrix (n_search_zone, 1+n_zone)

		'''

		R = np.concatenate((np.zeros((len(idx),1)), self.prc[idx]), axis=1)




		return R



	def _reward_so(self, t, idx):
		'''
		Reward matrix under SO

		args:
			- t (int): 			time period
			- idx (np.array): 	index of search zones

		returns:
			- R (np.array): 	reward matrix (n_search_zone, 1+n_zone)

		'''

		m_deriv = self._meeting_prob_deriv(self.y[t,idx], self.q[t,idx], self.v[t,idx], self.network.prmt.iloc[idx])
		m = self.m[t,idx]
		R = self.prc[idx]*(1+self.y[t,idx]*m_deriv/m).reshape((-1,1))
		R = np.concatenate((np.zeros((len(idx),1)), R), axis=1)

		return R





	def _discount(self, t, idx):
		'''
		Discount factor matrix

		args:
			- t (int): 			time period
			- idx (np.array): 	index of search zones
		returns:
			- gamma (np.array): 	discount factor matrix (n_search_zone, 1+n_zone)

		'''

		t = np.concatenate((np.zeros((len(idx),1)), 1+self.tau[t,idx]), axis=1)
		gamma = np.power(self.gamma, t)

		return gamma




	def _next_value(self, t, idx):
		'''
		Next value matrix

		args:
			- t (int): 			time period
			- idx (np.array): 	index of search zones


		returns:
			- V (np.array): 	next value matrix (n_search_zone, 1+n_zone)

		'''

		V = np.zeros((len(idx), 1+self.N))

		# unmatched vehicles
		if t+1 >= self.H:
			V[:,0] = self.V_fnl
		else:
			V[:,0] = np.max(self.Q[t+1,idx], axis=1)


		# matched vehicles
		for i in range(self.N):
			arrvl_time = t+1+self.tau[t,idx,i].reshape(-1) # if arrival time beyond time horizon, set as ending time
			# final values
			idx_fnl = arrvl_time >= self.H
			V[idx_fnl,i+1] = self.V_fnl

			# non-final values
			idx_not = np.logical_not(idx_fnl)
			V[idx_not,i+1] = np.max(self.Q[arrvl_time[idx_not],i], axis=1)


		return V



	def _update_strategy_by_Q(self, pi, t, i, idx):
		'''
		Update search strategy by Q-values

		args:
			- pi (np.array): 	current strategy
			- t (int): 			time period
			- i (int): 			current zone
			- idx (np.array): 	index of search zones


		'''

		Q_max = np.max(self.Q[t,i,idx])
		idx_max = self.Q[t,i] == Q_max
		pi[t,i,idx_max] = 1./np.sum(idx_max)

		return pi






























	def _objective_ue(self, x):
		'''
		Objective (potential function) under UE

		args:
			- x (np.array): 		relocation flow pattern

		returns:
			- obj (float): 			objective value
		'''

		obj = 0

		map_fnc = lambda args: quad(self._meeting_prob, 0, args[0], args=(args[1], self.v[args[2],args[3]], self.network.prmt.iloc[[args[3]]]))
		p_bar = np.sum(self.od*self.prc.reshape((1,self.N,self.N)), axis=2) # avg reward sum over trip dest (n_time,n_zone)
		for t in range(self.H):
			discount = self.gamma**t
			y = np.sum(x[t], axis=0) 
			args = list(zip(y, self.q[t], [t]*self.N, np.arange(self.N)))
			integrals = list(map(map_fnc, args))
			integrals = np.array([integral[0] for integral in integrals]) 
			obj += discount*np.sum(integrals*p_bar[t])


		return obj




	def _objective_so(self, x):
		'''
		Objective (potential function) under UE

		args:
			- x (np.array): 		relocation flow pattern

		returns:
			- obj (float): 			objective value
		'''

		p_bar = np.sum(self.od*self.prc.reshape((1,self.N,self.N)), axis=2) # avg reward sum over trip dest (n_time,n_zone)
		discount = np.power(self.gamma, np.arange(self.H)).reshape((-1,1)) # (n_time,1)
		y = np.sum(x, axis=1) # (n_time, n_zone)
		r = self.m*p_bar # (n_time, n_zone)
		obj = np.sum(y*r*discount) 

		return obj







	def load_equilibrium(self, x, fleet=None, q=None, od=None, prc=None, tau=None, v=None, gamma=1.):
		'''
		Load equilibrium based on relocation flow pattern

		args:
			- x (np.array): 		relocation flow (n_time, n_zone, n_zone)
			- fleet (int): 			fleet size
			- demand (np.array): 	demand pattern (n_time, n_zone)
			- od (np.array): 		od pattern (n_time, n_zone, n_zone)
			- prc (np.array): 		trip surge fare (n_org, n_dest)
			- tau (np.array): 		travel time (n_time, n_zone, n_zone) (if None, use network default values)
			- v (np.array): 		travel speed (n_time, n_zone) (if None, use network default values)
			- gamma (float): 		discount factor


		'''

		self.H, self.N, self.N = x.shape
		self.x = x
		self.pi = x/np.sum(x, axis=2).reshape(self.H, self.N, 1)

		if fleet is not None:
			self.y0 = np.ones(self.N, dtype=float)/self.N*fleet


		if q is not None:
			self.q = q

		if od is not None:
			self.od = od

		if prc is not None:
			self.prc = prc

		if tau is not None:
			self.tau = tau

		if v is not None:
			self.v = v

		self.gamma = gamma


		self._forward() 
		self._backward()









	



class Network:
	'''
	Network of a local ride-hail market

	'''

	def __init__(self, net, net_prmt):
		'''
		Load network

		'''
		df = pd.read_csv(net)

		# nodes
		self.nodes = np.sort(np.unique(np.append(df.from_node.values, df.to_node.values)).astype(int))
		
		# neighbors of each node
		self.neighbors = dict()
		for i in self.nodes:
			self.neighbors[i] = df[(df.from_node == i) & (df.neighbor == 1)].to_node.values.astype(int)



		n = len(self.nodes)
		self.tau = df.tau.values.reshape(n,n)

		# set net parameters
		self.prmt = pd.read_csv(net_prmt)

































