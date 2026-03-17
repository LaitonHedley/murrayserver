import numpy as np
from itertools import count

import random as random
import math 
import sys 
from collections import deque
from tabulate import tabulate

class Agent(object):
	# Itertools used to create an unique id for each agent:
	mkid = next(count())

	# """ Agent """
	def __init__(self, default_utility = 0.1, noise = 0.25, decay = 0.5, mismatchPenalty = None, outcome = True, lendeque = 250000):
	
		self.default_utility = default_utility
		self.noise = noise
		self.decay = decay
		self.temperature = 0.25*math.sqrt(2)
		self.mismatchPenalty = mismatchPenalty
		self.lendeque = lendeque
		self.id = Agent.mkid
		self.instance_history = {}
		self.t = 0

		self.sim = {}
		self.sim['att'] = []
		self.sim['f'] = []

		self.atts = []
		self.simvalues = {}
		self.full_isntance = True 
		self.outcomes = None 
		self.outcome = outcome 
		

	def respond(self, reward):
		if self.outcome:
			if self.mismatchPenalty is not None:
				if self.option[1] not in self.instance_history:
					self.instance_history[self.option[1]] = {(reward,self.option[0]):deque([],self.lendeque)}
					if self.default_utility is not None:
						self.instance_history[self.option[1]][(self.default_utility,self.option[0])] = deque([0],self.lendeque)
				elif (reward,self.option[0]) not in self.instance_history[self.option[1]]:
					self.instance_history[self.option[1]][(reward,self.option[0])] = deque([],self.lendeque)

				self.instance_history[self.option[1]][(reward,self.option[0])].append(self.t) 
			else:
				if self.option not in self.instance_history:
					self.instance_history[self.option] = {reward:deque([],self.lendeque)}
					if self.default_utility is not None:
						self.instance_history[self.option][self.default_utility] = deque([0],self.lendeque)
				elif reward not in self.instance_history[(self.option)]:
					self.instance_history[self.option][reward] = deque([],self.lendeque)

				self.instance_history[self.option][reward].append(self.t) 
		else:
			if self.mismatchPenalty is not None:
				if self.option[1] not in self.instance_history:
					self.instance_history[self.option[1]] = {self.option[0]:deque([],self.lendeque)}
				elif self.option[0] not in self.instance_history[self.option[1]]:
					self.instance_history[self.option[1]][self.option[0]] = deque([],self.lendeque)

				self.instance_history[self.option[1]][self.option[0]].append(self.t) 
			else:
				if self.option not in self.instance_history:
					self.instance_history[self.option] = deque([],self.lendeque)

				self.instance_history[self.option].append(self.t) 

	def populate_at(self, option, t, reward = None):
		if reward is not None:
			if self.mismatchPenalty is not None:
				if option[1] not in self.instance_history:
					self.instance_history[option[1]] = {(reward,option[0]):deque([],self.lendeque)}
				elif (reward,option[0]) not in self.instance_history[option[1]]:
					self.instance_history[option[1]][(reward, option[0])] = deque([],self.lendeque)
				self.instance_history[option[1]][(reward, option[0])].append(t)
			else:
				if (option) not in self.instance_history:
					self.instance_history[option] = {reward:deque([],self.lendeque)}
				elif reward not in self.instance_history[option]:
					self.instance_history[option][reward] = deque([],self.lendeque)
				self.instance_history[option][reward].append(t)
		else:
			if self.mismatchPenalty is not None:
				if option[1] not in self.instance_history:
					self.instance_history[option[1]] = {option[0]:deque([],self.lendeque)}
				elif (option[0]) not in self.instance_history[option[1]]:
					self.instance_history[option[1]][option[0]] = deque([],self.lendeque)
				self.instance_history[option[1]][option[0]].append(t)
			else:
				if (option) not in self.instance_history:
					self.instance_history[option] = deque([],self.lendeque)
				self.instance_history[option].append(t)
	
	# def compute_blended(self, t, options):
	# 	blends = []
	# 	if self.mismatchPenalty is not None:
	# 		for o, i in zip(options,count()):
	# 			a = o[1]
	# 			s = o[0]
	# 			if a in self.instance_history:
	# 				tmps =[]
	# 				rewards = []
	# 				for ro in self.instance_history[a]:
	# 					if len(self.instance_history[a][ro])>0:
	# 						tmp = np.copy(self.instance_history[a][ro])
	# 						tmp = t - tmp
	# 						tmp = math.log(sum(pow(tmp,-self.decay))) + self.noise*self.make_noise()
	# 						if (s,ro[1]) in self.simvalues:
	# 							tmp = tmp + self.mismatchPenalty*self.simvalues[s,ro[1]]
	# 						else:
	# 							tmp = tmp + self.mismatchPenalty*self.get_similarity(s,ro[1])
	# 						tmps.append(tmp)
	# 						rewards.append(ro[0])
	# 				if self.default_utility is not None:
	# 					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
	# 					tmps.append(tmp0)
	# 					rewards.append(self.default_utility)
	# 				tmps = np.array(tmps)
	# 				tmps = np.exp(tmps/self.temperature)
	# 				p = tmps/sum(tmps)
	# 				rewards = np.array(rewards)
	# 				result = sum(rewards*p)
	# 				blends.append((result,i))
	# 			elif self.default_utility is not None:
	# 				blends.append((self.default_utility,i))
	# 	else:
	# 		for o, i in zip(options,count()):
	# 			if o in self.instance_history:
	# 				tmps =[]
	# 				rewards = []
	# 				for r in self.instance_history[o]:
	# 					if len(self.instance_history[o][r])>0:
	# 						tmp = np.copy(self.instance_history[o][r])
	# 						tmp = t - tmp
	# 						tmp = math.log(sum(pow(tmp,-self.decay))) + self.noise*self.make_noise()
	# 						if self.mismatchPenalty is not None:
	# 							tmp = tmp + self.mismatchPenalty*self.get_similarity(o)
	# 						tmps.append(tmp)
	# 						rewards.append(r)
	# 				if self.default_utility is not None:
	# 					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
	# 					tmps.append(tmp0)
	# 					rewards.append(self.default_utility)
	# 				tmps = np.array(tmps)
	# 				tmps = np.exp(tmps/self.temperature)
	# 				p = tmps/sum(tmps)
	# 				rewards = np.array(rewards)
	# 				result = sum(rewards*p)
	# 				blends.append((result,i))
	# 			elif self.default_utility is not None:
	# 				blends.append((self.default_utility,i))
	# 	return blends 
	def make_noise(self):
		p = random.uniform(sys.float_info.epsilon, 1-sys.float_info.epsilon)
		result = math.log((1.0-p) / p)
		return result
	def choose(self, options, base = None):
		self.t += 1
		utilities = self.CompBlended(self.t, options, base)
		best_utility = max(utilities,key=lambda x:x[0])[0]
		best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
		self.option = options[best]
		return self.option 
	
	def reset(self):
		self.t = 0
		self.instance_history = {}
		self.sim = {}
		self.sim['att'] = []
		self.sim['f'] = []

		self.atts = []
		self.simvalues = {}
	
	def instances(self):
		if self.outcome:
			if self.mismatchPenalty is None:
				print(tabulate([[a,b,list(self.instance_history[a][b])] for a in self.instance_history for b in self.instance_history[a]], headers=['option','outcome','occurences']))
			else:
				print(tabulate([[(b[1],a),b[0],list(self.instance_history[a][b])] for a in self.instance_history for b in self.instance_history[a]], headers=['option','outcome','occurences']))
		else:
			if self.mismatchPenalty is None:
				print(tabulate([[a,list(self.instance_history[a])] for a in self.instance_history], headers=['option','occurences']))
			else:
				print(tabulate([[(b,a),list(self.instance_history[a][b])] for a in self.instance_history for b in self.instance_history[a]], headers=['option','occurences']))
	
	def prepopulate(self, option, reward=None):
		if reward is not None:
			if self.mismatchPenalty is not None:
				if option[1] not in self.instance_history:
					self.instance_history[option[1]] = {(reward,option[0]):deque([],self.lendeque)}
				elif (reward,option[0]) not in self.instance_history[option[1]]:
					self.instance_history[option[1]][(reward, option[0])] = deque([],self.lendeque)
				self.instance_history[option[1]][(reward, option[0])].append(0)
			else:
				if (option) not in self.instance_history:
					self.instance_history[option] = {reward:deque([],self.lendeque)}
				elif reward not in self.instance_history[option]:
					self.instance_history[option][reward] = deque([],self.lendeque)
				self.instance_history[option][reward].append(0)
		else:
			if self.mismatchPenalty is not None:
				if option[1] not in self.instance_history:
					self.instance_history[option[1]] = {option[0]:deque([],self.lendeque)}
				elif option[0] not in self.instance_history[option[1]]:
					self.instance_history[option[1]][option[0]] = deque([],self.lendeque)
				self.instance_history[option[1]][option[0]].append(0)
			else:
				if (option) not in self.instance_history:
					self.instance_history[option] = deque([],self.lendeque)
				
				self.instance_history[option].append(0)
	
	def similarity(self,attributes,function):
		self.sim['att'].append(attributes)
		self.sim['f'].append(function)
	
	def get_similarity(self,option, option2 = None):
		result = 0
		if self.mismatchPenalty is not None:
			np_option = np.asarray(option)
			np_option2 = np.asarray(option2)
			for att, f in zip(self.sim['att'],self.sim['f']):
				tmp = f(np_option[att].reshape(-1,1),np_option2[att].reshape(-1,1))
				if tmp.ndim == 1:
					result += sum(tmp)
				else:
					result += sum(sum(tmp))
			# print('op', option, option2)
			self.simvalues[(option,option2)] = result
			self.simvalues[(option2,option)] = result
		elif len(self.atts)>0:
			np_option = np.asarray(option[0])
			np_atts =  np.asarray(self.atts)
			for att, f in zip(self.sim['att'],self.sim['f']):
				result += sum(sum(f(np_option[att].reshape(1,-1),np_atts[:,att])))
			if option in self.instance_history:
				result = result - 1
		return result
	def equal_delay_feedback(self, reward, episode_history):	
		for i in reversed(range(len(episode_history))):
			s,a,r,t1 = episode_history[i]
			#update outcomes
			t = self.instance_history[(s, a)][r].pop()
			if reward not in self.instance_history[(s, a)]:
				self.instance_history[(s, a)][reward] = deque([],self.lendeque)

			self.instance_history[(s, a)][reward].append(t)

	def CompActivation(self, t, o, base = None):
		if self.outcome:
			tmps =[]
			rewards = []
			if self.mismatchPenalty is not None:
				a = o[1]
				s = o[0]
				if a in self.instance_history:
					for ro in self.instance_history[a]:
						if len(self.instance_history[a][ro])>0:
							if base is None:
								tmp = np.copy(self.instance_history[a][ro])
								tmp = t - tmp
								tmp = math.log(sum(pow(tmp,-self.decay))) + + self.noise*self.make_noise()
							else:	
								tmp = base + self.noise*self.make_noise()
							if (s,ro[1]) in self.simvalues:
								tmp = tmp + self.mismatchPenalty*self.simvalues[s,ro[1]]
							else:
								tmp = tmp + self.mismatchPenalty*self.get_similarity(s,ro[1])
							tmps.append(tmp)
							rewards.append(ro[0])
					# if self.default_utility is not None:
					# 	tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					# 	tmps.append(tmp0)
					# 	rewards.append(self.default_utility)
				elif self.default_utility is not None:
					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					tmps.append(tmp0)
					rewards.append(self.default_utility)
			else:
				if o in self.instance_history:
					for r in self.instance_history[o]:
						if len(self.instance_history[o][r])>0:
							if base is None:
								tmp = np.copy(self.instance_history[o][r])
								tmp = t - tmp
								tmp = math.log(sum(pow(tmp,-self.decay))) + self.noise*self.make_noise()
							else:
								tmp = base + self.noise*self.make_noise() 
							tmps.append(tmp)
							rewards.append(r)
					# if self.default_utility is not None:
					# 	tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					# 	tmps.append(tmp0)
					# 	rewards.append(self.default_utility)
				elif self.default_utility is not None:
					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					tmps.append(tmp0)
					rewards.append(self.default_utility)
			tmps = np.array(tmps)
			rewards = np.array(rewards)
			return tmps, rewards 
		else:
			tmps =[]
			if self.mismatchPenalty is not None:
				a = o[1]
				s = o[0]
				if a in self.instance_history:
					for ro in self.instance_history[a]:
						if len(self.instance_history[a][ro])>0:
							if base is None:
								tmp = np.copy(self.instance_history[a][ro])
								tmp = t - tmp
								tmp = math.log(sum(pow(tmp,-self.decay))) + + self.noise*self.make_noise()
							else:	
								tmp = base + self.noise*self.make_noise()
							if (s,ro) in self.simvalues:
								tmp = tmp + self.mismatchPenalty*self.simvalues[s,ro]
							else:
								tmp = tmp + self.mismatchPenalty*self.get_similarity(s,ro)
							tmps.append(tmp)
					if self.default_utility is not None:
						tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
						tmps.append(tmp0)
				elif self.default_utility is not None:
					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					tmps.append(tmp0)		
			else:
				if o in self.instance_history:
					if len(self.instance_history[o])>0:
						if base is None:
							tmp = np.copy(self.instance_history[o])
							tmp = t - tmp
							tmp = math.log(sum(pow(tmp,-self.decay))) + self.noise*self.make_noise()
						else:
							tmp = base + self.noise*self.make_noise() 
						tmps.append(tmp)
					if self.default_utility is not None:
						tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
						tmps.append(tmp0)
				elif self.default_utility is not None:
					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					tmps.append(tmp0)
			tmps = np.array(tmps)
			return tmps

	def CompProbability(self, t, o, base = None):
		if self.outcome:
			acts, rewards = self.CompActivation(t,o,base)
			tmps = np.exp(acts/self.temperature)
			p = tmps/sum(tmps)
			return p, rewards
		else:
			acts = self.CompActivation(t,o,base)
			tmps = np.exp(acts/self.temperature)
			p = tmps/sum(tmps)
			return p

	def CompBlended(self, t, options, base = None):
		blendeds = []
		if self.outcome:
			for o, i in zip(options,count()):
				p, rewards = self.CompProbability(t, o, base)
				result = sum(rewards*p)
				blendeds.append((result,i))
		elif self.outcomes is not None: 
			for o, i in zip(options,count()):
				p = self.CompProbability(t, o, base)
				result = sum(self.outcomes[o]*p)
				blendeds.append((result,i))
		else:
			tmp = []
			for o, i in zip(options,count()):
				acts = self.CompActivation(t,o,base)
				tmps = np.exp(acts/self.temperature)
				tmp.append(sum(tmps))
			tmp = np.array(tmp)
			tmp = tmp/sum(tmp)
			for i in range(len(tmp)):	
				blendeds.append((tmp[i],i))
		return blendeds 



			

