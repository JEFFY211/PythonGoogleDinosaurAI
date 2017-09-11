import numpy as np
import copy
from scipy.special import expit

class ImprovedLearner():
	def __init__(self):
		self.values = [
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			'''
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			'''
		]	
		self.height = 0
		self.dist = 0
		self.no_land = 0
		self.number = 0
		self.fitness = 0
	@property
	def hd_node(self):
		'''
		if(((self.values[0] * self.height) + (self.values[1] * self.dist) + self.values[2]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[0] * self.height) + (self.values[1] * self.dist) + self.values[2])
	@property
	def hd2_node(self):
		'''
		if(((self.values[3] * self.height) + (self.values[4] * self.dist) + self.values[5]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[3] * self.height) + (self.values[4] * self.dist) + self.values[5])

	@property
	def hidden_node1(self):
		'''
		if(((self.values[6] * self.hd_node) + (self.values[7] * self.hd2_node) + self.values[8]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[6] * self.hd_node) + (self.values[7] * self.hd2_node) + self.values[8])
	
	@property
	def hidden_node2(self):
		'''
		if(((self.values[9] * self.hd_node) + (self.values[10] * self.hd2_node) + self.values[11]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[9] * self.hd_node) + (self.values[10] * self.hd2_node) + self.values[11])

	@property
	def hidden1_node1(self):
		'''
		if(((self.values[12] * self.hidden_node1) + (self.values[13] * self.hidden_node2) + self.values[14]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[12] * self.hidden_node1) + (self.values[13] * self.hidden_node2) + self.values[14])

	@property
	def hidden1_node2(self):
		'''
		if(((self.values[15] * self.hidden_node1) + (self.values[16] * self.hidden_node2) + self.values[17]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[15] * self.hidden_node1) + (self.values[16] * self.hidden_node2) + self.values[17])
	
	@property
	def output_no_land(self):
		'''
		if(((self.values[18] * self.no_land) + self.values[19]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit(self.values[18] * self.no_land + self.values[19])

	@property
	def output_node1(self):
		'''
		if(((self.values[20] * self.hidden1_node1) + (self.values[21] * self.hidden1_node2) + self.values[22]) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[20] * self.hidden1_node1) + (self.values[21] * self.hidden1_node2) + self.values[22])

	@property
	def output_node2(self):
		'''
		if((((self.values[23] * self.hidden1_node1) + (self.values[24] * self.hidden1_node2) + self.values[25])) > 0):
			return 1
		else:
			return 0#
		'''
		return expit((self.values[23] * self.hidden1_node1) + (self.values[24] * self.hidden1_node2) + self.values[25])
	@property
	def outputValue(self):
		return expit((self.values[26] * self.output_node1) + (self.values[27] * self.output_node2) + (self.values[28] * self.output_no_land) + self.values[29])
		
	def output(self, distance, height, no_land_obs):
		self.height = height
		self.dist = distance
		self.no_land = no_land_obs
		return self.outputValue
	def randomize(self):
		for i in range(0, len(self.values)):
			self.values[i] = np.random.rand()
def clone(learner1):
	tmpLearner = ImprovedLearner()
	tmpLearner.values = copy.deepcopy(learner1.values)
	return tmpLearner

def mutate(learner1):
	tmpLearner = clone(learner1)
	for i in range(0, len(learner1.values)):
		tmp = np.random.rand()
		if(tmp < 0.4):
			tmpLearner.values[i] = learner1.values[i]  * (np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5)
		'''
		if(tmp > 0.8):
			tmpLearner.values[i] = learner1.values[i] * (0.75 + (np.random.rand()/2))
			#print(learner1.values[i] * (0.5 + np.random.rand()))
		elif(tmp < 0.2):
			tmpLearner.values[i] = learner1.values[i] * (-1 * (np.random.rand()/2) - 0.75)
			#print(learner1.values[i] * (-1 * np.random.rand() - 0.5))
		'''
	return tmpLearner

def cross_over(learner1, learner2):
	tmpLearner = clone(learner1)
	for i in range(0, len(learner1.values)):
		tmp = np.random.rand()
		if(tmp > 0.5):
			tmpLearner.values[i] = copy.copy(learner2.values[i])
	return tmpLearner
