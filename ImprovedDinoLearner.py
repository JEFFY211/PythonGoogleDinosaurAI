import numpy as np
import copy

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
		]	
		self.height = 0
		self.dist = 0
		self.no_land = 0
		self.number = 0
		self.fitness = 0
	@property
	def hd_node(self):
		return (self.values[0] * self.height + self.values[1]) + (self.values[2] * self.dist + self.values[3])
	@property
	def hd2_node(self):
		return (self.values[4] * self.height + self.values[5]) + (self.values[6] * self.dist + self.values[7])

	@property
	def hidden_node1(self):
		return (self.values[8] * self.hd_node + self.values[9]) + (self.values[10] * self.hd2_node + self.values[11])
	
	@property
	def hidden_node2(self):
		return (self.values[12] * self.hd_node + self.values[13]) + (self.values[14] * self.hd2_node + self.values[15])

	@property
	def hidden1_node1(self):
		return (self.values[16] * self.hidden_node1 + self.values[17]) + (self.values[18] * self.hidden_node2 + self.values[19])

	@property
	def hidden1_node2(self):
		return (self.values[20] * self.hidden_node1 + self.values[21]) + (self.values[22] * self.hidden_node2 + self.values[23])
	
	@property
	def output_no_land(self):
		return (self.values[24] * self.no_land + self.values[25])

	@property
	def output_node1(self):
		return (self.values[26] * self.hidden1_node1 + self.values[27]) + (self.values[28] * self.hidden1_node2 + self.values[29])

	@property
	def output_node2(self):
		return (self.values[30] * self.hidden1_node1 + self.values[31]) + (self.values[32] * self.hidden1_node2 + self.values[33])
	@property
	def outputValue(self):
		return (self.values[34] * self.output_node1 + self.values[35]) + (self.values[36] * self.output_node2 + self.values[37]) + (self.values[38] * self.output_no_land + self.values[39])
		
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
		'''
		if(tmp < 0.2):
			tmpLearner.values[i] = learner1.values[i]  * (np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5)
		'''
		if(tmp > 0.8):
			tmpLearner.values[i] = learner1.values[i] * (0.75 + (np.random.rand()/2))
			#print(learner1.values[i] * (0.5 + np.random.rand()))
		elif(tmp < 0.2):
			tmpLearner.values[i] = learner1.values[i] * (-1 * (np.random.rand()/2) - 0.75)
			#print(learner1.values[i] * (-1 * np.random.rand() - 0.5))
	return tmpLearner

def cross_over(learner1, learner2):
	tmpLearner = clone(learner1)
	for i in range(0, len(learner1.values)):
		tmp = np.random.rand()
		if(tmp > 0.8):
			tmpLearner.values[i] = copy.copy(learner2.values[i])
	return tmpLearner
