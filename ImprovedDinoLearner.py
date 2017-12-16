import numpy as np
import copy
from scipy.special import expit

counter = 0

class ImprovedLearner2():
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
		self.fitness1 = 0
		self.fitness2 = 0
		self.fitness3 = 0
		self.round = 0
	@property
	def hd_node(self):
		return (self.values[0] * self.height) + (self.values[1] * self.dist) + self.values[2]
	@property
	def hd2_node(self):
		return (self.values[3] * self.height) + (self.values[4] * self.dist) + self.values[5]

	@property
	def hidden_node1(self):
		return (self.values[6] * self.hd_node) + (self.values[7] * self.hd2_node) + self.values[8]
	
	@property
	def hidden_node2(self):
		return (self.values[9] * self.hd_node) + (self.values[10] * self.hd2_node) + self.values[11]

	@property
	def hidden1_node1(self):
		return (self.values[12] * self.hidden_node1) + (self.values[13] * self.hidden_node2) + self.values[14]

	@property
	def hidden1_node2(self):
		return (self.values[15] * self.hidden_node1) + (self.values[16] * self.hidden_node2) + self.values[17]
	
	@property
	def output_no_land(self):
		return self.values[18] * self.no_land + self.values[19]

	@property
	def output_node1(self):
		return (self.values[20] * self.hidden1_node1) + (self.values[21] * self.hidden1_node2) + self.values[22]

	@property
	def output_node2(self):
		return (self.values[23] * self.hidden1_node1) + (self.values[24] * self.hidden1_node2) + self.values[25]
	@property
	def outputValue(self):
		#return expit(((self.values[26] * self.output_node1) + (self.values[27] * self.output_node2) + (self.values[28] * self.output_no_land) + self.values[29]) * 8)
		#return -1 * self.values[0] * self.dist + 1300#self.values[1]
		return (self.values[26] * self.output_node1) + (self.values[27] * self.output_node2) + (self.values[28] * self.output_no_land) + self.values[29]

	def output(self, distance, height, no_land_obs):
		self.height = height
		self.dist = distance
		self.no_land = no_land_obs
		return self.outputValue
	def randomize(self):
		for i in range(0, len(self.values)):
			self.values[i] = np.random.rand()

class Genetic():
    def __init__(self, population, index_size, evaluateFunc, **kwargs):
        self.fitnessValues = np.array([], dtype=float)
        self.weightArrays = []
        self.evaluateFunc = evaluateFunc
        self.population = population
        self.index_size = index_size
        while(len(self.weightArrays) < population):
            self.weightArrays.append(np.random.uniform(-1, 1, index_size))

    def evaluate(self, weights):
        self.fitnessValues = np.concatenate([self.fitnessValues, self.evaluateFunc(weights)])

    def mutate(self, weights, prob):
        temp = copy.copy(weights)
        for i in temp:
            i = i + np.random.uniform(-1.0, 1.0) * 0.1
            '''
            if(np.random.random() < prob):
                i = i + np.random.uniform(-1.0, 1.0) * 0.1
            '''
        return temp

    def crossover(self, weight1, weight2, prob):
        tempWeights = []
        for i in range(len(weight1)):
            tempWeights.append((weight1[i] + weight2[i]) / 2)
        return tempWeights

    def learn(self, rounds):
        roundNum = 0
        while(roundNum < rounds):
            for i in range(self.population):
                self.evaluate(self.weightArrays[i])
            best = self.fitnessValues.argsort()[-3:]
            print(max(self.fitnessValues))
            tempWeights = []
            tempFitness = []
            for i in range(len(best)):
                tempWeights.append(copy.copy(self.weightArrays[i]))
                tempFitness.extend(self.fitnessValues.tolist())
            self.weightArrays = []
            self.fitnessValues = np.array([], dtype=float)
            self.weightArrays.extend(tempWeights)
            self.fitnessValues = np.concatenate([self.fitnessValues, tempFitness])
            while(len(self.weightArrays) < self.population * 0.5):
                self.weightArrays.append(self.mutate(tempWeights[np.random.choice(range(len(tempWeights)))], 0.2))
            while(len(self.weightArrays) < self.population * 0.75):
                self.weightArrays.append(self.crossover(tempWeights[np.random.choice(range(len(tempWeights)))], tempWeights[np.random.choice(range(len(tempWeights)))], 0.5))
            while(len(self.weightArrays) < self.population):
                self.weightArrays.append(np.random.uniform(-1, 1, self.index_size))
            
            roundNum += 1
                



class Learner():
    def __init__(self, num_inputs, params, **kwargs):
        self.network = []
        self.network.append(InputLayer(num_inputs, params[0][0], params[0][1]))
        self.num_inputs = num_inputs
        self.params = params
        index = 1
        while index < len(params):
            if(index == len(params) - 1):
                self.network.append(OutputLayer(params[index - 1][0], params[index][0], params[index][1], self.network[-1]))
            else:
                self.network.append(HiddenLayer(params[index - 1][0], params[index][0], params[index][1], self.network[-1]))
            index += 1

        return super().__init__(**kwargs)

    def set_weights(self, weights):
        #print(len(self.network), len(self.params))
        full_array = weights
        self.network[0].set_weights(full_array[:self.num_inputs * len(self.network[0].layer) + len(self.network[0].layer)])
        full_array = full_array[self.num_inputs * len(self.network[0].layer) + len(self.network[0].layer):]
        index = 1
        while index < len(self.network):
            #print(len(self.network[index - 1].layer) * len(self.network[index].layer) + len(self.network[index].layer))
            self.network[index].set_weights(full_array[:len(self.network[index - 1].layer) * len(self.network[index].layer) + len(self.network[index].layer)])
            full_array = full_array[len(self.network[index - 1].layer) * len(self.network[index].layer) + len(self.network[index].layer):]
            index += 1

    def output(self, input):
        return self.network[-1].output(input)

    def weight_size(self):
        num = self.num_inputs * self.params[0][0] + self.params[0][0]
        index = 1
        while index < len(self.params):
            num += self.params[index - 1][0] * self.params[index][0] + self.params[index][0]
            index += 1
        return num

class Layer():
    def __init__(self, inputs, nodes, activation, **kwargs):
        self.layer = []
        self.weights = []
        self.bias = []
        self.inputs = inputs
        while len(self.layer) < nodes:
            self.layer.append(Node(inputs, activation))
            self.weights.extend(self.layer[-1].weights)
            self.bias.append(self.layer[-1].bias)

        super().__init__(**kwargs)

    def set_weights(self, weights):
        weights_array = np.array_split(weights, len(self.layer))
        index = 0
        #print(weights_array)
        for i in self.layer:
            i.set_weights(weights_array[index][:self.inputs])
            i.set_bias(weights_array[index][self.inputs])

class InputLayer(Layer):
    def __init__(self, inputs, nodes, activation, **kwargs):
        super().__init__(inputs, nodes, activation, **kwargs)

    def output(self, input):
        output_array = []
        for i in self.layer:
            output_array.append(i.output(input))
        return output_array

class HiddenLayer(Layer):
    def __init__(self, inputs, nodes, activation, layer, **kwargs): 
        super().__init__(inputs, nodes, activation, **kwargs)
        self.prev_layer = layer

    def output(self, input):
        output_array = []
        input_array = self.prev_layer.output(input)
        for i in self.layer:
            output_array.append(i.output(input_array))
        return output_array

class OutputLayer(Layer):
    def __init__(self, inputs, nodes, activation, layer, **kwargs):
        self.prev_layer = layer
        super().__init__(inputs, nodes, activation, **kwargs)

    def output(self, input):
        output = 0.0
        input_array = self.prev_layer.output(input)
        for i in self.layer:
            output += i.output(input_array)
        return output


class Node():
    def __init__(self, inputs, activation, **kwargs):
        global counter
        counter += 1
        self.weights = np.random.uniform(-1, 1, inputs)
        self.bias = np.random.uniform(-1, 1)
        self.act = activation
        return super().__init__(**kwargs)
    
    def set_weights(self, input):
        self.weights = input

    def get_weights(self):
        return self.weights

    def set_bias(self, input):
        self.bias = input

    def get_bias(self):
        return self.bias

    def outputValue(self, input):
        return np.dot(input, self.weights) + self.bias

    def output(self, input):
        if self.act == "tanh":
            return np.tanh(self.outputValue(input))
        elif self.act == "sig":
            return expit(self.outputValue(input))
        elif self.act == "none":
            return self.outputValue(input)
        else:
            return -666
def clone(learner):
    learn = Learner(learner.num_inputs, learner.params)
    learn.set_weights(copy.copy(learner.get_weights()))
    learn.set_bias(copy.copy(learner.get_bias()))
    return learn

def mutate(learner1):
	tmpLearner = clone(learner1)
	'''
	tmp = np.random.rand()
	tmpLearner.values[0] = learner1.values[0]  * (np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5)
	'''
	for i in range(0, len(learner1.values)):
		tmp = np.random.rand()
		if(tmp < 0.1):
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

'''
def cross_over(learner1, learner2):
	tmpLearner = clone(learner1)
	for i in range(0, len(learner1.weights)):
		tmp = np.random.rand()
		if(tmp > 0.8):
			tmpLearner.weights[i] = copy.copy(learner2.weights[i])
    if(np.random.rand() > 0.8):
        tmpLearner.bias = copy.copy(learner2.bias)
    if(np.random.rand() > 0.8):
        tmpLearner.weight_bias = copy.copy(learner2.weight_bias)
	return tmpLearner
'''