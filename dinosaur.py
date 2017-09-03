import uinput
from mss import mss
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import heapq
import copy
import random
from pyvirtualdisplay import Display
from pyvirtualdisplay.smartdisplay import SmartDisplay
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import ImprovedDinoLearner
from multiprocessing.dummy import Pool as ThreadPool
import Queue
import csv
import sys

f = open('dinosaurLog.csv', 'wt')
learner_q = Queue.Queue()
tested_q = Queue.Queue()
display = Display(visible=1, size=(3840, 2160))
display.start()

dinosPerGeneration = 256
eliteDinos = 8 
tournaments = 256/8
truncatedSelection = False
tournamentSelection = True
def learning_func(position):
	profile = webdriver.FirefoxProfile()
	profile.set_preference('webdriver.load.strategy', 'unstable')
	if(position == 0):
		writer = csv.writer(f)
		#writer.writerow(('Generation', 'Fitness'))
		learning = True
		generation = 0
		for i in range(0, dinosPerGeneration - 1):
			neural = ImprovedDinoLearner.ImprovedLearner()
			neural.randomize()
			learner_q.put(neural)
		while learning:
			print("waiting for dinos")
			learner_q.join()
			genome = []
			fitness = []
			i = 0
			fitnessTotal = 0
			highestFitness = 0
			print("Generation: " + str(generation))
			while(not tested_q.empty()):
				genome.append(tested_q.get())
				tested_q.task_done()
				fitness.append(genome[i].fitness)
				if(genome[i].fitness > highestFitness):
					highestFitness = genome[i].fitness
				fitnessTotal += genome[i].fitness
				i = i + 1
				time.sleep(.1) 
			print("HighestFitness: " + str(highestFitness))
			writer.writerow((generation, fitnessTotal, highestFitness))
			i = 0
			if(truncatedSelection and not tournamentSelection):
				print("TRUNCATED SELECTION")
				while(len(fitness) > eliteDinos):
					mostFit = heapq.nlargest(eliteDinos, fitness)
					mostFitIndex1 = fitness.index(mostFit[0])
					mostFitIndex2 = fitness.index(mostFit[1])
					mostFitIndex3 = fitness.index(mostFit[2])
					mostFitIndex4 = fitness.index(mostFit[3])
					mostFitindex5 = fitness.index(mostFit[4])
					mostFitIndex6 = fitness.index(mostFit[5])
					mostFitIndex7 = fitness.index(mostFit[6])
					mostFitIndex8 = fitness.index(mostFit[7])
					#mostFitIndex9 = fitness.index(mostFit[8])
					#mostFitIndex10 = fitness.index(mostFit[9])
					#mostFitIndex11 = fitness.index(mostFit[10])
					#mostFitIndex12 = fitness.index(mostFit[11])
					#mostFitindex13 = fitness.index(mostFit[12])
					#mostFitIndex14 = fitness.index(mostFit[13])
					#mostFitIndex15 = fitness.index(mostFit[14])
					#mostFitIndex16 = fitness.index(mostFit[15])
					if(i is not mostFitIndex1 and i is not mostFitIndex2 ):#and i is not mostFitIndex3 and i is not mostFitIndex4):
						del genome[i]
						del fitness[i]
						i = 0
					else:
						if(i == len(genome) - 1):
							i = 0
						else:
							i = i + 1
			elif(tournamentSelection and not truncatedSelection):
				print("TOURNAMENT SELECTION")
				tournament = np.array_split(np.array(genome), tournaments)
				tournamentFitness = np.array_split(np.array(fitness), tournaments)
				genome = []
				fitness = []
				for j in range(0, tournaments):
					print("STARTING TOURNAMENT")
					while(len(tournamentFitness[j]) > 2):
						mostFit = heapq.nlargest(2, tournamentFitness[j])
						mostFitIndex1 = np.where(tournamentFitness[j] == mostFit[0])
						mostFitIndex2 = np.where(tournamentFitness[j] == mostFit[1])
						if(i is not mostFitIndex1[0][0] and i is not mostFitIndex2[0][0]):
							tournamentFitness[j] = np.delete(tournamentFitness[j], [i])
							tournament[j] = np.delete(tournament[j], [i])
							i = 0
						else:
							if(i == len(genome) - 1):
								print("RESETTING")
								i = 0
							else:
								i = i + 1
					genome.extend(tournament[j].tolist())
					fitness.extend(tournamentFitness[j].tolist())
			else:
				print("FATAL ERROR: CANNOT TOURNAMENT AND TRUNCATE!")
				break;
			newGenome = []
			opStart = time.time()
			for i in range(0, len(genome)):
				newGenome.append(ImprovedDinoLearner.clone(genome[i]))
			#print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < dinosPerGeneration * 0.4):
				genA = random.choice(newGenome)
				genB = random.choice(newGenome)
				crossed = ImprovedDinoLearner.cross_over(genA, genB)
				genome.append(crossed)
			#print("Time taken: " + str(time.time() - opStart))
			newGenome = []
			for i in range(0, len(genome)):
				newGenome.append(ImprovedDinoLearner.clone(genome[i]))
			#print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < dinosPerGeneration * 0.66):
				genA = random.choice(newGenome)
				genB = random.choice(newGenome)
				crossed = ImprovedDinoLearner.cross_over(genA, genB)
				mutated = ImprovedDinoLearner.mutate(crossed)
				genome.append(mutated)
			opStart = time.time()
			newGenome = []
			for i in range(0, len(genome)):
				newGenome.append(ImprovedDinoLearner.clone(genome[i]))
			#print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < dinosPerGeneration * (5/6)):
				genA = random.choice(newGenome)
				genB = random.choice(newGenome)
				mutated = ImprovedDinoLearner.mutate(genA)
				mutated1 = ImprovedDinoLearner.mutate(genB)
				genome.append(mutated)
				genome.append(mutated1)
			#print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < dinosPerGeneration):
				newBlood = ImprovedDinoLearner.ImprovedLearner()
				newBlood.randomize()
				genome.append(newBlood)
			random.shuffle(genome)
			for i in genome:
				i.fitness = i.fitness * 0.5 #Decay to fitness = lower lucky ones' fitness over time
				learner_q.put(i)
			generation = generation + 1
	elif(position == 1):
		browser = webdriver.Firefox(profile)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		mon = {'top':50, 'left':0, 'width':1275, 'height':360}#3840
		sct = mss()
	elif(position == 2):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(1300, 0)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':50, 'left':1300, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 3):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(0, 450)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':500, 'left':0, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 4):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(1300, 450)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':500, 'left':1300, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 5):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(0, 950)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':1000, 'left':0, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 6):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(1300, 950)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':1000, 'left':1300, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 7):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(0, 1350)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':1400, 'left':0, 'width':1275, 'height':360}
		sct = mss()
	elif(position == 8):
		browser = webdriver.Firefox(profile)
		browser.set_window_position(1300, 1350)
		browser.set_page_load_timeout(15)
		browser.set_window_size(1300, 400)
		#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
		browser.get('http://wayou.github.io/t-rex-runner/')
		#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
		browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
		
		#mon = {'top':50, 'left':0, 'width':3840, 'height':360}
		mon = {'top':1400, 'left':1300, 'width':1275, 'height':360}
		sct = mss()	
	with uinput.Device([uinput.KEY_SPACE, uinput.KEY_UP, uinput.KEY_DOWN]) as device, tf.Session() as sess:
		print("Starting ML program")
		prevGameOver = False
		printed = False	
		browser.find_element_by_tag_name("body").send_keys(Keys.UP)
		while not prevGameOver:
			frame = np.array(sct.grab(mon))
			#cv2.imshow("frame", frame)
			#cv2.waitKey()
			frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			thresh = np.full_like(frameGrey, 0)
			cv2.inRange(frameGrey, 71, 98, thresh)
			img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(thresh, contours, -1, (127,255,0), 3)
			x, y, w, h = 0, 0, 0, 0
			for i in contours:
				area = cv2.contourArea(i)
				if(area > 3000 and area < 4105):
					x, y, w, h = cv2.boundingRect(i)
					print(h, w)
					#if(h > 113 and h < 145 and w > 150 and w < 160):
					if(h > 60 and h < 65 and w > 65 and w < 75):
						print("Game Over verified - Starting Game")
						time.sleep(1.0)
						prevGameOver = True
		success = False
		while not success:
			startTime = 0
			firstObjectPos = -2700
			lastFirstObjectPos = 2700
			firstObjectHeight = 0
			firstObjectWidth = 0
			firstObjectSpeed = 0
			loopStart = 0
			loopEnd = 0
			floorHeight = 0 #TODO - Add floor height for birds!
			gameOver = True
			jumpedOverCactus = False
			jumped = False
			ducked = False
			jumps = 0
			while gameOver:
				frame = np.array(sct.grab(mon))
				frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				thresh = np.full_like(frameGrey, 0)
				cv2.inRange(frameGrey, 71, 98, thresh)
				img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				x, y, w, h = 0, 0, 0, 0
				for i in contours:
					area = cv2.contourArea(i)
					#if(area > 20000): #Game Over!
					if(area > 3000 and area < 4105):
						x, y, w, h = cv2.boundingRect(i)
						#if(h > 113 and h < 145 and w > 150 and w < 160):
						if(h > 60 and h < 65 and w > 65 and w < 75):
							dino = learner_q.get()
							#Record initial time and start game
							browser.find_element_by_tag_name("body").send_keys(Keys.UP)
							#device.emit_click(uinput.KEY_UP)
							time.sleep(2.0)
							startTime = time.time()
							gameOver = False
							break
			
			cactiJumped = 0
			jumpTime = 0
			
			while not gameOver:
				lastObjectPos = firstObjectPos
				noLandObs = 1
				loopStart = time.time()
				frame = np.array(sct.grab(mon))

				frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#height, width = frameGrey.shape
				#frameHalf = frameGrey[(height / 2):(height-(height/8)), (width/6):(width-(width/6))]
				thresh = np.full_like(frameGrey, 0)
				cv2.inRange(frameGrey, 71, 98, thresh)
				img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				#cv2.drawContours(thresh, contours, -1, (127,255,0), 3)
				x, y, w, h = 0, 0, 0, 0
				for i in contours:
					area = cv2.contourArea(i)
					#if(area > 1200 and area < 4750): #Tiny Cactus
					if(area > 500 and area < 1000):
						x, y, w, h = cv2.boundingRect(i)
						#if(h > 80):
						if(h > 65 and h < 70):
							if(abs(firstObjectPos) > (x + (w / 2)) ):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
								noLandObs = 0
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
					#elif(area > 5150 and area < 10100): #Bird or Big Cactus
					elif(area > 1000 and area < 2000):
						x, y, w, h, = cv2.boundingRect(i)
						'''Bird
						if(h > 200 and h < 217):
							if(firstObjectPos > (x + (w / 2)) and (x + (w/2) > 600) and h < 600):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (155, 0, 0), 5)
						'''
						if(h > 93 and h < 98):
							if(abs(firstObjectPos) > (x + (w / 2))):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
								noLandObs = 0
					elif(area > 3000 and area < 4105):
						x, y, w, h = cv2.boundingRect(i)
						if(h > 60 and h < 65 and w > 65 and w < 75):
							time.sleep(1.0)
							gameOver = True
							lastFirstObjectPos = 1500
							firstObjectPos = 1500
							timeRun = time.time() - startTime
							if(jumped):
								jumped = False
								timeRun = timeRun + 1 - (jumps / 3)
							if(ducked):
								ducked = False
								timeRun = timeRun + 1
							dino.fitness = timeRun
							learner_q.task_done()
							tested_q.put(dino)
							print("Time lived: " + str(timeRun) + "\tJumps: " + str(jumps))
							if(timeRun > 60):
								print("Successful dino!")
								success = True	
				#distanceTraveled = lastFirstObjectPos - firstObjectPos
				#print(distanceTraveled)
				#timeSpent = time.time() - loopStart 
				#print("Time spent:" + str(timeSpent))
				#firstObjectSpeed = (distanceTraveled / timeSpent) / 100
				#print("Obj Pos:" + str(firstObjectPos)) 
				#print("Speed:" + str(firstObjectSpeed))
				#print("Width:" + str(firstObjectWidth))
				#print("Height:" + str(firstObjectHeight))
				#print("NoLandObs:" + str(noLandObs))
				#print(gameOver)
				if not gameOver:
					outputValue = dino.output(firstObjectPos, firstObjectHeight, noLandObs)
					#if(outputValue < 3000):
					#	device.emit_click(uinput.KEY_DOWN)
					#else:
					if(outputValue > 500):
						#device.emit_click(uinput.KEY_UP)	
						browser.find_element_by_tag_name("body").send_keys(Keys.UP)
						time.sleep(0.6)
						jumps = jumps + 1
						if(firstObjectPos > 0):
							jumped = True
					'''
					elif(outputValue < 1000):
						browser.find_element_by_tag_name("body").send_keys(Keys.DOWN)
						time.sleep(0.1)
						#ducked = True
					'''
					#print(outputValue)
					#print("CactiJumped:" + str(cactiJumped))
					#print("NN:" + str(neural) + "," + str(generation))
					#print("---------")
				firstObjectPos = -2700
pool = ThreadPool(5)
results = pool.map(learning_func, [0,1,2,3,4,5,6,7,8])

print("THIS SHOULDN'T RUN")
display = Display(visible=1, size=(3840, 2160))
display.start()
pool = ThreadPool(2)
results = pool.map(learning_func, [0,1,2])
learning = True
neural1 = ImprovedDinoLearner.ImprovedLearner()
neural2 = ImprovedDinoLearner.ImprovedLearner()
learner_q.put(neural1)
learner_q.put(neural2)
print("Put dinoes in queue")
while learning:
	learner_q.join()
	genome = []
	fitness = []
	i = 0
	while(tested_q.empty()):
		genome.append(tested_q.get())
		fitness.append(genome[i].fitness)
		time.sleep(.1) 
	while(len(fitness) > 4):
		print("Genome length:" + str(len(genome)))
		print("Fitness length:" + str(len(fitness)))
		mostFit = heapq.nlargest(4, fitness)
		mostFitIndex1 = fitness.index(mostFit[0])
		mostFitIndex2 = fitness.index(mostFit[1])
		mostFitIndex3 = fitness.index(mostFit[2])
		mostFitIndex4 = fitness.index(mostFit[3])
		if(i is not mostFitIndex1 and i is not mostFitIndex2 and i is not mostFitIndex3 and i is not mostFitIndex4):
			print("Culling inferior dino #" + str(i))
			del genome[i]
			del fitness[i]
			i = 0
		else:
			if(i == len(genome) - 1):
				i = 0
			else:
				i = i + 1
	print("Generating new genome")
	newGenome = []
	opStart = time.time()
	for i in range(0, len(genome)):
		newGenome.append(ImprovedDinoLearner.clone(genome[i]))
		print("Generated new genome: " + str(i))
	print("Crossing and mutating")
	print("Time taken: " + str(time.time() - opStart))
	opStart = time.time()
	while(len(genome) < 8):
		genA = random.choice(newGenome)
		genB = random.choice(newGenome)
		crossed = ImprovedDinoLearner.cross_over(genA, genB)
		mutated = ImprovedDinoLearner.mutate(crossed)
		genome.append(crossed)
		genome.append(mutated)
		fitness.append(0)
		fitness.append(0)
		print("Crossed and mutated")
	print("Time taken: " + str(time.time() - opStart))
	opStart = time.time()
	print("Regenerating genome")
	newGenome = []
	for i in range(0, len(genome)):
		newGenome.append(ImprovedDinoLearner.clone(genome[i]))
		print("Regenerated new genome: " + str(i))
	print("Time taken: " + str(time.time() - opStart))
	opStart = time.time()
	print("Just mutating")
	while(len(genome) < 10):
		genA = random.choice(newGenome)
		genB = random.choice(newGenome)
		mutated = ImprovedDinoLearner.mutate(genA)
		mutated1 = ImprovedDinoLearner.mutate(genB)
		genome.append(mutated)
		genome.append(mutated1)
		fitness.append(0)
		fitness.append(0)
		print("Mutated genome")
	print("Time taken: " + str(time.time() - opStart))
	opStart = time.time()
	while(len(genome) < 12):
		newBlood = ImprovedDinoLearner.ImprovedLearner()
		newBlood.randomize()
		genome.append(newBlood)
		fitness.append(0)
	for i in genome:
		i.fitness = 0
		learner_q.put(i)
gameoverX = 1878
gameoverY = 1356
'''
profile = webdriver.FirefoxProfile()
profile.set_preference('webdriver.load.strategy', 'unstable')
browser = webdriver.Firefox(profile)
browser.set_page_load_timeout(15)
#browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
browser.get('http://wayou.github.io/t-rex-runner/')
#browser.execute_script("document.body.style.MozTransform = 'scale(3)';")
browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
'''
#mon = {'top':780, 'left':540, 'width':2760, 'height':1010}
mon = {'top':50, 'left':0, 'width':3840, 'height':360}

sct = mss()

#create genome to store neural networks
genome = []
#Create list to store how "fit" neural networks are
fitness = []
#Determines if the program is "learning" or not at the moment
learning = False
'''
class Learner():
	def __init__(self):
		#Create inputs
		self.object_height = tf.placeholder(tf.float32, name='obj_height')
		#self.object_width = tf.placeholder(tf.float32, name='obj_width')
		#self.object_speed = tf.placeholder(tf.float32, name='obj_speed')
		self.object_dist = tf.placeholder(tf.float32, name='obj_dist')
		self.no_land_obs = tf.placeholder(tf.float32, name='no_land_obs')
		'''
		#with tf.name_scope("Input_Layer"):
'''
			self.object_width_speed_weight_s = tf.Variable([0.0], dtype=tf.float32, name='object_ws_weight_s')
			self.object_width_speed_bias_w = tf.Variable([0.0], dtype=tf.float32, name='object_ws_bias_w')
			self.object_width_speed_bias_s = tf.Variable([0.0], dtype=tf.float32, name='object_ws_bias_s')

			self.object_dist_speed_weight_d = tf.Variable([0.0], dtype=tf.float32, name='object_ds_weight_d')
			self.object_dist_speed_weight_s = tf.Variable([0.0], dtype=tf.float32, name='object_ds_weight_s')
			self.object_dist_speed_bias_d = tf.Variable([0.0], dtype=tf.float32, name='object_ds_bias_d')
			self.object_dist_speed_bias_s = tf.Variable([0.0], dtype=tf.float32, name='object_ds_bias_s')
			self.object_dist_width_weight_d  = tf.Variable([0.0], dtype=tf.float32, name='object_dw_weight_d')
			self.object_dist_width_weight_w = tf.Variable([0.0], dtype=tf.float32, name='object_dw_weight_w')
			self.object_dist_width_bias_d = tf.Variable([0.0], dtype=tf.float32, name='object_dw_bias_d')
			self.object_dist_width_bias_w = tf.Variable([0.0], dtype=tf.float32, name='object_dw_bias_w')
			'''
'''
			self.object_height_dist_weight_h = tf.Variable([0.0], dtype=tf.float32, name='object_hd_weight_h')
			self.object_height_dist_weight_d = tf.Variable([0.0], dtype=tf.float32, name='object_hd_weight_d')
			self.object_height_dist_bias_h = tf.Variable([0.0], dtype=tf.float32, name='object_hd_bias_h')
			self.object_height_dist_bias_d = tf.Variable([0.0], dtype=tf.float32, name='object_hd_bias_d')

			self.object_height_dist2_weight_h = tf.Variable([0.0], dtype=tf.float32, name='object_hd2_weight_h')
			self.object_height_dist2_weight_d = tf.Variable([0.0], dtype=tf.float32, name='object_hd2_weight_d')
			self.object_height_dist2_bias_h = tf.Variable([0.0], dtype=tf.float32, name='object_hd2_bias_h') self.object_height_dist2_bias_d = tf.Variable([0.0], dtype=tf.float32, name='object_hd2_bias_d')
			'''
'''
			self.object_height_width_node = tf.add(tf.add(tf.multiply(self.object_height_width_weight_h, self.object_height), self.object_height_width_bias_h), tf.add(tf.multiply(self.object_height_width_weight_w, self.object_width), self.object_height_width_bias_w), name='object_height_width_node')
			self.object_height_speed_node = tf.add(tf.add(tf.multiply(self.object_height_speed_weight_h, self.object_height), self.object_height_speed_bias_h), tf.add(tf.multiply(self.object_height_speed_weight_s, self.object_speed), self.object_height_speed_bias_s), name='object_height_speed_node')
			self.object_width_speed_node = tf.add(tf.add(tf.multiply(self.object_width_speed_weight_w, self.object_width), self.object_width_speed_bias_w), tf.add(tf.multiply(self.object_width_speed_weight_s, self.object_speed), self.object_width_speed_bias_s), name='object_width_speed_node')
			self.object_dist_speed_node = tf.add(tf.add(tf.multiply(self.object_dist_speed_weight_d, self.object_dist), self.object_dist_speed_bias_d), tf.add(tf.multiply(self.object_dist_speed_weight_s, self.object_speed), self.object_dist_speed_bias_s), name='object_dist_speed_node')
			self.object_dist_width_node = tf.add(tf.add(tf.multiply(self.object_dist_width_weight_d, self.object_dist), self.object_dist_width_bias_d), tf.add(tf.multiply(self.object_dist_width_weight_w, self.object_width), self.object_dist_width_bias_w), name='object_dist_width_node')
			'''
'''
			self.object_height_dist_node = tf.add(tf.add(tf.multiply(self.object_height_dist_weight_h, self.object_height), self.object_height_dist_bias_h), tf.add(tf.multiply(self.object_height_dist_weight_d, self.object_dist), self.object_height_dist_bias_d), name='object_height_dist_node')
			self.object_height_dist2_node = tf.add(tf.add(tf.multiply(self.object_height_dist2_weight_h, self.object_height), self.object_height_dist2_bias_h), tf.add(tf.multiply(self.object_height_dist2_weight_d, self.object_dist), self.object_height_dist2_bias_d), name='object_height_dist2_node')


		#Construct the hidden layer
		with tf.name_scope("Hidden_Layer"):
			'''
'''
			self.hw_ds_node1_weight_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_ds_weight_hw')
			self.hw_ds_node1_weight_ds = tf.Variable([0.0], dtype=tf.float32, name='hw_ds_weight_ds')
			self.hw_ds_node1_bias_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_ds_bias_hw')
			self.hw_ds_node1_bias_ds = tf.Variable([0.0], dtype=tf.float32, name='hw_ds_bias_ds')
			self.ds_hs_node1_weight_ds = tf.Variable([0.0], dtype=tf.float32, name='ds_hs_weight_ds')
			self.ds_hs_node1_weight_hs = tf.Variable([0.0], dtype=tf.float32, name='ds_hs_weight_hs')
			self.ds_hs_node1_bias_ds = tf.Variable([0.0], dtype=tf.float32, name='ds_hs_bias_ds')
			self.ds_hs_node1_bias_hs = tf.Variable([0.0], dtype=tf.float32, name='ds_hs_bias_hs')
			self.ds_ws_node1_weight_ds = tf.Variable([0.0], dtype=tf.float32, name='ds_ws_weight_ds')
			self.ds_ws_node1_weight_ws = tf.Variable([0.0], dtype=tf.float32, name='ds_ws_weight_ws')
			self.ds_ws_node1_bias_ds = tf.Variable([0.0], dtype=tf.float32, name='ds_ws_bias_ds')
			self.ds_ws_node1_bias_ws = tf.Variable([0.0], dtype=tf.float32, name='ds_ws_bias_ws')
			self.hd_ds_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_ds_weight_hd')
			self.hd_ds_node1_weight_ds = tf.Variable([0.0], dtype=tf.float32, name='hd_ds_weight_ds')
			self.hd_ds_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_ds_bias_hd')
			self.hd_ds_node1_bias_ds = tf.Variable([0.0], dtype=tf.float32, name='hd_ds_bias_ds')
			self.dw_ds_node1_weight_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_ds_weight_dw')
			self.dw_ds_node1_weight_ds = tf.Variable([0.0], dtype=tf.float32, name='dw_ds_weight_ds')
			self.dw_ds_node1_bias_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_ds_bias_dw')
			self.dw_ds_node1_bias_ds = tf.Variable([0.0], dtype=tf.float32, name='dw_ds_bias_ds')
#
			self.dw_hs_node1_weight_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hs_weight_dw')
			self.dw_hs_node1_weight_hs = tf.Variable([0.0], dtype=tf.float32, name='dw_hs_weight_hs')
			self.dw_hs_node1_bias_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hs_bias_dw')
			self.dw_hs_node1_bias_hs = tf.Variable([0.0], dtype=tf.float32, name='dw_hs_bias_hs')
#
			self.dw_ws_node1_weight_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_ws_weight_dw')
			self.dw_ws_node1_weight_ws = tf.Variable([0.0], dtype=tf.float32, name='dw_ws_weight_ws')
			self.dw_ws_node1_bias_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_ws_bias_dw')
			self.dw_ws_node1_bias_ws = tf.Variable([0.0], dtype=tf.float32, name='dw_ws_bias_ws')
#
			self.dw_hw_node1_weight_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hw_weight_dw')
			self.dw_hw_node1_weight_hw = tf.Variable([0.0], dtype=tf.float32, name='dw_hw_weight_hw')
			self.dw_hw_node1_bias_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hw_bias_dw')
			self.dw_hw_node1_bias_hw = tf.Variable([0.0], dtype=tf.float32, name='dw_hw_bias_hw')
#
			self.dw_hd_node1_weight_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hd_weight_dw')
			self.dw_hd_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='dw_hd_weight_hd')
			self.dw_hd_node1_bias_dw = tf.Variable([0.0], dtype=tf.float32, name='dw_hd_bias_dw')
			self.dw_hd_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='dw_hd_bias_hd')
#
			self.hd_ws_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_ws_weight_hd')
			self.hd_ws_node1_weight_ws = tf.Variable([0.0], dtype=tf.float32, name='hd_ws_weight_ws')
			self.hd_ws_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_ws_bias_hd')
			self.hd_ws_node1_bias_ws = tf.Variable([0.0], dtype=tf.float32, name='hd_ws_bias_ws')
			self.hd_hs_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hs_weight_hd')
			self.hd_hs_node1_weight_hs = tf.Variable([0.0], dtype=tf.float32, name='hd_hs_weight_hs')
			self.hd_hs_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hs_bias_hd')
			self.hd_hs_node1_bias_hs = tf.Variable([0.0], dtype=tf.float32, name='hd_hs_bias_hs')
			self.hd_hw_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hw_weight_hd')
			self.hd_hw_node1_weight_hw = tf.Variable([0.0], dtype=tf.float32, name='hd_hw_weight_hw')
			self.hd_hw_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hw_bias_hd')
			self.hd_hw_node1_bias_hw = tf.Variable([0.0], dtype=tf.float32, name='hd_hw_bias_hw')
#
			self.hw_hs_node1_weight_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_hs_weight_hw')
			self.hw_hs_node1_weight_hs = tf.Variable([0.0], dtype=tf.float32, name='hw_hs_weight_hs')
			self.hw_hs_node1_bias_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_hs_bias_hw')
			self.hw_hs_node1_bias_hs = tf.Variable([0.0], dtype=tf.float32, name='hw_hs_bias_hs')
#
			self.hw_ws_node2_weight_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_ws_weight_hw')
			self.hw_ws_node2_weight_ws = tf.Variable([0.0], dtype=tf.float32, name='hw_ws_weight_ws')
			self.hw_ws_node2_bias_hw = tf.Variable([0.0], dtype=tf.float32, name='hw_ws_bias_hw')
			self.hw_ws_node2_bias_ws = tf.Variable([0.0], dtype=tf.float32, name='hw_ws_bias_ws')
#
			self.ws_hs_node3_weight_ws = tf.Variable([0.0], dtype=tf.float32, name='ws_hs_weight_ws')
			self.ws_hs_node3_weight_hs = tf.Variable([0.0], dtype=tf.float32, name='ws_hs_weight_hs')
			self.ws_hs_node3_bias_ws = tf.Variable([0.0], dtype=tf.float32, name='ws_hs_bias_ws')
			self.ws_hs_node3_bias_hs = tf.Variable([0.0], dtype=tf.float32, name='ws_hs_bias_hs') 
#
			self.hw_ds_node4 = tf.add(tf.add(tf.multiply(self.hw_ds_node1_weight_hw, self.object_height_width_node), self.hw_ds_node1_bias_hw), tf.add(tf.multiply(self.hw_ds_node1_weight_ds, self.object_dist_speed_node), self.hw_ds_node1_bias_ds), name='hw_ds_node1')

			self.ds_hs_node5 = tf.add(tf.add(tf.multiply(self.ds_hs_node1_weight_ds, self.object_dist_speed_node), self.ds_hs_node1_bias_ds), tf.add(tf.multiply(self.ds_hs_node1_weight_hs, self.object_height_speed_node), self.ds_hs_node1_bias_hs), name='ds_hs_node1')

			self.ds_ws_node6 = tf.add(tf.add(tf.multiply(self.ds_ws_node1_weight_ds, self.object_dist_speed_node), self.ds_ws_node1_bias_ds), tf.add(tf.multiply(self.ds_ws_node1_weight_ws, self.object_width_speed_node), self.ds_ws_node1_bias_ws), name='ds_ws_node1')

			self.hd_ds_node7 = tf.add(tf.add(tf.multiply(self.hd_ds_node1_weight_hd, self.object_height_dist_node), self.hd_ds_node1_bias_hd), tf.add(tf.multiply(self.hd_ds_node1_weight_ds, self.object_dist_speed_node), self.hd_ds_node1_bias_ds), name='hd_ds_node1')

			self.dw_ds_node8 = tf.add(tf.add(tf.multiply(self.dw_ds_node1_weight_dw, self.object_dist_width_node), self.dw_ds_node1_bias_dw), tf.add(tf.multiply(self.dw_ds_node1_weight_ds, self.object_dist_speed_node), self.dw_ds_node1_bias_ds), name='dw_ds_node1')


			self.dw_hs_node9 = tf.add(tf.add(tf.multiply(self.dw_hs_node1_weight_dw, self.object_dist_width_node), self.dw_hs_node1_bias_dw), tf.add(tf.multiply(self.dw_hs_node1_weight_hs, self.object_height_speed_node), self.dw_hs_node1_bias_hs), name='dw_hs_node1')


			self.dw_hd_node10 = tf.add(tf.add(tf.multiply(self.dw_hd_node1_weight_dw, self.object_dist_width_node), self.dw_hd_node1_bias_dw), tf.add(tf.multiply(self.dw_hd_node1_weight_hd, self.object_height_dist_node), self.dw_hd_node1_bias_hd), name='dw_hd_node1')


			self.dw_hw_node11 = tf.add(tf.add(tf.multiply(self.dw_hw_node1_weight_dw, self.object_dist_width_node), self.dw_hw_node1_bias_dw), tf.add(tf.multiply(self.dw_hw_node1_weight_hw, self.object_height_width_node), self.dw_hw_node1_bias_hw), name='dw_hw_node1')


			self.hd_ws_node12 = tf.add(tf.add(tf.multiply(self.hd_ws_node1_weight_hd, self.object_height_dist_node), self.hd_ws_node1_bias_hd), tf.add(tf.multiply(self.hd_ws_node1_weight_ws, self.object_width_speed_node), self.hd_ws_node1_bias_ws), name='hd_ws_node1')

			self.hd_hs_node13 = tf.add(tf.add(tf.multiply(self.hd_hs_node1_weight_hd, self.object_height_dist_node), self.hd_hs_node1_bias_hd), tf.add(tf.multiply(self.hd_hs_node1_weight_hs, self.object_height_speed_node), self.hd_hs_node1_bias_hs), name='hd_hs_node1')

			self.hd_hw_node14 = tf.add(tf.add(tf.multiply(self.hd_hw_node1_weight_hd, self.object_height_dist_node), self.hd_hw_node1_bias_hd), tf.add(tf.multiply(self.hd_hw_node1_weight_hw, self.object_height_width_node), self.hd_hw_node1_bias_hw), name='hd_hw_node1')

			self.hw_ws_node2 = tf.add(tf.add(tf.multiply(self.hw_ws_node2_weight_hw, self.object_height_width_node), self.hw_ws_node2_bias_hw), tf.add(tf.multiply(self.hw_ws_node2_weight_ws, self.object_width_speed_node), self.hw_ws_node2_bias_ws), name='hw_ws_node2')

			self.ws_hs_node3 = tf.add(tf.add(tf.multiply(self.ws_hs_node3_weight_ws, self.object_width_speed_node), self.ws_hs_node3_bias_ws), tf.add(tf.multiply(self.ws_hs_node3_weight_hs, self.object_height_speed_node), self.ws_hs_node3_bias_hs), name='ws_hs_node3')

			self.hw_hs_node1 = tf.add(tf.add(tf.multiply(self.hw_hs_node1_weight_hw, self.object_height_width_node), self.hw_hs_node1_bias_hw), tf.add(tf.multiply(self.hw_hs_node1_weight_hs, self.object_height_speed_node), self.hw_hs_node1_bias_hs), name='hw_hs_node1')
			'''
'''
			self.hd_hd2_node1_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node1_weight_hd')
			self.hd_hd2_node1_weight_hd2 = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node1_weight_hd2')
			self.hd_hd2_node1_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node1_bias_hd')
			self.hd_hd2_node1_bias_hd2 = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node1_bias_hd2')
			
			self.hd_hd2_node2_weight_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node2_weight_hd')
			self.hd_hd2_node2_weight_hd2 = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node2_weight_hd2')
			self.hd_hd2_node2_bias_hd = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node2_bias_hd')
			self.hd_hd2_node2_bias_hd2 = tf.Variable([0.0], dtype=tf.float32, name='hd_hd2_node2_bias_hd2')

			self.hd_hd2_node1 = tf.add(tf.add(tf.multiply(self.hd_hd2_node1_weight_hd, self.object_height_dist_node), self.hd_hd2_node1_bias_hd), tf.add(tf.multiply(self.hd_hd2_node1_weight_hd2, self.object_height_dist2_node), self.hd_hd2_node1_bias_hd2))

			self.hd_hd2_node2 = tf.add(tf.add(tf.multiply(self.hd_hd2_node2_weight_hd, self.object_height_dist_node), self.hd_hd2_node2_bias_hd), tf.add(tf.multiply(self.hd_hd2_node2_weight_hd2, self.object_height_dist2_node), self.hd_hd2_node2_bias_hd2))

			self.hidden1_hidden2_weight_h1 = tf.Variable([0.0], dtype=tf.float32, name='hidden1_hidden2_weight_h1')
			self.hidden1_hidden2_weight_h2 = tf.Variable([0.0], dtype=tf.float32, name='hidden1_hidden2_weight_h2')
			self.hidden1_hidden2_bias_h1 = tf.Variable([0.0], dtype=tf.float32, name='hidden1_hidden2_bias_h1')
			self.hidden1_hidden2_bias_h2 = tf.Variable([0.0], dtype=tf.float32, name='hidden1_hidden2_bias_h2')

			self.hidden1_2_hidden2_weight_h1 = tf.Variable([0.0], dtype=tf.float32, name="2_hidden1_hidden2_weight_h1")
			self.hidden1_2_hidden2_weight_h2 = tf.Variable([0.0], dtype=tf.float32, name='2_hidden1_hidden2_weight_h2')
			self.hidden1_2_hidden2_bias_h1 = tf.Variable([0.0], dtype=tf.float32, name='2_hidden1_hidden2_bias_h1')
			self.hidden1_2_hidden2_bias_h2 = tf.Variable([0.0], dtype=tf.float32, name='2_hidden1_hidden2_bias_h2')

			self.hidden_node1 = tf.add(tf.add(tf.multiply(self.hidden1_hidden2_weight_h1, self.hd_hd2_node1), self.hidden1_hidden2_bias_h1), tf.add(tf.multiply(self.hidden1_hidden2_weight_h2, self.hd_hd2_node1), self.hidden1_hidden2_bias_h2), name='hidden_node1')

			self.hidden_node2 = tf.add(tf.add(tf.multiply(self.hidden1_2_hidden2_weight_h1, self.hd_hd2_node1), self.hidden1_2_hidden2_bias_h1), tf.add(tf.multiply(self.hidden1_2_hidden2_weight_h2, self.hd_hd2_node1), self.hidden1_2_hidden2_bias_h2), name='hidden_node2')
		#Construct output neuron
		with tf.name_scope("Output_Layer"):
			'''
'''
			self.output_node1_weight = tf.Variable([0.0], dtype=tf.float32, name='node1_weight')
			self.output_node1_bias = tf.Variable([0.0], dtype=tf.float32, name='node1_bias')

			self.output_node2_weight = tf.Variable([0.0], dtype=tf.float32, name='node2_weight')
			self.output_node2_bias = tf.Variable([0.0], dtype=tf.float32, name='node2_bias')

			self.output_node3_weight = tf.Variable([0.0], dtype=tf.float32, name='node3_weight')
			self.output_node3_bias = tf.Variable([0.0], dtype=tf.float32, name='node3_bias')

			self.output_node4_weight = tf.Variable([0.0], dtype=tf.float32, name='node4_weight')
			self.output_node4_bias = tf.Variable([0.0], dtype=tf.float32, name='node4_bias')
			self.output_node5_weight = tf.Variable([0.0], dtype=tf.float32, name='node5_weight')
			self.output_node5_bias = tf.Variable([0.0], dtype=tf.float32, name='node5_bias')
			self.output_node6_weight = tf.Variable([0.0], dtype=tf.float32, name='node6_weight')
			self.output_node6_bias = tf.Variable([0.0], dtype=tf.float32, name='node6_bias')
			self.output_node7_weight = tf.Variable([0.0], dtype=tf.float32, name='node7_weight')
			self.output_node7_bias = tf.Variable([0.0], dtype=tf.float32, name='node7_bias')
			self.output_node8_weight = tf.Variable([0.0], dtype=tf.float32, name='node8_weight')
			self.output_node8_bias = tf.Variable([0.0], dtype=tf.float32, name='node8_bias')

			self.output_node9_weight = tf.Variable([0.0], dtype=tf.float32, name='node9_weight')
			self.output_node9_bias = tf.Variable([0.0], dtype=tf.float32, name='node9_bias')

			self.output_node10_weight = tf.Variable([0.0], dtype=tf.float32, name='node10_weight')
			self.output_node10_bias = tf.Variable([0.0], dtype=tf.float32, name='node1_bias')

			self.output_node11_weight = tf.Variable([0.0], dtype=tf.float32, name='node11_weight')
			self.output_node11_bias = tf.Variable([0.0], dtype=tf.float32, name='node11_bias')

			self.output_node12_weight = tf.Variable([0.0], dtype=tf.float32, name='node12_weight')
			self.output_node12_bias = tf.Variable([0.0], dtype=tf.float32, name='node12_bias')
			self.output_node13_weight = tf.Variable([0.0], dtype=tf.float32, name='node13_weight')
			self.output_node13_bias = tf.Variable([0.0], dtype=tf.float32, name='node13_bias')
			self.output_node14_weight = tf.Variable([0.0], dtype=tf.float32, name='node14_weight')
			self.output_node14_bias = tf.Variable([0.0], dtype=tf.float32, name='node14_bias')
			'''			
'''
			self.output_no_land_weight = tf.Variable([0.0], dtype=tf.float32, name='no_land_obs_weight')
			self.output_no_land_bias = tf.Variable([0.0], dtype=tf.float32, name='no_land_obs_bias')
			
			self.output_node1_weight = tf.Variable([0.0], dtype=tf.float32, name='node1_weight')
			self.output_node1_bias = tf.Variable([0.0], dtype=tf.float32, name='node1_bias')

			self.output_node2_weight = tf.Variable([0.0], dtype=tf.float32, name='node2_weight')
			self.output_node2_bias = tf.Variable([0.0], dtype=tf.float32, name='node2_bias')

			#self.output1 = tf.add(tf.add(tf.add(tf.add(tf.multiply(self.output_node1_weight, self.hw_hs_node1), self.output_node1_bias), tf.add(tf.multiply(self.output_node2_weight, self.hw_ws_node2), self.output_node2_bias)), tf.add(tf.multiply(self.output_node3_weight, self.ws_hs_node3), self.output_node3_bias)), tf.add(tf.multiply(self.output_node4_weight, self.hw_ds_node4), self.output_node4_bias))

			#self.output2 = tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.add(tf.multiply(self.output_node5_weight, self.ds_hs_node5), self.output_node5_bias), tf.add(tf.multiply(self.output_node6_weight, self.ds_ws_node6), self.output_node6_bias)), tf.add(tf.multiply(self.output_node7_weight, self.hd_ds_node7), self.output_node7_bias)), tf.add(tf.multiply(self.output_node8_weight, self.dw_ds_node8), self.output_node8_bias)), tf.add(tf.multiply(self.output_node9_weight, self.dw_hs_node9), self.output_node9_bias)), tf.add(tf.multiply(self.output_node10_weight, self.dw_hd_node10), self.output_node10_bias)), tf.add(tf.multiply(self.output_node11_weight, self.dw_hw_node11), self.output_node11_bias)), tf.add(tf.multiply(self.output_node12_weight, self.hd_ws_node12), self.output_node12_bias)), tf.add(tf.multiply(self.output_node13_weight, self.hd_hs_node13), self.output_node13_bias)), tf.add(tf.multiply(self.output_node14_weight, self.hd_hw_node14), self.output_node14_bias))

			#self.output = tf.add(self.output1, self.output2, name='output')
			#self.output = tf.add(tf.add(tf.add(tf.multiply(self.output_node5_weight, self.ds_hs_node5), self.output_node5_bias), tf.add(tf.multiply(self.output_node7_weight, self.hd_ds_node7), self.output_node7_bias)), tf.add(tf.multiply(self.output_node13_weight, self.hd_hs_node13), self.output_node13_bias), name="output")
			self.output = tf.add(tf.add(tf.add(tf.multiply(self.output_node2_weight, self.hidden_node2), self.output_node2_bias), tf.add(tf.multiply(self.output_node2_weight, self.hidden_node2), self.output_node2_bias)), tf.add(tf.multiply(self.output_no_land_weight, self.no_land_obs), self.output_no_land_bias), name='output')
			
		
		#Put the Variables into a list so that it is easy to iterate through
		self.weight = [
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
				0
			]
		self.weights = [
				#self.object_height_speed_weight_h,
				#self.object_height_speed_weight_s,
				self.object_height_dist_weight_h,
				self.object_height_dist_weight_d,
				self.object_height_dist2_weight_h,
				self.object_height_dist2_weight_d,
				#self.object_dist_speed_weight_d,
				#self.object_dist_speed_weight_s,
				#self.ds_hs_node1_weight_ds,
				#self.ds_hs_node1_weight_hs,
				#self.hd_ds_node1_weight_hd,
				#self.hd_ds_node1_weight_ds,
				#self.hd_hs_node1_weight_hd,
				#self.hd_hs_node1_weight_hs,
				#self.output_node5_weight,
				#self.output_node7_weight,
				#self.output_node13_weight,
				self.hd_hd2_node1_weight_hd,
				self.hd_hd2_node1_weight_hd2,
				self.hd_hd2_node2_weight_hd,
				self.hd_hd2_node2_weight_hd2,
				self.hidden1_hidden2_weight_h1,
				self.hidden1_hidden2_weight_h2,
				self.hidden1_2_hidden2_weight_h1,
				self.hidden1_2_hidden2_weight_h2,
				self.output_no_land_weight,
				self.output_node1_weight,
				self.output_node2_weight
				]

		'''
'''
		self.weights = [
				self.object_height_width_weight_h, 
				self.object_height_width_weight_w,
				self.object_height_speed_weight_h,
				self.object_height_speed_weight_s,
				self.object_width_speed_weight_w,
				self.object_width_speed_weight_s,
				self.object_dist_speed_weight_d,
				self.object_dist_speed_weight_s,
				self.object_dist_width_weight_d,
				self.object_dist_width_weight_w,
				self.object_height_dist_weight_h,
				self.object_height_dist_weight_d,
				self.hw_hs_node1_weight_hw,
				self.hw_hs_node1_weight_hs,
				self.hw_ws_node2_weight_hw,
				self.hw_ws_node2_weight_ws,
				self.ws_hs_node3_weight_ws,
				self.ws_hs_node3_weight_hs,
				self.hw_ds_node1_weight_hw,
				self.hw_ds_node1_weight_ds,
				self.ds_hs_node1_weight_ds,
				self.ds_hs_node1_weight_hs,
				self.ds_ws_node1_weight_ds,
				self.ds_ws_node1_weight_ws,
				self.hd_ds_node1_weight_hd,
				self.hd_ds_node1_weight_ds,
				self.dw_ds_node1_weight_dw,
				self.dw_ds_node1_weight_ds,
				self.dw_hs_node1_weight_dw,
				self.dw_hs_node1_weight_hs,
				self.dw_ws_node1_weight_dw,
				self.dw_ws_node1_weight_ws, self.dw_hw_node1_weight_dw,
				self.dw_hw_node1_weight_hw,
				self.dw_hd_node1_weight_dw,
				self.dw_hd_node1_weight_hd,
				self.hd_ws_node1_weight_hd,
				self.hd_ws_node1_weight_ws,
				self.hd_hs_node1_weight_hd,
				self.hd_hs_node1_weight_hs,
				self.hd_hw_node1_weight_hd,
				self.hd_hw_node1_weight_hw,

				self.output_node1_weight,
				self.output_node2_weight,
				self.output_node3_weight,
				self.output_node4_weight,
				self.output_node5_weight,
				self.output_node6_weight,
				self.output_node7_weight,
				self.output_node8_weight,
				self.output_node9_weight,
				self.output_node10_weight,
				self.output_node11_weight,
				self.output_node12_weight,
				self.output_node13_weight,
				self.output_node14_weight
			]
		'''
'''
		self.biases = [
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
				0
			]
		self.bias = [
				#self.object_height_speed_bias_h,
				#self.object_height_speed_bias_s,
				self.object_height_dist_bias_h,
				self.object_height_dist_bias_d,
				self.object_height_dist2_bias_h,
				self.object_height_dist2_bias_d,
				#self.object_dist_speed_bias_d,
				#self.object_dist_speed_bias_s,
				#self.ds_hs_node1_bias_ds,
				#self.ds_hs_node1_bias_hs,
				#self.hd_ds_node1_bias_hd,
				#self.hd_ds_node1_bias_ds,
				#self.hd_hs_node1_bias_hd,
				#self.hd_hs_node1_bias_hs,
				#self.output_node5_bias,
				#self.output_node7_bias,
				#self.output_node13_bias,
				self.hd_hd2_node1_bias_hd,
				self.hd_hd2_node1_bias_hd2,
				self.hd_hd2_node2_bias_hd,
				self.hd_hd2_node2_bias_hd2,
				self.hidden1_hidden2_bias_h1,
				self.hidden1_hidden2_bias_h2,
				self.hidden1_2_hidden2_bias_h1,
				self.hidden1_2_hidden2_bias_h2,
				self.output_no_land_bias,
				self.output_node1_bias,
				self.output_node2_bias
			]
		'''
'''
		self.bias = [
				self.object_height_width_bias_h, 
				self.object_height_width_bias_w,
				self.object_height_speed_bias_h,
				self.object_height_speed_bias_s,
				self.object_width_speed_bias_w,
				self.object_width_speed_bias_s,
				self.object_dist_speed_bias_d,
				self.object_dist_speed_bias_s,
				self.object_dist_width_bias_d,
				self.object_dist_width_bias_w,
				self.object_height_dist_bias_h,
				self.object_height_dist_bias_d,
				self.hw_hs_node1_bias_hw,
				self.hw_hs_node1_bias_hs,
				self.hw_ws_node2_bias_hw,
				self.hw_ws_node2_bias_ws,
				self.ws_hs_node3_bias_ws,
				self.ws_hs_node3_bias_hs,
				self.hw_ds_node1_bias_hw,
				self.hw_ds_node1_bias_ds,
				self.ds_hs_node1_bias_ds,
				self.ds_hs_node1_bias_hs,
				self.ds_ws_node1_bias_ds,
				self.ds_ws_node1_bias_ws,
				self.hd_ds_node1_bias_hd,
				self.hd_ds_node1_bias_ds,
				self.dw_ds_node1_bias_dw,
				self.dw_ds_node1_bias_ds,
				self.dw_hs_node1_bias_dw,
				self.dw_hs_node1_bias_hs,
				self.dw_ws_node1_bias_dw,
				self.dw_ws_node1_bias_ws,
				self.dw_hw_node1_bias_dw,
				self.dw_hw_node1_bias_hw,
				self.dw_hd_node1_bias_dw,
				self.dw_hd_node1_bias_hd,
				self.hd_ws_node1_bias_hd,
				self.hd_ws_node1_bias_ws,
				self.hd_hs_node1_bias_hd,
				self.hd_hs_node1_bias_hs,
				self.hd_hw_node1_bias_hd,
				self.hd_hw_node1_bias_hw,

				self.output_node1_bias,
				self.output_node2_bias,
				self.output_node3_bias,
				self.output_node4_bias,
				self.output_node5_bias,
				self.output_node6_bias,
				self.output_node7_bias,
				self.output_node8_bias,
				self.output_node9_bias,
				self.output_node10_bias,
				self.output_node11_bias,
				self.output_node12_bias,
				self.output_node13_bias,
				self.output_node14_bias
			]
		'''
'''
		weights = tf.stack([
				object_height_width_weight_h, 
				object_height_width_weight_w,
				object_height_speed_weight_h,
				object_height_speed_weight_s,
				object_width_speed_weight_w,
				object_width_speed_weight_s,
				hw_hs_node1_weight_hw,
				hw_hs_node1_weight_hs,
				hw_ws_node2_weight_hw,
				hw_ws_node2_weight_ws,
				ws_hs_node3_weight_ws,
				ws_hs_node3_weight_hs,
				output_node1_weight,
				output_node2_weight,
				output_node3_weight

			])
		bias = tf.stack([
				object_height_width_bias_h, 
				object_height_width_bias_w,
				object_height_speed_bias_h,
				object_height_speed_bias_s,
				object_width_speed_bias_w,
				object_width_speed_bias_s,
				hw_hs_node1_bias_hw,
				hw_hs_node1_bias_hs,
				hw_ws_node2_bias_hw,
				hw_ws_node2_bias_ws,
				ws_hs_node3_bias_ws,
				ws_hs_node3_bias_hs,
				output_node1_bias,
				output_node2_bias,
				output_node3_bias
			])
		'''
'''
	def randomize_weights(self):
		#This is UNSAFE. Only do this if bias and weight are same size!
		'''
'''
		weights_unpack = tf.unstack(weights)
		bias_unpack = tf.unstack(bias)
		assert(weights_unpack.length == bias_unpack.length)
		for i in range(0, weights.length):
			weights[i].assign(tf.random_uniform())
			bias[i].assign(tf.random_uniform())
		weights = tf.stack(weights_unpack)
		bias = tf.stack(bias_unpack)
		'''
'''
	#	weight_randomized_ops = []
		for i in range(0, len(self.weight)):
			self.weight[i] = np.random.rand()
			#weight_randomized_ops.append(self.weights[i].assign(tf.random_uniform([1])))
		#return weight_randomized_ops
	def randomize_bias(self):
#		bias_randomized_ops = []
		for i in range(0, len(self.biases)):
			self.biases[i] = np.random.rand()
			#bias_randomized_ops.append(self.bias[i].assign(tf.random_uniform([1])))
		#return bias_randomized_ops
	def initialize(self, sess):
		for i in range(0, len(self.weights)):
			sess.run(self.weights[i].initializer)
			sess.run(self.bias[i].initializer)
	def output_val(self, distance, height, noLandObs):
		return sess.run(self.output, {self.object_height: height, self.object_dist:distance, self.no_land_obs:noLandObs, self.object_height_dist_weight_h:self.weight[0], self.object_height_dist_weight_d: self.weight[1], self.object_height_dist2_weight_h: self.weight[2], self.object_height_dist2_weight_d: self.weight[3], self.hd_hd2_node1_weight_hd:self.weight[4], self.hd_hd2_node1_weight_hd2:self.weight[5], self.hd_hd2_node2_weight_hd: self.weight[6], self.hd_hd2_node2_weight_hd2: self.weight[7], self.hidden1_hidden2_weight_h1: self.weight[8], self.hidden1_hidden2_weight_h2:self.weight[9], self.hidden1_2_hidden2_weight_h1: self.weight[10], self.hidden1_2_hidden2_weight_h2: self.weight[11], self.output_no_land_weight: self.weight[12], self.output_node1_weight: self.weight[13], self.output_node2_weight:self.weight[14], self.object_height_dist_bias_h:self.biases[0], self.object_height_dist_bias_d: self.biases[1], self.object_height_dist2_bias_h: self.biases[2], self.object_height_dist2_bias_d: self.biases[3], self.hd_hd2_node1_bias_hd:self.biases[4], self.hd_hd2_node1_bias_hd2:self.biases[5], self.hd_hd2_node2_bias_hd: self.biases[6], self.hd_hd2_node2_bias_hd2: self.biases[7], self.hidden1_hidden2_bias_h1: self.biases[8], self.hidden1_hidden2_bias_h2:self.biases[9], self.hidden1_2_hidden2_bias_h1: self.biases[10], self.hidden1_2_hidden2_bias_h2: self.biases[11], self.output_no_land_bias: self.biases[12], self.output_node1_bias: self.biases[13], self.output_node2_bias:self.biases[14]})
'''
'''
	def mutate(learner1, learner2, sess):
		assert(learner1.weights.length == learner2.weights.length) #This serves two purposes - making sure they are Learners and that they are the same size (not perfect by far though)
		#Pick random variables and "swap" them
		for i in range(0, len(learner1.weights.length)):
			print(np.random.rand())
			if(np.random.rand() > 0.75):
				tmp = learner1.weights[i].eval()
				sess.run(learner1.weights[i].assign(learner2.weights[i].eval()))
				sess.run(learner2.weights[i].assign(tmp))
'''
'''
def cross_over(learner1, learner2, sess):
	assert(len(learner1.weights) == len(learner2.weights)) #This serves two purposes - making sure they are Learners and that they are the same size (not perfect by far though)
	#Pick random variables and "swap" them
	for i in range(0, len(learner1.weight)):
		if(np.random.rand() > 0.6):
			tmp = copy.copy(learner1.weight[i])
			learner1.weight[i] = copy.copy(learner2.weight[i])
			learner2.weight[i] = tmp
			tmp = learner1.weights[i].eval()
			sess.run(learner1.weights[i].assign(learner2.weights[i].eval()))
			sess.run(learner2.weights[i].assign(tmp))
	for i in range(0, len(learner1.bias)):
		if(np.random.rand() > 0.6):
			tmp = copy.copy(learner1.biases[i])
			learner1.biases[i] = copy.copy(learner2.biases[i])
			learner2.biases[i] = tmp
			tmp = learner1.bias[i].eval()
			sess.run(learner1.bias[i].assign(learner2.bias[i].eval()))
			sess.run(learner2.bias[i].assign(tmp))
def mutate(learner1, learner2, sess):
	for i in range(0, len(learner1.weights)):
		if(np.random.rand() > 0.60):
			learner2.weight[i] = learner1.weight[i] * np.random.rand()
			#sess.run(learner2.weights[i].assign(tf.multiply(learner1.weights[i].eval(), tf.random_uniform([1]))))
		#elif(np.random.rand() < 0.20):
			#sess.run(learner2.weights[i].assign(tf.multiply(learner1.weights[i].eval(), tf.random_uniform([1]))))
	for i in range(0, len(learner1.bias)):
		if(np.random.rand() > 0.60):
			learner2.biases[i] = learner1.biases[i] * np.random.rand()
			#sess.run(learner2.bias[i].assign(tf.multiply(learner1.bias[i].eval(), tf.random_uniform([1]))))
		#elif(np.random.rand() < 0.20):
			#sess.run(learner2.bias[i].assign(tf.multiply(learner1.bias[i].eval(), tf.random_uniform([1]))))
'''
'''
def clone(learner1, sess):
	tmp = Learner()
	tmp.initialize(sess)
	for i in range(0, len(tmp.weight)):
		tmp.weight[i] = copy.copy(learner1.weight[i])
		#sess.run(tmp.weights[i].assign(learner1.weights[i]))
		#sess.run(tmp.weights[i].assign(learner1.weights[i].read_value()))
	for i in range(0, len(tmp.biases)):
		tmp.biases[i] = copy.copy(learner1.biases[i])
		#sess.run(tmp.bias[i].assign(learner1.bias[i]))
		#sess.run(tmp.bias[i].assign(learner1.bias[i].read_value()))
	return tmp
'''	
print("Remember to do modprobe uinput")
with uinput.Device([uinput.KEY_SPACE, uinput.KEY_UP, uinput.KEY_DOWN]) as device, tf.Session() as sess:
	writer = tf.summary.FileWriter('/home/usaid/Documents/Python/TensorFlow/TensorboardLogs/dinosaurLogs', sess.graph)
	print("Starting ML program")
	prevGameOver = False
	printed = False	
	browser.find_element_by_tag_name("body").send_keys(Keys.UP)
	while not prevGameOver:
		frame = np.array(sct.grab(mon))
		#cv2.imshow("frame", frame)
		#cv2.waitKey()
		frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		thresh = np.full_like(frameGrey, 0)
		cv2.inRange(frameGrey, 71, 98, thresh)
		img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(thresh, contours, -1, (127,255,0), 3)
		#cv2.imshow("thresh", thresh)
		#cv2.waitKey()
		x, y, w, h = 0, 0, 0, 0
		for i in contours:
			area = cv2.contourArea(i)
			'''Dino
			if(area > 2105):
				x, y, w, h = cv2.boundingRect(i)
				cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
				print(x, w)
				printed = True
			'''
			if(area > 1000 and area < 2000):
				x, y, w, h = cv2.boundingRect(i)
				print(w, h)
				cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
				#printed = True
			elif(area > 500 and area < 1000):
				x, y, w, h = cv2.boundingRect(i)
				print(w, h)
				cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
				#printed = True
			#if(area > 20000): #Game Over!
			if(area > 3000 and area < 4105):
				x, y, w, h = cv2.boundingRect(i)
				print(h, w)
				#if(h > 113 and h < 145 and w > 150 and w < 160):
				if(h > 60 and h < 65 and w > 65 and w < 75):
					print("Game Over verified - Starting Game")
					time.sleep(1.0)
					prevGameOver = True
					neural1 = ImprovedDinoLearner.ImprovedLearner()
					neural2 = ImprovedDinoLearner.ImprovedLearner()
					neural3 = ImprovedDinoLearner.ImprovedLearner()
					neural4 = ImprovedDinoLearner.ImprovedLearner()
					neural5 = ImprovedDinoLearner.ImprovedLearner()
					neural6 = ImprovedDinoLearner.ImprovedLearner()
					neural7 = ImprovedDinoLearner.ImprovedLearner()
					neural8 = ImprovedDinoLearner.ImprovedLearner()
					neural9 = ImprovedDinoLearner.ImprovedLearner()
					neural10 = ImprovedDinoLearner.ImprovedLearner()
					neural11 = ImprovedDinoLearner.ImprovedLearner()
					neural12 = ImprovedDinoLearner.ImprovedLearner()
					genome.extend((neural1, neural2, neural3, neural4, neural5, neural6, neural7, neural8, neural9, neural10, neural11, neural12))
					for i in range(0, len(genome)):
						genome[i].randomize()
						#genome[i].initialize(sess)
						#genome[i].randomize_weights()
						#genome[i].randomize_bias()
						#sess.run(genome[i].randomize_weights())
						#sess.run(genome[i].randomize_bias())
						fitness.append(0)#make sure genome and fitness are same length in case I want to increase NN in genome
					learning = True
					#device.emit_click(uinput.KEY_UP)
		#print("----")
		if printed:
			cv2.imshow("cont", thresh)
			cv2.waitKey()

	'''
	dinoSeen = False
	while not dinoSeen:
		frame = np.array(sct.grab(mon))
		frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		thresh = np.full_like(frameGrey, 0)
		cv2.inRange(frameGrey, 71, 98, thresh)
		img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		x, y, w, h = 0, 0, 0, 0
		for i in contours:
			area = cv2.contourArea(i)
			if(area > 10100 and area < 15600): #Dino!
				if not dinoSeen:
					print("I see you, dino!")
					dinoSeen = True
					cv2.imshow("dinoStart", thresh)
					cv2.waitKey(30)
					break
	'''
	print("Game has started. Survive")
	generation = 0
	while learning:
		print("Testing Generation: " + str(generation))
		generationTime = time.time()
		for neural in range(0, len(genome)):
			print("Testing NN #:" + str(neural))
			startTime = 0
			firstObjectPos = 2700
			lastFirstObjectPos = 2700
			firstObjectHeight = 0
			firstObjectWidth = 0
			firstObjectSpeed = 0
			loopStart = 0
			loopEnd = 0
			floorHeight = 0 #TODO - Add floor height for birds!
			gameOver = True
			jumpedOverCactus = False
			while gameOver:
				frame = np.array(sct.grab(mon))
				frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				thresh = np.full_like(frameGrey, 0)
				cv2.inRange(frameGrey, 71, 98, thresh)
				img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				x, y, w, h = 0, 0, 0, 0
				for i in contours:
					area = cv2.contourArea(i)
					#if(area > 20000): #Game Over!
					if(area > 3000 and area < 4105):
						x, y, w, h = cv2.boundingRect(i)
						#if(h > 113 and h < 145 and w > 150 and w < 160):
						if(h > 60 and h < 65 and w > 65 and w < 75):
							#Record initial time and start game
							browser.find_element_by_tag_name("body").send_keys(Keys.UP)
							#device.emit_click(uinput.KEY_UP)
							time.sleep(2.0)
							startTime = time.time()
							gameOver = False
							break
			
			cactiJumped = 0
			jumpTime = 0
			while not gameOver:
				lastObjectPos = firstObjectPos
				noLandObs = 1
				loopStart = time.time()
				frame = np.array(sct.grab(mon))

				frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#height, width = frameGrey.shape
				#frameHalf = frameGrey[(height / 2):(height-(height/8)), (width/6):(width-(width/6))]
				thresh = np.full_like(frameGrey, 0)
				cv2.inRange(frameGrey, 71, 98, thresh)
				img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				#cv2.drawContours(thresh, contours, -1, (127,255,0), 3)
				x, y, w, h = 0, 0, 0, 0
				for i in contours:
					area = cv2.contourArea(i)
					'''
					if(area > 100.0):
						print(area)
					'''
					#if(area > 1200 and area < 4750): #Tiny Cactus
					if(area > 500 and area < 1000):
						x, y, w, h = cv2.boundingRect(i)
						#if(h > 80):
						if(h > 65 and h < 70):
							if(firstObjectPos > (x + (w / 2)) and (x + (w/2) > 840)):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
								noLandObs = 0
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
					#elif(area > 5150 and area < 10100): #Bird or Big Cactus
					elif(area > 1000 and area < 2000):
						x, y, w, h, = cv2.boundingRect(i)
						'''Bird
						if(h > 200 and h < 217):
							if(firstObjectPos > (x + (w / 2)) and (x + (w/2) > 600) and h < 600):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (155, 0, 0), 5)
						'''
						#if(h > 113 and h < 145):
						if(h > 93 and h < 98):
							if(firstObjectPos > (x + (w / 2)) and (x + (w/2) > 840)):
								firstObjectPos = (x + (w / 2))
								firstObjectHeight = h
								noLandObs = 0
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (55, 0, 0), 5)
						#else:
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 10)
					#elif(area > 10100 and area < 15600): #Dino!
						#x, y, w, h = cv2.boundingRect(i)
						#cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 0, 0), 2)
					#elif(area > 20000): #Game Over!
					elif(area > 3000 and area < 4105):
						x, y, w, h = cv2.boundingRect(i)
						print("Found possible game over")
						print(w, h)
						#if(h > 113 and h < 145 and w > 150 and w < 160):
						if(h > 60 and h < 65 and w > 65 and w < 75):
							print("Game Over verified")
							time.sleep(1.0)
							#cv2.rectangle(thresh, (x, y), (x+w, y+h), (100, 0, 0), 10)
							gameOver = True
							lastFirstObjectPos = 1500
							firstObjectPos = 1500
							timeRun = time.time() - startTime
							fitness[neural] = timeRun
							print("Time lived: " + str(timeRun))
							if(timeRun  > 300):
								fitness[neural] = timeRun
								learning = False
								break
							'''
							cv2.imshow("gameOver", thresh)
							while(cv2.waitKey(0) != 27):
								print("Waiting for real action")	
							'''
						#	break
				#distanceTraveled = lastFirstObjectPos - firstObjectPos
				#print(distanceTraveled)
				#timeSpent = time.time() - loopStart 
				#print("Time spent:" + str(timeSpent))
				#firstObjectSpeed = (distanceTraveled / timeSpent) / 100
				print("Obj Pos:" + str(firstObjectPos)) 
				#print("Speed:" + str(firstObjectSpeed))
				#print("Width:" + str(firstObjectWidth))
				#print("Height:" + str(firstObjectHeight))
				#print("NoLandObs:" + str(noLandObs))
				#print(gameOver)
				if not gameOver:
					#outputValue = genome[neural].output_val(firstObjectPos, firstObjectHeight, noLandObs)#sess.run(genome[neural].output, {genome[neural].object_height: firstObjectHeight, genome[neural].object_dist:firstObjectPos, genome[neural].no_land_obs:noLandObs})
					outputValue = genome[neural].output(firstObjectPos, firstObjectHeight, noLandObs)
					#if(outputValue < 3000):
					#	device.emit_click(uinput.KEY_DOWN)
					#else:
					if(outputValue > 2000):
						#device.emit_click(uinput.KEY_UP)	
						browser.find_element_by_tag_name("body").send_keys(Keys.UP)
					elif(outputValue < 1000):
						browser.find_element_by_tag_name("body").send_keys(Keys.DOWN)
					print(outputValue)
					#print("CactiJumped:" + str(cactiJumped))
					print("NN:" + str(neural) + "," + str(generation))
					print("---------")
					'''
				#if(firstObjectPos < 600):
					if(firstObjectPos < 880 and firstObjectPos > 860):
						if not jumpedOverCactus:
							cactiJumped = cactiJumped + 1
						firstObjectPos = 2700
						jumpedOverCactus = True
					elif(firstObjectPos < 860):
						jumpedOverCactus = False
						firstObjectPos = 2700
					'''
				firstObjectPos = 2700
		if(neural == (len(genome) - 1)):
			print("Sacrificing dinos")
			#Find the fittest
			i = 0
			while(len(fitness) > 4):
				print("Genome length:" + str(len(genome)))
				print("Fitness length:" + str(len(fitness)))
				mostFit = heapq.nlargest(4, fitness)
				mostFitIndex1 = fitness.index(mostFit[0])
				mostFitIndex2 = fitness.index(mostFit[1])
				mostFitIndex3 = fitness.index(mostFit[2])
				mostFitIndex4 = fitness.index(mostFit[3])
				if(i is not mostFitIndex1 and i is not mostFitIndex2 and i is not mostFitIndex3 and i is not mostFitIndex4):
					print("Culling inferior dino #" + str(i))
					del genome[i]
					del fitness[i]
					i = 0
				else:
					if(i == len(genome) - 1):
						i = 0
					else:
						i = i + 1
			print("Generating new genome")
			newGenome = []
			opStart = time.time()
			for i in range(0, len(genome)):
				newGenome.append(ImprovedDinoLearner.clone(genome[i]))
				print("Generated new genome: " + str(i))
			print("Crossing and mutating")
			print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < 8):
				genA = random.choice(newGenome)
				genB = random.choice(newGenome)
				crossed = ImprovedDinoLearner.cross_over(genA, genB)
				mutated = ImprovedDinoLearner.mutate(crossed)
				genome.append(crossed)
				genome.append(mutated)
				fitness.append(0)
				fitness.append(0)
				print("Crossed and mutated")
			print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			print("Regenerating genome")
			newGenome = []
			for i in range(0, len(genome)):
				newGenome.append(ImprovedDinoLearner.clone(genome[i]))
				print("Regenerated new genome: " + str(i))
			print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			print("Just mutating")
			while(len(genome) < 10):
				genA = random.choice(newGenome)
				genB = random.choice(newGenome)
				mutated = ImprovedDinoLearner.mutate(genA)
				mutated1 = ImprovedDinoLearner.mutate(genB)
				genome.append(mutated)
				genome.append(mutated1)
				fitness.append(0)
				fitness.append(0)
				print("Mutated genome")
			print("Time taken: " + str(time.time() - opStart))
			opStart = time.time()
			while(len(genome) < 12):
				newBlood = ImprovedDinoLearner.ImprovedLearner()
				newBlood.randomize()
				genome.append(newBlood)
				fitness.append(0)
			'''
			tmpNN1 = clone(genome[mostFitIndex1], sess)
			tmpNN2 = clone(genome[mostFitIndex2], sess)
			print(tmpNN1.weights)

			#Cross Over fit NNs
			tmpNN1_1 = clone(tmpNN1, sess)
			tmpNN2_1 = clone(tmpNN2, sess)
			cross_over(tmpNN1_1, tmpNN2_1, sess)
			genome = []
			genome.append(tmpNN1)
			genome.append(tmpNN2)
			genome.append(tmpNN1_1)
			genome.append(tmpNN2_1)

			#Mutate fit NNs
			tmpNN1_2 = clone(tmpNN1, sess)
			tmpNN2_2 = clone(tmpNN2, sess)
			mutate(tmpNN1, tmpNN1_2, sess)
			mutate(tmpNN2, tmpNN2_2, sess)
			
			#Cross Over the Mutated NNs
			tmpNN1_3 = Learner()#clone(tmpNN1_2, sess)
			tmpNN2_3 = Learner()#clone(tmpNN2_2, sess)
			#cross_over(tmpNN1_3, tmpNN2_3, sess)
			tmpNN1_3.initialize(sess)
			tmpNN1_3.randomize()
			tmpNN2_3.initialize(sess)
			tmpNN2_3.randomize()
			genome.append(tmpNN1_2)
			genome.append(tmpNN2_2)
			genome.append(tmpNN1_3)
			genome.append(tmpNN2_3)
			'''
		print("Generation Time: " + str(time.time() - generationTime))
		generation = generation + 1
				
					
#cv2.imshow("dinoRun", thresh)
#cv2.waitKey(0)
print("Ended video")
cv2.destroyAllWindows()
browser.quit()
display.stop()
#fvs.stop()
'''
Alright, so what do I need to do?

Import libraries

Create list for genomes. It is going to store the weight and bias of each node.
Create list for "fitness". It is going to keep track of which genomes are the most "fit" to live.


Initialize each node. Inputs to the neural network are:
	-  what is ahead (so we are going to attach a weight based on what is coming up)
		- It's type (SMALL_CACTUS, BIG_CACTUS, etc.)
		- It's height
		- It's width
		- It's speed
		^--- Each of these will require a node. The type will require three nodes, two linked together for small and big cactus
			and one attached to second layer as bird
	 In total, this would create a map like:
		SMALL_CACTUS ---> CACTUS ---> OBSTACLE_TYPE ---> HIDDEN_LAYER --> OUTPUT
		BIG_CACTUS -------^
		BIRD -------------------------^
		OBJECT_HEIGHT -----------------------------------^
		OBJECT_WIDTH  -----------------------------------^
		OBJECT_SPEED  -----------------------------------^
Initialize tf variables

Start tf session
	While game_over (signifying beginning) isn't detected and learning hasn't started:
		Search for game over.
		If found:
			Create 8 NN with random weights and biases and put them in the genome
			Indicate learning has started
	while the neural network doesn't live for 5 minutes (arbitrary number, I'm thinking I need to have something else to figure out which is most fit)
		For each NN:
			Start by detecting game over
			If detected game over:
				Record initial time
				Start game
				while game over isn't detected:
					Detect first obstacle height/width/speed/type
					Send inputs to neural network
						If output is less than 0.45, duck
						Else if output is between 0.45 and 0.55, keep going (do nothing)
						Else jump
			
				After game is over:
					Record time spent (I'm going to be lazy and have "fitness" being how long it lives. THere are limitations with this:
					The cactus doesn't always appear at a certaim time
					Sometimes, you get less cacti
					If this doesn't work, I'm going to track obstacles avoided/detected instead
		Pick the most 2 most fit neural networks.
		Swap one of their input nodes (first layer). This would be six values,	
		"Mutate" another (random) node. Change it's values to a random number.
		This is the second one in the genome list.
		Repeat until 6 "mutated" neurals are created.
		Add the 6 and the initial 2 to the genome (replacing what was in there)
'''
