from mss import mss
#from PIL import Image
import numpy as np
#Had to change from import Queue - make sure this is right one!
import queue
#from multiprocessing.queues import Queue 
import tensorflow as tf
import cv2
#from imutils.video import FileVideoStream
#from imutils.video import FPS
import time
import heapq
import copy
import random
#from pyvirtualdisplay import Display
#from pyvirtualdisplay.smartdisplay import SmartDisplay
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import ImprovedDinoLearner
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from tkinter import *
from deap import algorithms, base, creator, tools

window = Tk()

f = open('dinosaurLog.csv', 'wt')
learner1_q = queue.Queue()
learner2_q = queue.Queue()
learner3_q = queue.Queue()
learner4_q = queue.Queue()
tested_q = queue.Queue()
tested_q1 = queue.Queue()
tested_q2 = queue.Queue()
tested_q3 = queue.Queue()
tested_q4 = queue.Queue()
graph_q = multiprocessing.Queue()



#display = Display(visible=1, size=(3840, 2160))
#display.start()

dinosPerGeneration = 64
eliteDinos = 8 
tournaments = 4
truncatedSelection = False 
tournamentSelection = True
def learning_func(position):
    profile = webdriver.FirefoxProfile()
    profile.set_preference('webdriver.load.strategy', 'unstable')
    learning = True
    generationsInWindow = 0
    if(position == 0):
        writer = csv.writer(f)
        #writer.writerow(('Generation', 'Fitness'))
        generation = 0
        #neurals = []
        #neuralTempFitness = []
        for i in range(0, dinosPerGeneration - 1):
            neural = ImprovedDinoLearner.ImprovedLearner()
            neural.randomize()
            learner_q.put(neural)
            #neurals.append(neural)
        while learning:
            print("waiting for dinos")
            genome = []
            fitness = []
            i = 0
            fitnessTotal = 0
            highestFitness = 0
            print("Generation: " + str(generation + 1))
            learner_q.join()
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
            writer.writerow((generation + 1, fitnessTotal, highestFitness))
            graph_q.put((generation + 1, highestFitness, fitnessTotal / dinosPerGeneration))
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
                    if(i is not mostFitIndex1 and i is not mostFitIndex2 ):
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
                print(fitness)
                tournament = np.array_split(np.array(genome), tournaments)
                tournamentFitness = np.array_split(np.array(fitness), tournaments)
                genome = []
                fitness = []
                for j in range(0, tournaments):
                    print("STARTING TOURNAMENT")
                    tempFitness = np.sort(tournamentFitness[j], axis=None)
                    print(tempFitness)
                    tempFitness = tempFitness[-2:]
                    print("Cutted tempFitness")
                    print(tempFitness)
                    index2 = np.where(tournamentFitness[j] == tempFitness[0])
                    print("Created index2")
                    index1 = np.where(tournamentFitness[j] == tempFitness[1])
                    print("Created indexes")
                    genome.append(tournament[j][index1[0][0]])
                    genome.append(tournament[j][index2[0][0]])
                    fitness.append(tempFitness[1])
                    fitness.append(tempFitness[0])
                    print(index1[0][0], index2[0][0], tempFitness[1], tempFitness[0], tournamentFitness[j][index1[0][0]])
            else:
                print("FATAL ERROR: CANNOT TOURNAMENT AND TRUNCATE!")
                break;
            newGenome = []
            opStart = time.time()
            for i in range(0, len(genome)):
                print(genome[i].fitness)
                newGenome.append(ImprovedDinoLearner.clone(genome[i]))
            opStart = time.time()
            while(len(genome) < dinosPerGeneration * 0.2):
                genA = random.choice(newGenome)
                genB = random.choice(newGenome)
                crossed = ImprovedDinoLearner.cross_over(genA, genB)
                genome.append(crossed)
            opStart = time.time()
            while(len(genome) < dinosPerGeneration * 0.6):
                genA = random.choice(newGenome)
                genB = random.choice(newGenome)
                crossed = ImprovedDinoLearner.cross_over(genA, genB)
                mutated = ImprovedDinoLearner.mutate(crossed)
                genome.append(mutated)
            opStart = time.time()
            while(len(genome) < dinosPerGeneration * (5/6)):
                genA = random.choice(newGenome)
                genB = random.choice(newGenome)
                mutated = ImprovedDinoLearner.mutate(genA)
                mutated1 = ImprovedDinoLearner.mutate(genB)
                genome.append(mutated)
                genome.append(mutated1)
            while(len(genome) < dinosPerGeneration):
                newBlood = ImprovedDinoLearner.ImprovedLearner()
                newBlood.randomize()
                genome.append(newBlood)
            random.shuffle(genome)
            for i in genome:
                i.fitness1 = 0
                i.fitness2 = 0
                i.fitness3 = 0
                i.fitness = 0 #Decay to fitness = lower lucky ones' fitness over time
                learner_q.put(i)
            generation = generation + 1
    elif(position == 1):
        browser = webdriver.Firefox(profile)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':50, 'left':0, 'width':1275, 'height':360}#3840
        sct = mss()
    elif(position == 2):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(1300, 0)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':50, 'left':1300, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 3):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(0, 450)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':500, 'left':0, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 4):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(1300, 450)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':500, 'left':1300, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 5):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(0, 950)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':1000, 'left':0, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 6):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(1300, 950)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':1000, 'left':1300, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 7):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(0, 1350)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        
        mon = {'top':1400, 'left':0, 'width':1275, 'height':360}
        sct = mss()
    elif(position == 8):
        browser = webdriver.Firefox(profile)
        browser.set_window_position(1300, 1350)
        browser.set_page_load_timeout(15)
        browser.set_window_size(1300, 400)
        browser.get('http://usaidpro.github.io/dino/')
        browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
        mon = {'top':1400, 'left':1300, 'width':1275, 'height':360}
        sct = mss()    
    print("Starting ML program")
    prevGameOver = False
    printed = False    
    browser.find_element_by_tag_name("body").send_keys(Keys.UP)
    ducking = False
    while not prevGameOver:
        frame = np.array(sct.grab(mon))
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
                if(h > 60 and h < 65 and w > 65 and w < 75):
                    print("Game Over verified - Starting Game")
                    time.sleep(1.0)
                    prevGameOver = True
            if(area > 2000 and position == 3):
                x, y, w, h = cv2.boundingRect(i)
                print(x, w, (x + w/2))
    success = False
    while not success:
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
                if(area > 3000 and area < 4105):
                    x, y, w, h = cv2.boundingRect(i)
                    if(h > 60 and h < 65 and w > 65 and w < 75):
                        dino = learner_q.get()
                        #Record initial time and start game
                        browser.find_element_by_tag_name("body").send_keys(Keys.UP)
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
            thresh = np.full_like(frameGrey, 0)
            cv2.inRange(frameGrey, 71, 98, thresh)
            img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = 0, 0, 0, 0
            for i in contours:
                area = cv2.contourArea(i)
                if(area > 500 and area < 1000):
                    x, y, w, h = cv2.boundingRect(i)
                    if(h > 65 and h < 70):
                        if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w/2)) > 60):
                            firstObjectPos = (x + (w / 2))
                            firstObjectHeight = h
                            noLandObs = 0
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
                        if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w / 2)) > 60):
                            firstObjectPos = (x + (w / 2))
                            firstObjectHeight = h
                            noLandObs = 0
                elif(area > 3000 and area < 4105):
                    x, y, w, h = cv2.boundingRect(i)
                    if(h > 60 and h < 65 and w > 65 and w < 75):
                        time.sleep(1.0)
                        gameOver = True
                        lastFirstObjectPos = 1300
                        firstObjectPos = 1300
                        timeRun = time.time() - startTime
                        if(jumped):
                            jumped = False
                            timeRun = timeRun + 1 - (jumps / 6)
                        if(ducked):
                            ducked = False
                            timeRun = timeRun + 1
                        dino.fitness = timeRun
                        learner_q.task_done()
                        tested_q.put(dino)
                        print("Time lived: " + str(timeRun) + "\tJumps: " + str(jumps))
                        try:
                            if generationsInWindow % 60 == 0:
                                browser.get('http://usaidpro.github.io/dino/');
                                browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
                                time.sleep(2.0)
                                browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                                time.sleep(5.0)
                        except:
                            print("No connection to Internet!")
                        
                        generationsInWindow+=1
                        print(generationsInWindow, position)
                        if(timeRun > 60):
                            print("Successful dino!")
                            success = True    
            if not gameOver:
                outputValue = dino.output(firstObjectPos, firstObjectHeight, noLandObs)
                if(outputValue < 0.35 and not ducking):
                    browser.find_element_by_tag_name("body").send_keys(Keys.SPACE)
                    ducking = True
                    time.sleep(0.05)
                else:
                    if(ducking):
                        browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                        ducking = False
                if(outputValue > 0.65):
                    #device.emit_click(uinput.KEY_UP)    
                    #browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                    browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                    time.sleep(0.6)
                    jumps = jumps + 1
            firstObjectPos = 1300
#profile = webdriver.FirefoxProfile()
#profile.set_preference('webdriver.load.strategy', 'unstable')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=520,320") #At 250% zoom
browser = webdriver.Chrome(chrome_options=chrome_options)
browser.set_page_load_timeout(15)
#browser.set_window_size(1300, 400)
browser.get('http://usaidpro.github.io/dino/')
browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(1.0)'; cont.style.top = '300px'")
'''
browser1 = webdriver.Firefox(profile)
browser1.set_page_load_timeout(15)
browser1.set_window_size(1300, 400)
browser1.get('http://usaidpro.github.io/dino/')
browser1.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
browser2 = webdriver.Firefox(profile)
browser2.set_window_position(1300, 0)
browser2.set_page_load_timeout(15)
browser2.set_window_size(1300, 400)
browser2.get('http://usaidpro.github.io/dino/')
browser2.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
browser3 = webdriver.Firefox(profile)
browser3.set_window_position(0, 450)
browser3.set_page_load_timeout(15)
browser3.set_window_size(1300, 400)
browser3.get('http://usaidpro.github.io/dino/')
browser3.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
browser4 = webdriver.Firefox(profile)
browser4.set_window_position(1300, 450)
browser4.set_page_load_timeout(15)
browser4.set_window_size(1300, 400)
browser4.get('http://usaidpro.github.io/dino/')
browser4.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")

browser1.find_element_by_tag_name("body").send_keys(Keys.UP)
browser2.find_element_by_tag_name("body").send_keys(Keys.UP)
browser3.find_element_by_tag_name("body").send_keys(Keys.UP)
browser4.find_element_by_tag_name("body").send_keys(Keys.UP)
'''
time.sleep(1.0)
print("Sending UP key to start")
browser.find_element_by_tag_name("body").send_keys(Keys.UP)

time.sleep(8.0)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 30
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

num = 0
def dinoRun(number, array):
    ducking = False
    if(number == 0):
        mon = {'top':50, 'left':0, 'width':1275, 'height':360}
        browser = browser1
    elif(number == 1):
        mon = {'top':50, 'left':1300, 'width':1275, 'height':360}
        browser = browser2
    elif(number == 2):
        mon = {'top':500, 'left':0, 'width':1275, 'height':360}
        browser = browser3
    elif(number == 3):
        mon = {'top':500, 'left':1300, 'width':1275, 'height':360}
        browser = browser4
    sct = mss()
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
    jumped = False
    ducked = False
    jumps = 0
    dino = ImprovedDinoLearner.ImprovedLearner()
    dino.values = array
    while gameOver:
        frame = np.array(sct.grab(mon))
        frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = np.full_like(frameGrey, 0)
        cv2.inRange(frameGrey, 71, 98, thresh)
        img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for i in contours:
            area = cv2.contourArea(i)
            if(area > 3000 and area < 4105):
                x, y, w, h = cv2.boundingRect(i)
                if(h > 60 and h < 65 and w > 65 and w < 75):
                    #Record initial time and start game
                    browser.find_element_by_tag_name("body").send_keys(Keys.UP)
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
        thresh = np.full_like(frameGrey, 0)
        cv2.inRange(frameGrey, 71, 98, thresh)
        img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for i in contours:
            area = cv2.contourArea(i)
            if(area > 500 and area < 1000):
                x, y, w, h = cv2.boundingRect(i)
                if(h > 65 and h < 70):
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w/2)) > 60):
                        firstObjectPos = (x + (w / 2))
                        firstObjectHeight = h
                        noLandObs = 0
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
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w / 2)) > 60):
                        firstObjectPos = (x + (w / 2))
                        firstObjectHeight = h
                        noLandObs = 0
            elif(area > 3000 and area < 4105):
                x, y, w, h = cv2.boundingRect(i)
                if(h > 60 and h < 65 and w > 65 and w < 75):
                    time.sleep(1.0)
                    gameOver = True
                    lastFirstObjectPos = 1300
                    firstObjectPos = 1300
                    timeRun = time.time() - startTime
                    if(jumped):
                        jumped = False
                        timeRun = timeRun + 1 - (jumps / 6)
                    if(ducked):
                        ducked = False
                        timeRun = timeRun + 1
                    dino.fitness = timeRun
                    print("Time lived: " + str(timeRun) + "\tJumps: " + str(jumps))
                    #queue.put(timeRum)
                    return timeRun
                    '''
                    try:
                        if generationsInWindow % 60 == 0:
                            browser.get('http://usaidpro.github.io/dino/');
                            browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
                            time.sleep(2.0)
                            browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                            time.sleep(5.0)
                    except:
                        print("No connection to Internet!")
                    generationsInWindow+=1
                    print(generationsInWindow, position)
                    '''
                    if(timeRun > 60):
                        print("Successful dino!")
        if not gameOver:
            outputValue = dino.output(firstObjectPos, firstObjectHeight, noLandObs)
            if(outputValue < 0.35 and not ducking):
                browser.find_element_by_tag_name("body").send_keys(Keys.SPACE)
                ducking = True
                time.sleep(0.05)
            else:
                if(ducking):
                    browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                    ducking = False
            if(outputValue > 0.65):
                #device.emit_click(uinput.KEY_UP)    
                #browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                time.sleep(0.6)
                jumps = jumps + 1
        firstObjectPos = 1300
'''
def evaluate(individual):
    result = 0.0
    p_q = multiprocessing.Queue()
    p = multiprocessing.Process(target=dinoRun, args(num, individual, p_q,))
    num+=1
    if(num > 3):
        num = 0
    p.start()
    result = p_q.get()
    p.join()
    return result
'''
'''
dinoNum = 0
def evalDino(individual):
    global dinoNum
    result = dinoRun(dinoNum, individual)
    if(dinoNum > 3):
        dinoNum = 0
    else:
        dinoNum+=1
    return result
'''
generationsRun = 0
def evaluateOne(individual):
    print("Starting evaluation")
    global generationsRun
    ducking = False
    mon = {'top':410, 'left':0, 'width':1275, 'height':360}
    sct = mss()
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
    jumped = False
    ducked = False
    jumps = 0
    dino = ImprovedDinoLearner.ImprovedLearner2()
    dino.values = individual 
    floor = 0
    while gameOver:
        frame = np.array(sct.grab(mon))
        frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = np.full_like(frameGrey, 0)
        cv2.inRange(frameGrey, 71, 98, thresh)
        img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for i in contours:
            area = cv2.contourArea(i)
            
            x,y,w,h = cv2.boundingRect(i)
            if(area > 5500 and area < 6105):
                print(area)
                cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(i)
                print(x, y, w, h)
                if(h > 60 and h < 75 and w > 65 and w < 85):
                    print("Found gameover")
                    #Record initial time and start game
                    browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                    time.sleep(2.0)
                    startTime = time.time()
                    gameOver = False
                    break
    cactiJumped = 0
    jumpTime = 0
    
    while not gameOver:
        lastObjectPos = firstObjectPos
        noLandObs = 0
        loopStart = time.time()
        frame = np.array(sct.grab(mon))

        frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = np.full_like(frameGrey, 0)
        cv2.inRange(frameGrey, 71, 98, thresh)
        img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for i in contours:
            area = cv2.contourArea(i)
            if(area > 500 and area < 1000):
                x, y, w, h = cv2.boundingRect(i)
                if(h > 65 and h < 70):
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w/2)) > 60):
                        firstObjectPos = x
                        #firstObjectPos = (x + (w / 2))
                        firstObjectHeight = h
                        #firstObjectHeight = y + h
                        noLandObs = -200
            elif(area > 1000 and area < 2000):
                x, y, w, h, = cv2.boundingRect(i)
                #print(h)
                #Bird
                if(h < 90 and h > 60):
                    #255
                    #202
                    #150
                    if(firstObjectPos > (x + (w / 2)) and (x + (w/2)) > 60):
                        firstObjectPos = x
                        #firstObjectPos = (x + (w / 2))
                        firstObjectHeight = floor - y

                        print(x, y, w, h, firstObjectHeight)
                        noLandObs = 200
                        #firstObjectHeight = y + h
                    #cv2.rectangle(thresh, (x, y), (x+w, y+h), (155, 0, 0), 5)
                if(h > 90 and h < 98):#93
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w / 2)) > 60):
                        #firstObjectPos = (x + (w / 2))
                        firstObjectPos = x
                        firstObjectHeight = h
                        #firstObjectHeight = y + h
                        noLandObs = -200
                        floor = y + h
            elif(area > 3000 and area < 4105):
                x, y, w, h = cv2.boundingRect(i)
                if(h > 60 and h < 65 and w > 65 and w < 75):
                    time.sleep(1.0)
                    gameOver = True
                    lastFirstObjectPos = 1300
                    firstObjectPos = 1300
                    timeRun = time.time() - startTime
                    if(jumped):
                        jumped = False
                        timeRun = timeRun + 1 - (jumps / 6)
                    if(ducked):
                        ducked = False
                        timeRun = timeRun + 1
                    dino.fitness = timeRun
                    print("Time lived: " + str(timeRun) + "\tJumps: " + str(jumps))
                    #queue.put(timeRum)
                    try:
                        if generationsRun % 60 == 0:
                            browser.get('http://usaidpro.github.io/dino/');
                            browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)'; cont.style.top = '300px'")
                            time.sleep(2.0)
                            browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                            time.sleep(5.0)
                    except:
                        print("No connection to Internet!")
                    generationsRun+=1
                    print(generationsRun)
                    return (timeRun,)
                    if(timeRun > 60):
                        print("Successful dino!")
        if not gameOver:
            outputValue = dino.output(firstObjectPos, firstObjectHeight, noLandObs)
            if(outputValue > 650 and not ducking):
                browser.find_element_by_tag_name("body").send_keys(Keys.SPACE)
                ducking = True
                #time.sleep(0.05)
            '''
            else:
                if(ducking and outputValue > 650):
                    browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                    ducking = False
            '''
            if(ducking and outputValue < 350):
                    browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                    ducking = False
            if(firstObjectHeight > 100):
                print(outputValue)
            if(outputValue < 350):
                #device.emit_click(uinput.KEY_UP)    
                #browser.find_element_by_tag_name("body").send_keys(Keys.RETURN)
                browser.find_element_by_tag_name("body").send_keys(Keys.UP)
                jumped = True
                #time.sleep(0.1)
                #time.sleep(0.6)
                #jumps = jumps + 1
        firstObjectPos = 1300

toolbox.register("evaluate", evaluateOne)
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=-1.0, up=1.0, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selRoulette)
random.seed(64)
#dinoPool = multiprocessing.Pool(processes=4)
#toolbox.register("map", dinoPool.map)

pop = toolbox.population(n=24)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=60, stats=stats, halloffame=hof)
gen = logbook.select("gen")
mins = logbook.select("min")
maxs = logbook.select("max")
avgs = logbook.select("avg")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, maxs, "b-", label="Maximum RunTime")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Runtime", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, avgs, "r-", label="Average RunTime")
ax2.set_ylabel("Avg Time", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="bottom right")

plt.show()

'''
#pool.close()
def main_func():
    tpool = ThreadPool(5)
    results = tpool.map(learning_func, [0,1,2,3,4,5,6,7,8])
def plot():    #Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global line,ax,canvas
    global y_val
    global x_val
    global aver_val
    y_val = []
    x_val = []
    aver_val = []
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Generations', fontsize=30)
    ax.set_ylabel('Time survived (secs)', fontsize=30)
    ax.set_title('Dinosaur AI Training', fontsize=50)
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.show()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    line, = ax.plot([1,2,3], [1,2,10], '-b', label='Longest Surviving Time')
    line, = ax.plot([0,0,0], [0,2,10], '-g', label='Average Survival Time')
    ax.legend()




def updateplot(q):
    try:       #Try to check if there is data in the queue
        result=graph_q.get_nowait()

        if result !='Q':
                 #here get crazy with the plotting, you have access to all the global variables that you defined in the plot function, and have the data that the simulation sent.
         x_val.append(result[0])
         y_val.append(result[1])
         aver_val.append(result[2])
         line, = ax.plot(x_val, y_val, '-b', label='Longest Surviving Time')
         line1, = ax.plot(x_val, aver_val, '-g', label='Average Survival Time')
             #line.set_ydata([1,result,10])
             ax.draw_artist(line)
         ax.draw_artist(line1)
             canvas.draw()
             window.after(500,updateplot,q)
    else:
        print(result)
    except:
        window.after(500,updateplot,q)

#training = multiprocessing.Process(None, main_func)
#training.start()

plot()
updateplot(graph_q)
window.mainloop()
'''
