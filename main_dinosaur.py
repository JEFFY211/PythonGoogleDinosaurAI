from mss import mss
import numpy as np
import tensorflow as tf
import cv2
import time
import heapq
import copy
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from tkinter import *
from deap import algorithms, base, creator, tools
from pybrain.structure import FeedForwardNetwork, LinearLayer, FullConnection, SigmoidLayer, TanhLayer
import pickle

window = Tk()

n = FeedForwardNetwork()
inLayer = LinearLayer(3)
hiddenLayer = TanhLayer(4)
#hiddenLayer2 = TanhLayer(4)
outLayer = TanhLayer(1)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
#n.addModule(hiddenLayer2)
n.addOutputModule(outLayer)

in2hidden = FullConnection(inLayer, outLayer)
#hidden2hidden = FullConnection(hiddenLayer, hiddenLayer2)
hidden2out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in2hidden)
#n.addConnection(hidden2hidden)
n.addConnection(hidden2out)

n.sortModules()

"""
Create Chrome window to screen record
"""
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=520,320") #At 250% zoom
browser = webdriver.Chrome(chrome_options=chrome_options)
browser.set_page_load_timeout(15)
#Available online as https://usaidpro.github.io/dino. Uses modified keybindings to make ducking easier for the AI
browser.get('file:///C:/Users/usaid/Documents/Github/usaidpro.github.io/dino/index.html')
#To keep dinosaur in one spot (there is a bug moving it)
browser.execute_script("setInterval(function (){Runner.instance_.tRex.xPos = 21}, 2000)")

time.sleep(1.5)
print("Sending UP key to start")
browser.find_element_by_tag_name("body").send_keys(Keys.UP)
time.sleep(8.0)

num = 0
generationsRun = 0
def evaluateOne(individual):
    global generationsRun
    global n
    global browser
    ducking = False
    mon = {'top':360, 'left':0, 'width':1275, 'height':360}
    sct = mss()
    startTime = 0
    firstObjectPos = 1300
    firstObjectPosY = 10
    lastFirstObjectPos = 1300
    firstObjectHeight = 0
    firstObjectWidth = 0
    firstObjectSpeed = 55
    lastObjectSpeed = 55
    lastLastSpeed = 55
    loops = 1
    floorHeight = 0
    gameOver = True
    jumpedOverCactus = False
    jumped = False
    ducked = False
    jumps = 0

    #Set the weights of the neural network based on input from DEAP
    n._setParameters(individual)
    floor = 0
    body = browser.find_element_by_tag_name("body")

    """
    Checking for game over from previous game.
    This is just for making sure the game didn't start accidentally due to
    buffered keypress
    """
    while gameOver:
        #Opencv vision processing the screen recorded frames
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
                
                x, y, w, h = cv2.boundingRect(i)

                if(h > 60 and h < 75 and w > 65 and w < 85):

                    #Record initial time and start game
                    body.send_keys(Keys.UP)
                    time.sleep(2.0)
                    startTime = time.time()
                    gameOver = False
                    break

    cactiJumped = 0
    jumpTime = 0
    
    """
    Actual game loop. Keeps checking latest screen frame, processing it, and
    then sending information about first obstacle to neural network
    """
    while not gameOver:
        lastObjectPos = firstObjectPos
        noLandObs = 0
        frame = np.array(sct.grab(mon))

        frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = np.full_like(frameGrey, 0)
        cv2.inRange(frameGrey, 71, 98, thresh)
        img, contours, hireachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for i in contours:
            area = cv2.contourArea(i)
            if(area > 500 and area < 1350):
                x, y, w, h = cv2.boundingRect(i)
                #Short Cactus
                if(h > 75 and h < 90):
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w/2)) > 60):
                        firstObjectPos = x
                        firstObjectPosY = y
                        firstObjectHeight = h
                        firstObjectWidth = w
                        noLandObs = -2
                        
            elif(area > 1800 and area < 2800):
                x, y, w, h, = cv2.boundingRect(i)
                #Bird
                if(h < 90 and h > 60):
                    if(firstObjectPos > (x + (w / 2)) and (x + (w/2)) > 60) and floor - y < 150:
                        firstObjectPos = x
                        firstObjectPosY = y
                        firstObjectHeight = floor - y
                        firstObjectWidth = w
                        noLandObs = 2
                #Tall Cactus
                elif(h > 100 and h < 120):
                    if(abs(firstObjectPos) > (x + (w / 2)) and (x + (w / 2)) > 60):
                        firstObjectPos = x
                        firstObjectHeight = h
                        firstObjectPosY = y
                        firstObjectWidth = w
                        noLandObs = -2
                        floor = y + h

            elif(area > 5500 and area < 6105):
                x, y, w, h = cv2.boundingRect(i)
                #Game Over, recording time (returning time back to DEAP)
                if(h > 60 and h < 75 and w > 65 and w < 85):
                    timeRun = time.time() - startTime
                    print("Loop time: " + str(timeRun/loops))
                    time.sleep(1.0)
                    gameOver = True
                    lastFirstObjectPos = 1300
                    firstObjectPos = 1300
                    generationsRun+=1
                   
                    return (timeRun,)

        #The speed is the average of the previous three speed values (to reduce noise due to processing lag)
        if(firstObjectPos < lastFirstObjectPos):
            firstObjectSpeed = ((lastFirstObjectPos - firstObjectPos) + lastObjectSpeed + lastLastSpeed) / 3.0
            lastFirstObjectPos = firstObjectPos
            lastLastSpeed = lastObjectSpeed
            lastObjectSpeed = firstObjectSpeed
        elif(firstObjectPos > lastFirstObjectPos):
            lastFirstObjectPos = firstObjectPos

        """
        Send values to neural network and perform actions based on its output
        """
        if not gameOver:
            outputValue = n.activate([firstObjectPos / 1300, firstObjectHeight / 200, firstObjectSpeed / 100])
            
            if(outputValue < -0.25 and not ducking):
                body.send_keys(Keys.SPACE)
                ducking = True

            else:
                if(ducking and outputValue > -0.25):
                    body.send_keys(Keys.RETURN)
                    ducking = False
            
            if(outputValue > 0.25):
                body.send_keys(Keys.UP)
                jumped = True

        firstObjectPos = 1300
        loops += 1
        if(time.time() - startTime > 120):
            return (120,) #To speed up learning - it has already figured it out by now, no need to keep it going
        
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

#To make sure neural network weights start the same every time code is initially run
#So I can test different mutations/breeding methods and see which is better for research
np.random.seed(64)

#Initialize DEAP and modify values for genetic algorithms
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(n.params))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluateOne)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=6)

pop = toolbox.population(n=12)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

trainingTime = time.time()
print("Starting training")
if __name__ == "__main__":
    
    #REMEMBER TO CHANGE THE OBJ FILE TO SAVE NEW RUN
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=175, stats=stats, halloffame=hof)
    
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
    ax1.get_shared_y_axes().join(ax1, ax2)
    line2 = ax2.plot(gen, avgs, "r-", label="Average RunTime")
    ax2.set_ylabel("Avg Time", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="bottom right")

    #Saving statistics
    file = open("GaussianTournment1.obj", "wb")

    pickle.dump(logbook, file)

    file.close()

    print("TRAINING TIME: " + str(time.time() - trainingTime))
    plt.show()