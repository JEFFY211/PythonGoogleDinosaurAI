import numpy as np  
import matplotlib.pyplot as plt  
from scipy.special import expit
def graph(x_range):  
    x = np.array(x_range)  
    #y = expit((x*2-1) * 8)#(1 / (1 + np.exp(-(x * 8))))
    y = expit(x* 8)
    plt.plot(x, y)  
    plt.show()
array = np.linspace(-1, 1, 100)
print(array)
print(1 / (1 + np.exp(-0.01010101)))
graph(array)
