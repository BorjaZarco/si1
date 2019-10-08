import matplotlib.pyplot as plt
import numpy as np

def displayPlot(m, n, x, y):
    plt.figure('Adaline Figure [with data]')
    px1,px2,py1,py2 = list(),list(),list(),list()
   
    for coor,out in zip(x,y):
        if(out==-1):
            px1.append(coor[0])
            py1.append(coor[1])
        else:
            px2.append(coor[0])
            py2.append(coor[1])

    plt.plot(px1, py1, 'o', c='y')
    plt.plot(px2, py2, 'o', c='r')
    vecX = np.linspace(-1,8,100)
    vecY = np.multiply(m, vecX) + n
    plt.plot(vecX,vecY)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def displayError(perceptron):
    plt.figure('SSE Progress')
    plt.plot(list(range(0,perceptron.get_epochs())), perceptron.get_errors())
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()


