import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perlin_noise import PerlinNoise
from enum import IntEnum
from dataclasses import dataclass
from george import kernels
import george
import seaborn as sns



XType = [('o1','i4'), ('o2','i4'), ('x','f4'), ('y','f4'), ('do1','i4'), ('do2','i4')]
YType = [('o1','i4'), ('o2','i4'), ('x','f4'), ('y','f4')]
SType = [('o1','i4'), ('o2','i4'), ('x','f4'), ('y','f4')]
ActionType = [('do1','i4'),('do2','i4')]

radPerTick = np.pi*(300/180)*(1/1023)


class ServoTick(IntEnum):
    decrement = -1
    stay = 0.0
    increment = 1


@dataclass
class Action:
    do1: ServoTick
    do2: ServoTick

    # def arr():
    #     return np.array((do1,do2), dtype = ActionType)


class Dynamics:
    def __init__(self, a1, a2, a3, a4, a5):
        self.params = (a1,a2,a3,a4,a5)

    def kinematics(self, o1, o2):
        a1,a2,a3,a4,a5 = self.params
        x2,y2 = (a1*np.cos(o1), a1*np.sin(o1))
        x4,y4 = (a4*np.cos(o2)-a5, a4*np.sin(o2))
        p2 = np.asarray((x2,y2))
        p4 = np.asarray((x4,y4))

        p4p2 = np.linalg.norm(p4-p2)
        p2p4 = p4p2
        p2ph = (a2**2 - a3**2 + p4p2**2)/(2*p4p2)
        ph = p2 + (p2ph)/(p4p2)*(p4-p2)
        p3ph = np.sqrt(a2**2 - p2ph**2)

        xh,yh = (ph[0],ph[1])
        x3 = xh + (p3ph)/(p2p4)*(y4-y2)
        y3 = yh - (p3ph)/(p2p4)*(x4-x2)
        p3 = np.asarray((x3,y3))

        p1 = np.asarray([0,0])
        p5 = np.asarray([-a5, 0])
        return p1,p2,p3,p4,p5

    def transition(self, state, action:Action):
        o1 = state['o1']+action.do1
        o2 = state['o2']+action.do2
        points = self.kinematics(o1*radPerTick, o2*radPerTick)
        p3 = points[2]
        next_state = np.array((o1, o2, p3[0], p3[1]), dtype = SType)
        return next_state, points


def expand(state, actions, analyticModel):
    successors = []
    for action in actions:
        new_state = analyticModel(state, action)
        successors.append(new_state)
    return successors


def constAction(action:Action):
    while True:
        yield(action)

def randomAction(action):
    np.random.seed(420)
    while True:
        o1 = np.random.randint(-1,2)
        o2 = np.random.randint(-1,2)
        yield Action(ServoTick(o1), ServoTick(o2))



def calcTrajectory(policy, dynamics, x_0, N):
    state = x_0

    X = np.array(np.zeros((N,1)), dtype = XType)
    Y = np.array(np.zeros((N,1)), dtype = YType)

    for i in range(N):
        action = next(policy)
        x = np.array((state['o1'], state['o2'], state['x'], state['y'], action.do1, action.do2), dtype = X.dtype)
        state, points = dynamics.transition(state, action)
        y = np.array((state['o1'], state['o2'], state['x'], state['y']), dtype = Y.dtype)
        X[i] = x
        Y[i] = y

    assert(len(X) == len(Y))
    return X,Y

def printTrajectory(X,Y):
    for x,y in zip(X,Y):
        print(f"S,A:{x}\nS':{y}\n")

def unwrap(x):
    return np.hstack([x[col[0]] for col in XType])


def trainPredictors(x,y):
    pxKernel = np.var(y['x']) * kernels.ExpSquaredKernel(0.5, ndim=6)
    pyKernel = np.var(y['y']) * kernels.ExpSquaredKernel(0.5, ndim=6)
    trainingSet = unwrap(x)
    xModel = george.GP(pxKernel)
    xModel.compute(trainingSet)
    yModel = george.GP(pyKernel)
    yModel.compute(trainingSet)
    return xModel,yModel

def calcState(dynamics, o1_0, o2_0):
   p1,p2,p3,p4,p5 = dynamics.kinematics(o1_0, o2_0)
   x_0 = np.array((o1_0, o2_0, p3[0], p3[1]), dtype=SType)
   return x_0

def calcData(dynamics, x_0, N):
   x_i, y_i = calcTrajectory(randomAction(Action(ServoTick.decrement,ServoTick.increment)), dynamics, x_0, N)
   return x_i, y_i

def scheduleParams(lifetime, N, method='constant'):
    # paramSchedule = []
    for i in range(lifetime//N):
        k = (1+0.5*i/N)
        psi =np.array([k*63, k*75, k*75, k*63, k*25])
        dynamics = Dynamics(k*63, k*75, k*75, k*63, k*25)
        # paramSchedule.append(dynamics)
        yield dynamics, psi

def noAdaptation():
    pass


def dataGenerator(X,Y): 
    pass


def main():
    #noise = PerlinNoise(octaves=10, seed=1)
    #xpix, ypix = 1000, 1000
    #rewardMap = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    psi0 = np.array([63, 75, 75, 63, 25])
    # print(np.shape(psi0))
    dynamics = Dynamics(63, 75, 75, 63, 25)
    o1 = (np.pi*1/4)/radPerTick
    o2 = (np.pi*3/4)/radPerTick
    x_0 = calcState(dynamics, o1,o2)
    print(x_0)
    lifetime = 100
    N = 10
    XTrain,YTrain = calcData(dynamics, x_0, 500)
    y1Model,y2Model = trainPredictors(XTrain,YTrain)
    paramSchedule = scheduleParams(lifetime, N)


    X = np.empty((0,1), dtype=XType)
    Y = np.empty((0,1), dtype=YType)
    initState = x_0
    psis =[]
    for dynamics, psi in paramSchedule:
        x_i, y_i = calcTrajectory(randomAction(Action(ServoTick.decrement,ServoTick.increment)), dynamics, initState, N)
        initState = x_i[len(x_i)-1]
        X = np.append(X, x_i, axis=0)
        Y = np.append(Y, y_i, axis=0)
        psis.append(psi)

    ## No adaptation, nonparametric
    pred1, pred_var1 = y1Model.predict(np.squeeze(YTrain['x']), unwrap(X), return_var=True)
    pred2, pred_var2 = y2Model.predict(np.squeeze(YTrain['y']), unwrap(X), return_var=True)


    ###
    experience = [np.squeeze(YTrain['x']),np.squeeze(YTrain['y'])]
    accumulation = [[],[]]
    for x,y in zip(X,Y):
        y1_real = y['x']
        y2_real = y['y']

        y1_pred = y1Model.predict(experience[0],unwrap(X))
        y2_pred = y2Model.predict(experience[1],unwrap(X))

        # print(unwrap(x))

    y1_errors, y2_errors = [],[]

    for i in range(lifetime//N):
        start = N*i
        end = N*(i+1)
        for pred,actual in zip(pred1,Y['x'][start:end]):
            y1_errors.append(np.abs((pred-actual)*100/pred))
        for pred,actual in zip(pred2,Y['y'][start:end]):
            y2_errors.append(np.abs((pred-actual)*100/pred))
        # print(np.mean())
        # print(np.mean())

    # sns.set_style("white")
    # sns.set(font_scale=3)
    plt.rcParams.update({'font.size': 32})

    fig,ax1 = plt.subplots()
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Prediction % Error')

    ax1.plot(range(lifetime), y1_errors, linewidth=4)
    ax1.plot(range(lifetime), y2_errors, linewidth=4)
    for i in range(N):
        ax1.axvline(i*N, color='k')

    ax1.legend(["End Effector X", "End Effector Y"])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Parameter % Error')
    psix = np.linspace(0,lifetime-lifetime//N,N)
    # print(psix)
    psiy = np.linalg.norm((np.array(psi0)-np.array(psis))/np.linalg.norm(psi0), axis=1)
    # print(psiy)
    ax2.scatter(psix,psiy, marker = '^', color = 'k', linewidths=10)
    fig.suptitle('Degradation due to Nonstationary Dynamics without Adaptation', fontsize=42)


    plt.show()




if __name__ =="__main__":
    main()
