import numpy as np
cimport numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as anim
from perlin_noise import PerlinNoise
from os import path
from itertools import cycle


cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    bint isnan(double x)
    double M_PI

cdef double pi = M_PI
cdef double radPerTick= 300*pi/(180*1023)

cdef enum ServoTick:
    decrement = -1
    stay = 0
    increment = 1

cdef struct Action:
    ServoTick do1
    ServoTick do2

cdef struct State:
    double x
    double y
    int o1
    int o2

cdef struct Point:
    double x
    double y

cdef struct Params:
    double a1
    double a2
    double a3
    double a4
    double a5

cdef class Dynamics:
    cdef float a1, a2, a3, a4, a5
    cdef Action[9] actions

    cpdef update_params(self, Params p):
        self.a1 = p.a1
        self.a2 = p.a2
        self.a3 = p.a3
        self.a4 = p.a4
        self.a5 = p.a5

    def __init__(self, Params p):
        self.update_params(p)
        self.actions = self.list_actions()

    def get_params(self):
            return self.a1,self.a2,self.a3,self.a4,self.a5


    cpdef kinematics(self, double o1, double o2):
        cdef double x2, y2, x3, y3, x4, y4, p2p4, p2ph, p3ph, xh, yh, a1, a2, a3, a4, a5
        cdef Point p1, p2, p3, p4, p5, ph
        cdef Point points[5]
        o1 = (o1 + pi) % (2 * pi) - pi
        o2 = (o2 + pi) % (2 * pi) - pi
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        x2 = a1*cos(o1)
        y2 = a1*sin(o1)
        x4 = a4*cos(o2)-a5
        y4 = a4*sin(o2)
        p2 = Point(x2,y2)
        p4 = Point(x4,y4)

        # p2p4 = np.linalg.norm(p4-p2)
        p2p4 = sqrt((p4.x-p2.x)**2+(p4.y-p2.y)**2)
        p2ph = (a2**2 - a3**2 + p2p4**2)/(2*p2p4)
        # ph = p2 + (p2ph)/(p2p4)*(p4-p2)
        ph.x = p2.x + (p2ph/p2p4) * (p4.x-p2.x)
        ph.y = p2.y + (p2ph/p2p4) * (p4.y-p2.y)
        p3ph = sqrt(a2**2 - p2ph**2)
        if isnan(p3ph):
            # print("Invalid kinematics")
            # print(p3ph)
            return None

        xh = ph.x
        yh = ph.y
        x3 = xh + (p3ph)/(p2p4)*(y4-y2)
        y3 = yh - (p3ph)/(p2p4)*(x4-x2)
        p3 = Point(x3,y3)
        p1 = Point(0,0)
        p5 = Point(-a5, 0)
        points[:] = [p1,p2,p3,p4,p5]
        return points

    cpdef get_range(self):
        cdef int xlim,ylim
        cdef Point points[5]
        cdef Point p3
        xs = []
        ys = []

        for o1 in np.linspace(0,radPerTick*1023,100):
            for o2 in np.linspace(0,radPerTick*1023,100):
                output = self.kinematics(o1,o2)
                if output is not None:
                    points[:] = output
                    p3 = points[2]
                    xs.append(p3.x)
                    ys.append(p3.y)
        return ( (min(xs), max(xs)), (min(ys), max(ys)) )




    cpdef transition(self, State s, Action a):
        cdef int o1, o2
        cdef Point points [5]
        o1 = s.o1 + a.do1
        o2 = s.o2 + a.do2
        if o1 < 0 or o1 > 1023 or o2 < 0 or o2 > 1023:
            return None, None
        points = self.kinematics(o1*radPerTick, o2*radPerTick)
        if points is not None:
            p3 = points[2]
            next_state = State(p3.x, p3.y, o1, o2)
        else:
            next_state = None
        return next_state, points

    cpdef list_actions(self):
        actions = []
        for a_index in range(3):
            for b_index in range(3):
                do1 = 1*(a_index - 1)
                do2 = 1*(b_index - 1)
                a = Action(do1,do2)
                actions.append(a)
        return actions

class Environment:
    def __init__(self, Dynamics dyn, State s, double sensize, int history_length, double variance=0):
        self.dynamics = dyn
        self.state = s
        self.sensor_size = sensize
        self.history_length = history_length
        self.history = []
        self.variance = variance

    def transition(self, Action a):
        next_state,_ = self.dynamics.transition(self.state, a)
        next_state['x'] + np.random.uniform(-self.variance, self.variance)
        next_state['y'] + np.random.uniform(-self.variance, self.variance)
        return next_state

    def set_state(self, state):
        if len(self.history) == self.history_length:
            self.history.pop(0)
        self.history.append(state)
        self.state = state

    def get_history(self):
        return list(reversed(self.history))

cdef distance(State a, State b):
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))

class Reward:

    def __init__(self, double gamma, limits, array=None):
       self.discount_factor = gamma
       seed = 69
       noise1 = PerlinNoise(octaves=8,seed=seed)
       noise2 = PerlinNoise(octaves=16, seed=seed)
       noise3 = PerlinNoise(octaves=32, seed=seed)
       self.map = lambda x : noise1(x) + 0.5*noise2(x) + 0.25*noise3(x) + 2
       self.x_range = limits[0][1]-limits[0][0]
       self.y_range = limits[1][1]-limits[1][0]
       if array is None:
           self.array = np.array([[self.map([i/self.x_range, j/self.y_range])\
                                   for j in range(int(np.ceil(self.x_range)))]\
                                   for i in range(int(np.ceil(self.y_range)))])

    def __call__(self, State state, history, radius, verbose=0):
        r = self.map([state.x/(self.x_range), state.y/(self.y_range)])
        length = len(history)
        staleness = 0
        max_staleness = (length*(length+1))/2
        for t, other_state in enumerate(history):
            dist = distance(state,other_state)
            # if (verbose>1):
            #     print(f"Dist:{dist}\nState:{state}\nOther:{other_state}")
            # if dist < radius:
            #     ratio = t/length
            #     r = r * ratio
            #     if (verbose>0):
            #         print(f"R: {r}, t/length: {ratio}")
                # return r
            if dist < radius:
               staleness  = staleness + (length-t)
        reward = r*(1-staleness/(max_staleness+1))
        return reward

    def get_gamma(self):
        return self.discount_factor

    def get_array(self):
        return self.array

cpdef expand(State s, Dynamics dynamics):
    successors = []
    succ_points = []
    cdef ServoTick do1,do2
    for a in dynamics.actions:
        s_prime, points = dynamics.transition(s,a)
        successors.append(s_prime)
        succ_points.append(points)
    return successors, succ_points

class Planner:
    def __init__(self, env, Dynamics dyn, reward):
        self.env = env
        self.dynamics = dyn
        self.reward = reward


    def plan(self, state, horizon, history):
        actions = self.dynamics.list_actions()
        reward = self.reward(state, history, radius = 0.2)
        # print(history)

        if horizon > 0:
            successors,_ = expand(state, self.dynamics)
            subplans = []
            values = []
            controls = []
            for s_1 in successors:
                new_hist = history.copy()
                new_hist.insert(0,state)
                subplan, value, control = self.plan(s_1, horizon-1, new_hist)
                subplans.append(subplan)
                values.append(value)
                controls.append(control)

            if horizon == 3:
                # print(subplans)
                # print(values)
                pass



            # print(subplans)
            # print()
            f = lambda i: values[i]
            max_val_idx = max(range(len(values)), key=f)
            control = controls[max_val_idx]
            plan = subplans[max_val_idx]
            value = values[max_val_idx]
            # if (horizon == 3):
            #     print(f"Values:{values}:")
            #     print(f"Chosen: Value {value} at {max_val_idx}")
            action = actions[max_val_idx]
            # next_state = self.dynamics.transition(state, action)
            plan.insert(0, state)
            # value.insert(0, reward)
            value = value*self.reward.get_gamma() + reward
            control.insert(0, action)


            return plan, value,control
        else:
            return [state], reward, [None]

def draw_traj(s_act, s_exp, reward_array):
    fig = plt.figure()
    ax = plt.axes(xlim= (-140,140), ylim=(0,150))
    exp_circ = matplotlib.patches.Circle((-12,0),radius=3, facecolor='r')
    act_circ = matplotlib.patches.Circle((-12,0),radius=2, facecolor='k')
    ax.legend((exp_circ,act_circ),("Expected next state", "Actual next state"))
    ax.add_artist(exp_circ)
    ax.add_artist(act_circ)
    ax.set_aspect('equal')
    x_size,y_size = np.shape(reward_array)
    ax.imshow(reward_array, origin='lower', cmap='viridis', extent=(-x_size//2,x_size//2,0,y_size))
    def init():
        exp_circ.set_center((0,0))
        act_circ.set_center((-12,0))
        return exp_circ, act_circ

    def animate(i):
        exp_circ.set_center((s_exp[i]['x'], s_exp[i]['y']))
        act_circ.set_center((s_act[i]['x'], s_act[i]['y']))
        return exp_circ,act_circ
    animation = anim.FuncAnimation(fig, animate, init_func=init, frames = len(s_exp), interval=40, blit=True)
    animation.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

def drift_generator():
    for k in cycle([1,0.95,0.9,0.85,0.8,0.75,0.7,0.75,0.8,0.85,0.9,0.95]):
        yield k

def main():
    schedule = drift_generator()
    psi = Params(63, 75, 75, 63, 25)
    cdef Dynamics dyn = Dynamics(psi)
    cdef double o1 = pi*1/4
    cdef double o2 = pi*3/4
    cdef Point[5] points0 = dyn.kinematics(o1,o2)
    p3 = points0[2]
    s0 = State(p3.x, p3.y, int(o1/radPerTick), int(o2/radPerTick))

    reward_file = "reward_map.npy"
    if path.exists(reward_file):
        reward_array = np.load(reward_file)
        reward_func = Reward(0.9, limits=dyn.get_range(), array=reward_array)
        print("Reward map loaded from file.")
    else:
        print("Generating reward map.")
        reward_func = Reward(0.9, limits=dyn.get_range())
        reward_array = reward_func.get_array()
        np.save(reward_file, reward_array)


    # Change paramaters below for experimentation
    env = Environment(dyn, s0, 0.2, history_length=10, variance = 1)
    k = 1
    theta = Params(psi['a1']*k, psi['a2']*k, psi['a3']*k, psi['a4']*k, psi['a5']*k)
    planner = Planner(env, Dynamics(theta), reward_func)
    lifetime = 100
    epoch_length = 10
    adaptation_threshold = 10
    planning_horizon = 3
    execution_horizon = 3
    assert(planning_horizon>=execution_horizon)

    states_expected = []
    states_actual = []
    rewards = []
    errors = []
    all_errors = []
    execution_plan = []
    adapting = True
    print(f"""Running with...
          Execution Lifetime: {lifetime} steps
          Drift Epoch: {epoch_length} steps
          Planning horizon: {planning_horizon} steps
          Steps before replanning: {execution_horizon}
          Adaptation status: {adapting}, threshold: {adaptation_threshold}""")

    for i in range(lifetime):
        if i%epoch_length == 0:
            j = next(schedule)
            new_psi = Params(j*psi['a1'], j*psi['a2'], j*psi['a3'], j*psi['a4'], psi['a5'])
            planner.env.dynamics.update_params(new_psi)
            print(f"Environment Drift, psi = {new_psi}")
        plan, reward, actions = planner.plan(planner.env.state, planning_horizon, planner.env.get_history())
        s_pred = plan[1]
        states_expected.append(s_pred)


        if len(execution_plan) <= (planning_horizon - execution_horizon):
            execution_plan = [action for action in actions if action is not None]
            # print(f"Replanning, {execution_plan}")
        a = execution_plan.pop(0)
        # print(a)
        s_act = planner.env.transition(a)

        r = planner.reward(s_act, planner.env.get_history(), radius=0.2, verbose=0)
        rewards.append(r)

        planner.env.set_state(s_act)
        states_actual.append(s_act)
        dist = distance(s_pred,s_act)
        errors.append(dist)
        all_errors.append((dist,s_pred,s_act))
        if (adapting and sum(errors)>adaptation_threshold):
            a1,a2,a3,a4,a5 = planner.dynamics.get_params()
            new_theta = Params(0.5*(new_psi['a1']+a1),
                               0.5*(new_psi['a2']+a2),
                               0.5*(new_psi['a3']+a3),
                               0.5*(new_psi['a4']+a4),
                               0.5*(new_psi['a5']+a5))
            planner.dynamics.update_params(new_theta)
            print(f"Adapting, theta = {new_theta}")
            errors = []
    print(f"Total Reward: {sum(rewards)}") 

    draw_traj(states_actual,states_expected,reward_array)

if __name__ == "__main__":
    main()
