import numpy as np
cimport numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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
            print("Invalid kinematics")
            print(p3ph)
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
                do1 = a_index - 1
                do2 = b_index - 1
                a = Action(do1,do2)
                actions.append(a)
        return actions

class Environment:
    # cdef Dynamics model
    # cdef State state
    # cdef double sensor_size
    # cdef int memory_length
    # cdef list(State) history

    def __init__(self, Dynamics dyn, State s, double sensize, int mem):
        self.model = dyn
        self.state = s
        self.sensor_size = sensize
        self.memory_length = mem
        self.memory = []

    def transition(self, Action a):
        next_state,_ = self.model.transition(self.state, a)
        return next_state

    def set_state(self, state):
        if len(self.memory) == self.memory_length:
            self.memory.pop(0)
        self.memory.append(state)
        self.state = state

    def get_memory(self):
        return list(reversed(self.memory))


cdef distance(State a, State b):
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))

cdef class Reward:
    cdef double discount_factor

    def __init__(self, double gamma):
       self.discount_factor = gamma

    def __call__(self, State state, memory, radius):
        r = state.y+np.abs(state.x)
        length = len(memory)
        # print(f"State: {state}")
        # print(f"History: {memory}")
        for t, other_state in enumerate(memory):
            dist = distance(state,other_state)
            # print(f"Dist:{dist}\nState:{state}\nOther:{other_state}")
            if dist < radius:
                ratio = t/length
                # print(f"R: {r}, t/length: {ratio}")
                r = r * ratio
                return r
        return r


cpdef expand(State s, Dynamics model):
    successors = []
    succ_points = []
    cdef ServoTick do1,do2
    for a in model.actions:
        s_prime, points = model.transition(s,a)
        successors.append(s_prime)
        succ_points.append(points)
    return successors, succ_points

class Planner:
    def __init__(self, env, Dynamics dyn):
        self.env = env
        self.dynamics = dyn
        self.reward = Reward(0.9)


    def plan(self, state, horizon, history):
        actions = self.dynamics.list_actions()
        reward = self.reward(state, history, radius = 0.2)
        # print(state)
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
            value = value+reward
            control.insert(0, action)


            return plan, value,control
        else:
            return [state], reward, [None]



def draw_traj(s_exp, s_act):
    fig = plt.figure()
    ax = plt.axes(xlim= (-70,70), ylim=(0,150))
    exp_circ = matplotlib.patches.Circle((-12,0),radius=5, facecolor='b')
    act_circ = matplotlib.patches.Circle((-12,0),radius=5, facecolor='k')
    ax.add_artist(exp_circ)
    ax.add_artist(act_circ)
    ax.set_aspect('equal')
    def init():
        exp_circ.set_center((0,0))
        act_circ.set_center((-12,0))
        return exp_circ, act_circ

    def animate(i):
        exp_circ.set_center((s_exp[i]['x'], s_exp[i]['y']))
        act_circ.set_center((s_exp[i]['x'], s_exp[i]['y']))
        return exp_circ,act_circ
    animation = anim.FuncAnimation(fig, animate, init_func=init, frames = len(s_exp), interval=40, blit=True)
    animation.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def main():
    cdef Dynamics dyn = Dynamics(Params(63, 75, 75, 63, 25))
    cdef double o1 = pi*1/4
    cdef double o2 = pi*3/4
    cdef Point[5] points0 = dyn.kinematics(o1,o2)
    p3 = points0[2]
    s0 = State(p3.x, p3.y, int(o1/radPerTick), int(o2/radPerTick))
    env = Environment(dyn, s0, 0.2, 50)
    planner = Planner(env, dyn)
    lifetime = 20
    states_expected = []
    states_actual = []
    rewards = []
    errors = []

    for i in range(lifetime):
        plan, reward, actions = planner.plan(planner.env.state, 3, env.get_memory())
        s_pred = plan[1]
        states_expected.append(s_pred)

        s_act = planner.env.transition(actions[0])

        r = planner.reward(s_act, planner.env.get_memory(), radius=0.2)
        rewards.append(r)

        planner.env.set_state(s_act)
        states_actual.append(s_act)
        errors.append(distance(s_pred,s_act))
    print(f"Total Reward: {sum(rewards)}")
    draw_traj(states_actual,states_expected)

if __name__ == "__main__":
    main()
