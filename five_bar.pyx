import numpy as np
cimport numpy as np
cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double M_PI

cdef double pi = M_PI
cdef double radPerTick= np.pi*(300/180)*(1/1023)

cdef enum ServoAction:
    decrement = -1
    stay = 0
    increment = 1

cdef struct Action:
    ServoAction do1
    ServoAction do2

cdef struct State:
    double x
    double y
    int o1
    int o2

cdef struct Point:
    double x
    double y

cdef class Dynamics:
    cdef float a1, a2, a3, a4, a5

    cpdef update_params(self, float a1, float a2, float a3, float a4, float a5):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def __init__(self, float a1, float a2, float a3, float a4, float a5):
        self.update_params(a1,a2,a3,a4,a5)


    cpdef kinematics(self, double o1, double o2):
        cdef double x2, y2, x3, y3, x4, y4, p2p4, p2ph, p3ph, xh, yh, a1, a2, a3, a4, a5
        cdef Point p1,p2, p3, p4, p5, ph
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
        p3ph = np.sqrt(a2**2 - p2ph**2)

        xh = ph.x
        yh = ph.y
        x3 = xh + (p3ph)/(p2p4)*(y4-y2)
        y3 = yh - (p3ph)/(p2p4)*(x4-x2)
        p3 = Point(x3,y3)

        p1 = Point(0,0)
        p5 = Point(-a5, 0)
        return p1,p2,p3,p4,p5

    cpdef transition(self, State s, Action a):
        cdef int o1, o2
        cdef Point [:] points
        o1 = s.o1 + a.do1
        o2 = s.o2 + a.do2
        if o1 < 0 or o1 > 1023 or o2 < 0 or o2 > 1023:
            return None, None
        points = self.kinematics(o1*radPerTick, o2*radPerTick)
        p3 = points[2]
        next_state = State(p3[0], p3[1], o1, o2)
        return next_state, points

cdef class Reward:
    cdef double [:,:] reward_map
    cdef double [:,:] freshness

    def __init__(self):
       pass 

cdef class Environment:
    cdef Dynamics model
    cdef Reward reward

    def __init__(self, Dynamics dyn):
        self.model = dyn
        self.reward = Reward()

def main():
    cdef Dynamics five_bar = Dynamics(63, 75, 75, 63, 25)
    env = Environment(five_bar)
    cdef double o1, o2
    o1 = np.pi*1/4
    o2 = np.pi*3/4
    data = five_bar.kinematics(o1,o2)
    print(data)

if __name__ == "__main__":
    main()
