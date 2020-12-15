import numpy as np
cimport numpy as np
cdef extern from "math.h":
    double sin(double x)
    double cos(double x)

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
        cdef np.ndarray p1,p2, p3, p4, p5, ph
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        x2 = a1*cos(o1)
        y2 = a1*sin(o1)
        x4 = a4*cos(o2)-a5
        y4 = a4*sin(o2)
        p2 = np.asarray((x2,y2))
        p4 = np.asarray((x4,y4))

        p2p4 = np.linalg.norm(p4-p2)
        p2ph = (a2**2 - a3**2 + p2p4**2)/(2*p2p4)
        ph = p2 + (p2ph)/(p2p4)*(p4-p2)
        p3ph = np.sqrt(a2**2 - p2ph**2)

        xh = ph[0]
        yh = ph[1]
        x3 = xh + (p3ph)/(p2p4)*(y4-y2)
        y3 = yh - (p3ph)/(p2p4)*(x4-x2)
        p3 = np.asarray((x3,y3))

        p1 = np.array([0,0])
        p5 = np.asarray([-a5, 0])
        return p1,p2,p3,p4,p5


cdef Dynamics five_bar = Dynamics(63, 75, 75, 63, 25)
cdef float radPerTick
radPerTick= np.pi*(300/180)*(1/1023)
o1 = (np.pi*1/4)
o2 = (np.pi*3/4)
data = five_bar.kinematics(o1,o2)
print(data)
