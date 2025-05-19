import numpy as np


from scipy import special
import math

def generate_box_sample(bins, dim):
    step_size = 1/bins
    k = bins+1
    size = np.power(k,dim)
    place = np.zeros((size, dim))
    for h in range(size):
        to_decode = h
        for d in range(dim):
            place[h,d] = (to_decode%k)*step_size
            to_decode = int(to_decode/k)
    return place

def generate_sphere_sample(size, dim):
    place = np.random.normal(size=(size, dim))
    for h in range(size):
        place[h,:] = place[h,:]/np.sqrt(np.sum(np.square(place[h,:])))
    return place

def generate_cantor_sample(length):
    place = np.zeros((np.power(2,length),1))
    for h in range(np.power(2,length)):
        to_decode = h
        for d in range(length):
            bit = to_decode%2
            if bit==1:
                bit=2
            place[h] = place[h]+bit/np.power(3,d+1)
            to_decode = int(to_decode/2)
    euclead_delta = 1/np.power(3,length)
    return place, euclead_delta

mapping = [[0,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1], [2,2]]
def generate_sierpinski_sample(length):
    place = np.zeros((np.power(8,length),2))
    for h in range(np.power(8,length)):
        to_decode = h
        for d in range(length):
            acht = to_decode%8
            bit0 = mapping[acht][0]
            bit1 = mapping[acht][1]
            place[h][0] = place[h][0]+bit0/np.power(3,d+1)
            place[h][1] = place[h][1]+bit1/np.power(3,d+1)
            to_decode = int(to_decode/8)
    euclead_delta_x = 1/np.power(3,length)
    euclead_delta_y = 1/np.power(3,length)
    euclead_delta = np.sqrt(euclead_delta_x**2+euclead_delta_y**2)
    return place, euclead_delta

def generate_weierstrass_sample(length):
    place = np.zeros((np.power(2,length),2))
    for h in range(np.power(2,length)):
        to_decode = h
        for d in range(length):
            bit = to_decode%2
            place[h][0] = place[h][0]+bit/np.power(2,d+1)
            to_decode = int(to_decode/2)
        for k in range(25):
            freq = np.power(2,k)
            place[h][1] = place[h][1] + np.sin(freq*place[h][0])/np.sqrt(freq)
    euclead_delta_x = 1/np.power(2,length)
    #here we use 1/2-Holderness of Weierstrass function
    #the next constant can be found in https://math.stackexchange.com/questions/136445/h%C3%B6lder-continuous-but-not-differentiable-function
    C = (2-1)/((2*np.sqrt(0.5)-1)*(1-np.sqrt(0.5)))
    euclead_delta_y = C*np.power(euclead_delta_x, 0.5)
    euclead_delta = np.sqrt(euclead_delta_x**2+euclead_delta_y**2)
    return place, euclead_delta

def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def generate_lorenz_sample(n_iterations):
    dt = 0.001
    num_steps = n_iterations

    place = np.empty((n_iterations + 1, 3))  # Need one more for the initial values
    place[0] = (0., 1., 1.05)  # Set initial values

    for i in range(num_steps):
        place[i + 1] = place[i] + lorenz(place[i]) * dt
    return place

mapping2 = [[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,2],[0,2,0],[0,2,1],[0,2,2],            [1,0,0],[1,0,2],[1,2,0],[1,2,2],[2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,2],[2,2,0],[2,2,1],[2,2,2]]
def generate_menger_sample(length):
    place = np.zeros((np.power(20,length),3))
    for h in range(np.power(20,length)):
        to_decode = h
        for d in range(length):
            twenty = to_decode%20
            bit0 = mapping2[twenty][0]
            bit1 = mapping2[twenty][1]
            bit2 = mapping2[twenty][2]
            place[h][0] = place[h][0]+bit0/np.power(3,d+1)
            place[h][1] = place[h][1]+bit1/np.power(3,d+1)
            place[h][2] = place[h][2]+bit2/np.power(3,d+1)
            to_decode = int(to_decode/20)
    euclead_delta_x = 1/np.power(3,length)
    euclead_delta = euclead_delta_x*np.sqrt(3)
    return place, euclead_delta

