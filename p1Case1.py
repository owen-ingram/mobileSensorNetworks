from ctypes import c_uint32 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
num_node = 100 # Number of sensor pos dimensions = 2 # Space dimensions
dimensions = 2
d = 15 # distance btw pos
k = 1.4 # scaling factor
r = k*d # interaction range e 0.1
c1 = 60
c2 = 2* np.sqrt(c1)
a = 5
b = 5
c = np.abs(a-b)/np.sqrt(4*a*b)
EPSILON = 0.1
# time steps
delta_t = 0.008
iter = np.arange(0, 7, delta_t)
iteration = iter.shape[0]
# keep track of all position for traj plot
x_arr = np.zeros([num_node, iteration])
y_arr = np.zeros([num_node, iteration])
# current position
pos = np.random.uniform (size=(num_node, dimensions), high=150)
pos_old = pos
# current velocity
velocities = np.zeros([num_node, dimensions])
# all velocities for graph
vel_graph = np.zeros([num_node, iteration])
# all connectivity for graph
connectivity = np.zeros([iteration, 1])
# navigation feedback param
c1_mt = 20
c2_mt = 2 * np.sqrt(c1_mt)

def adj_matrix():
    matrix = distance_matrix(pos, pos)
    matrix[(matrix <= r) & (matrix > 0)] = 1
    matrix[np.where(matrix != 1)] = 0
    return matrix

def sigma_norm(z):
    val = EPSILON*(z**2)
    val = np.sqrt(1 + val) - 1
    val = val/EPSILON
    return val
    
def bump(z, h = 0.2):
    if 0 <= z < h:
        return 1
    elif h <= z <= 1:
        val = (z-h)/(1-h)
        val = np.cos(np.pi*val)
        val = (1+val)/2
        return val
    else:
        return 0

def spatial_adj(i, j):
    val_1 = pos[j] - pos[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm)/sigma_norm(r)
    val = bump(val_2)
    return val

def nij(i, j):
    val_1 = pos[j] - pos[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm**2
    val = val_1/np.sqrt(val_2)
    return val

def sigma1(z):
    val = 1 + z **2
    val = np.sqrt(val)
    val = z/val
    return val

def phi(z):
    val_1 = a + b
    val_2 = sigma1(z + c)
    val_3 = a - b
    val = val_1 * val_2 + val_3
    val = val / 2
    return val

def phi_alpha(z):
    input_1 = z/sigma_norm(r)
    input_2 = z - sigma_norm(d)
    val_1 = bump(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val

def u(i):
    gradient = np.array([0.0, 0.0])
    consensus = np.array([0.0, 0.0])
    for j in range(0, num_node):
        distance = np.linalg.norm(pos_old[j] - pos_old[i])
        if distance <= r:
            val_1 = pos_old[j] - pos_old[i]
            norm = np.linalg.norm(val_1)
            phi_alpha_val = phi_alpha(sigma_norm(norm))
            val = phi_alpha_val * nij(i, j)
            gradient += val
            val_2 = velocities[j] - velocities[i]
            consensus += spatial_adj(i, j) * val_2
    val = c1 * gradient + c2 * consensus
    return val

#main function for running
def main():
  global q_mt_x_old, q_mt_y_old
  for t in range(0, iteration):
    connectivity[t] = (1 / num_node) * np.linalg.matrix_rank (adj_matrix()) 
    if t == 0:
        plot_link(t)
        for i in range(0, num_node):
            x_arr[i, t] = pos[i, 0]
            y_arr[i, t] = pos [1, 1]
    else:
        for i in range(0, num_node):
            u_i = u(i)
            # calculate q, p and record
            old_pos = np.array([x_arr[i, t-1], y_arr[i, t-1]])
            new_vel = velocities[i, :] + u_i * delta_t
            new_pos = old_pos + delta_t * new_vel + (delta_t ** 2 / 2) * u_i
            [x_arr[i, t], y_arr[i, t]] = new_pos
            velocities[i, :] = new_vel
            pos[i, :] = new_pos
            # magnitude of each velocity
            vel_graph[i, t] = np.linalg.norm(new_vel)
            # adjust base on how many frames you want
    if (t+1) % 50 == 0:
        plot_link(t)

#plot functions
def plot_traj ():
    for i in range(0, num_node):
        plt.plot(x_arr[i, :], y_arr[i, :]) 
    plt.title("Trajectory")
    plt.show()
    
def plot_vel():
    for i in range(0, num_node):
        velocity_i = vel_graph[i, :]
        plt.plot(velocity_i)
    plt.title("Velocity")
    plt.show()
    
def plot_connectivity(): 
    m = connectivity 
    plt.plot(connectivity)
    plt.title("Connectivity")
    plt.show()

def plot_link(t):
    plt.scatter(pos[:,0], pos[:,1], marker = '.', color = 'black')
    for i in range(0, num_node):
        for j in range(0, num_node):
            if i != j and np.linalg.norm(pos[j] - pos[i]) < r:
                plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color ='blue')
    plt.show()

main()
plot_traj()
plot_vel()
plot_connectivity()