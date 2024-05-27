import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

EPSILON = 0.1
H = 0.2
N = 150
M = 2
D = 15
K = 1.2
R = K*D
A = 5
B = 5
C = np.abs(A - B)/np.sqrt(4*A*B)
DELTA_T = 0.009
ITERATION_VALUES = np.arange(0, 15, DELTA_T)
ITERATION = ITERATION_VALUES.shape[0]
nodes = np.random.uniform(size =(N,M), high = 70)
nodes_old = nodes
nodes_velocity_p = np.zeros([N, M])
POSITION_X = np.zeros([N, ITERATION])
POSITION_Y = np.zeros([N, ITERATION])
velocity_magnitudes = np.zeros([N, ITERATION])
connectivity = np.zeros([ITERATION, 1])
target_points = np.zeros([ITERATION, M])
center_of_mass = np.zeros([ITERATION, M])
fig = plt.figure()
C1_ALPHA = 320
C2_ALPHA = 2*np.sqrt(C1_ALPHA)
c1_mt = 4
c2_mt = 2*np.sqrt(c1_mt)
q_mt = np.array([150, 150])
q_mt_x1 = 50
q_mt_y1 = 100
q_mt_x1_old = q_mt_x1
q_mt_y1_old = q_mt_y1
obstacles = np.array([[100, 25],[200, 25]])
Rk = np.array([15, 30])
num_obstacles = obstacles.shape[0]
c1_beta = 1500
c2_beta = 2*np.sqrt(c1_beta)
r_prime = 0.22 * K * R
d_prime = 10
def adj_matrix():
    matrix = distance_matrix(nodes, nodes)
    matrix[(matrix <= R) & (matrix > 0)] = 1
    matrix[np.where(matrix != 1)] = 0
    return matrix

def sigma_norm(z):
    return (np.sqrt(1+EPSILON * z*z) -1)/2
r_beta = sigma_norm(np.linalg.norm(r_prime))
d_beta = sigma_norm(np.linalg.norm(d_prime))
r_alpha = sigma_norm(R)
d_alpha = sigma_norm(D)

def bump(z):
        if 0 <= z < H:
            return 1
        elif H <= z < 1:
            val = ((1+np.cos(np.pi*((z-H)/(1-H))))/2)
            return val
        else:
            return 0
            
def sigma_1(z):
    val = (z / np.sqrt(1 + z **2))
    return val

def phi(z):
    val = (((A+B)*(sigma_1(z+C))+(A-B))/2)
    return 
    
def phi_alpha(z):
    val = bump(z/r_alpha)*(phi(z - d_alpha))
    return val
    
def phi_beta(z):
    val1 = bump(z/d_beta)
    val2 = sigma_1(z-d_beta) - 1
    return val1 * val2  
  
def get_a_ij(i, j):
    return bump(sigma_norm(np.linalg.norm(nodes[j] - nodes[i])) / sigma_norm(R))

def get_n_ij(i, j):
    val = ((nodes[j] - nodes[i])/(np.sqrt(1 + EPSILON * np.linalg.norm(nodes[j] - nodes[i])**2)))
    return val

def get_u_i(i, mu, a_k, P, p_i_k, q_i_k, b_i_k, n_i_k, old_position):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            val_1 = nodes[j] - nodes[i]
            norm = np.linalg.norm(val_1)
            phi_alpha_val = phi_alpha(sigma_norm(norm))
            val = phi_alpha_val * get_n_ij(i, j)
            sum_1 += val

            val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
            sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2 - c1_mt * (nodes[i] - q_mt) + \
          c1_beta * phi_beta(sigma_norm(np.linalg.norm(q_i_k - old_position))) * n_i_k + \
          c2_beta * b_i_k * (p_i_k - nodes_velocity_p[i])
    return val

def plot_neighbors(t):
    plt.plot(target_points[0:t, 0], target_points[0:t, 1])
    for i in range(0, num_obstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro', color='black')
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=0.5)
    plt.show()
    
def plot_trajectory():
    plt.title("Trajectory")
    for i in range(0, num_obstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstacles[i ,0], obstacles[i, 1]), Rk[i], color='red'))
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
    plt.show()
    
def plot_velocity():
    plt.title("velocities")
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        plt.plot(velocity_i)
    plt.show()
    
def plot_connectivity():
    plt.title("Connectivity")
    plt.plot(connectivity)
    plt.show()
    
def plot_center_of_mass():
    plt.title("Center of Mass")
    for i in range (0, num_obstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])
    plt.show()

def main():
    for t in range(0, ITERATION):
        adjacency_matrix = adj_matrix()
        connectivity[t] = (1/N) * np.linalg.matrix_rank(adjacency_matrix)
        center_of_mass[t] = np.array([np.linalg.matrix_rank(adjacency_matrix)])
        if t == 0:
            plot_neighbors(t)
            target_points[t] = np.array([q_mt_x1_old, q_mt_y1_old])
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                old_velocity = nodes_velocity_p[i, :]
                old_position = np.array([POSITION_X[i, t-1], POSITION_Y[i, t-1]])

                mu = Rk/np.linalg.norm((old_position - obstacles[0]))
                a_k = (old_position - obstacles[0])/np.linalg.norm((old_position - obstacles[0]))
                P = 1 - np.matmul(a_k.T, a_k)
                p_i_k = mu * P * old_velocity
                q_i_k = mu * old_position + (1 - mu) * obstacles[0]
                b_i_k = bump(sigma_norm(np.linalg.norm
                                                 (q_i_k - old_position))/d_beta)
                n_i_k = (q_i_k - old_position)/(np.sqrt(1 + EPSILON *
                                                        (np.linalg.norm(q_i_k - old_position))**2))
            u_i = get_u_i(i, mu, a_k, P, p_i_k, q_i_k, b_i_k, n_i_k, old_position)
            new_position = old_position + DELTA_T * old_velocity+ (DELTA_T ** 2 / 2) * u_i
            new_velocity = (new_position - old_position) / DELTA_T
            [POSITION_X[i, t], POSITION_Y[i, t]] = new_position
            nodes_velocity_p[i, :] = new_velocity
            nodes[i, :] = new_position
            velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
        if (t+1) % 200 == 0:
            plot_neighbors(t)
            
main()
plot_trajectory()
plot_velocity()
plot_connectivity()
plot_center_of_mass()
