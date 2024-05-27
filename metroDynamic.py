import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

iteration = 80
delta_t = 0.008
num_node = 50
dimension = 2
area = 3
nodes = np.random.uniform(size=(num_node, dimension), high=area)
variance = 2
gt = 40
gt_array = np.full((num_node, 1), gt)
measure = gt + np.random.uniform(size=(num_node, 1), high=variance, low=-variance)
r = 1
measure_all = np.zeros([num_node, iteration])
w = np.zeros([num_node, num_node])

def num_neighbors():
  matrix =  distance_matrix(nodes,nodes)
  matrix[(matrix <= r) & (matrix > 0)] = 1
  matrix[np.where(matrix != 1)] = 0
  return matrix, np.sum(matrix, axis=1)

def plot_neighbors():
  plt.plot(nodes[:, 0], nodes[:, 1], 'o', color='black')
  for i in range(0, num_node):
      for j in range(0, num_node):
          distance = np.linalg.norm(nodes[j] - nodes[i])
          if distance <= r:
              plt.plot([nodes[i, 0], nodes[j, 0]],
                        [nodes[i, 1], nodes[j, 1]],
                        'b-', lw=0.5)
  plt.show()

def plot_value(measure):
  x = np.arange(0, num_node)
  plt.plot(x, measure, 'o', color='black')
  plt.plot(x, measure, color='black')
  plt.plot(x, gt_array, 'o', color='blue')
  plt.plot(x, gt_array, color='blue')
  plt.show()

def plot_covergence_error():
  x = np.arange(0, iteration)
  plt.title("Error Convergence")
  for i in range(0, num_node):
    plt.plot(x, gt - measure_all[i, :])
  plt.show()
  x = np.arange(0, iteration)
  plt.title("Mean Square Error Convergence")
  for i in range(0, num_node):
    plt.plot(x, (gt-measure_all[i, :])**2)
  plt.show()

def plot_min_max(degree_neighbors):
  x = np.arange(0, iteration)
  plt.title("Max and Min Neighbor Error Convergence")
  plt.plot(x, (gt - measure_all[np.argmax(degree_neighbors), :]), label="Max Neighbor")
  plt.plot(x, (gt - measure_all[np.argmin(degree_neighbors), :]), label="Min Neighbor")
  leg = plt.legend(loc='upper right')
  plt.show()
  plt.title("Max and Min Neighbor Mean Square Error Convergence")
  plt.plot(x, (gt - measure_all[np.argmax(degree_neighbors), :])**2, label="Max Neighbor")
  plt.plot(x, (gt - measure_all[np.argmin(degree_neighbors), :])**2, label="Min Neighbor")
  leg = plt.legend(loc='upper right')
  plt.show()

def metropolis(i, j, adj_matrix, degree_neighbors):
    if adj_matrix[i, j]:
        return 1 / (1 + max(degree_neighbors[i], degree_neighbors[j]))
    elif i == j:
        if not degree_neighbors[i]:
            return 1
        else:
            return 1 - sum(metropolis(i, neighbor, adj_matrix, degree_neighbors) for neighbor in range(len(adj_matrix[i])) if adj_matrix[i, neighbor])
    else:
        return 0

def main():
    for iter in range(iteration):
        if iter == 0:
            plot_neighbors()
            measure_all[:, iter] = measure.flatten()
            plot_value(measure.flatten())
        else:
            r_dynamic = r - np.random.uniform(0, 0.3)
            adj_matrix, degree_neighbors = num_neighbors()

            for i in range(num_node):
                w[i] = [metropolis(i, j, adj_matrix, degree_neighbors) for j in range(num_node)]
                measure_all[i, iter] = np.dot(w[i], measure_all[:, iter-1]) / np.sum(w[i])

    plot_value(measure_all[:, -1])
    plot_covergence_error()
    plot_min_max(degree_neighbors)

main()
