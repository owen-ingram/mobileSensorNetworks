import matplotlib.pyplot as plt
import numpy as np

numNodes = 50
r=15
area = 50
iteration = 200
neighborList=[0]*numNodes
V1=[0]*numNodes
n1=[0]*numNodes
zi=[0]*numNodes
errorc1=np.zeros([iteration,numNodes])
errorc2=np.zeros([iteration,numNodes])
errorAvgc1=np.zeros([iteration,numNodes])
errorAvgc2=np.zeros([iteration,numNodes]) 
groundTruth=40

def findneighbors(n,Xi,r):

    allNi=[]
    
    for j in range(n):
        neighbors = []
        for i in range(n):
            if i != j and np.linalg.norm(Xi[i] - Xi[j]) <= r:
                neighbors.append(i)
        allNi.append(neighbors)

    return allNi

def consensus1(Xi,Ni,A,e):
    newXi = Xi.copy()
    num_nodes = len(Xi)
    
    for i in range(num_nodes):
        neighbors = Ni[i]
        for j in neighbors:
            newXi[i] += e * A[i][j] * (Xi[j] - Xi[i])
    
    return newXi

def consensus2(Xi,Ni):
    newXi = []
    num_nodes = len(Xi)

    for i in range(num_nodes):
        sum_neighbors = sum(Xi[j] for j in Ni[i])
        newXi.append((1 / (1 + len(Ni[i]))) * (Xi[i] + sum_neighbors))

    return newXi

def adjacencyMatrix(Ni):
    AMatrix = np.zeros((numNodes, numNodes))

    for i in range(len(Ni)):
        for j in Ni[i]:
            AMatrix[i][j] = 1
    
    return AMatrix


def main():
    Xi=np.random.random((numNodes,2))
    Xi= area*Xi

    e=.02

    target=np.array([25,25])

    neighborList=findneighbors(numNodes,Xi,r)
    
    for j in range(numNodes):
        cov_mat=np.stack((Xi[j],target), axis = 1)
        Calccov=np.cov(cov_mat)
        cv=(Calccov[0,0]-Calccov[1,1])/100
        
        V1[j]=((np.linalg.norm(Xi[j]-target)**2)+cv)/(r**2)
        n1[j]=np.random.normal(0.0,V1[j])
        zi[j]=groundTruth+n1[j]
    
    xic1=zi.copy()
    xic2=zi.copy()

    initialaverage=sum(zi)/len(zi)
    mfirst=zi.copy()

    iter=0
    while (iter < iteration):

        A = adjacencyMatrix(neighborList)

        xic1=consensus1(xic1,neighborList,A,e)
        xic2=consensus2(xic2,neighborList)

        for x in range(numNodes):
            
            errorc1[iter,x]=xic1[x]-groundTruth
            errorc2[iter,x]=xic2[x]-groundTruth
            errorAvgc1[iter,x]=xic1[x]-initialaverage
            errorAvgc2[iter,x]=xic2[x]-initialaverage
        
        iter+=1

    plt.figure(1)
    plt.plot(errorc1)
    plt.title("Error of all nodes consensus 1")

    plt.figure(2)
    plt.plot(errorc2)
    plt.title("Error of all nodes consensus 2")

    plt.figure(3)
    plt.plot(errorAvgc1)
    plt.title("Error between average of all nodes consensus 1")

    plt.figure(4)
    plt.plot(errorAvgc2)
    plt.title("Error between average of all nodes consensus 2")

    plt.figure(5)
    plt.plot(mfirst,label='initial',color='blue',marker='o',markerfacecolor='blue',markersize=4)
    plt.plot(xic1,label='last',color='orange',marker='o',markerfacecolor='orange',markersize=4)
    plt.legend()
    plt.title("Initial and final measurement consensus 1")

    plt.figure(6)
    plt.plot(mfirst,label='initial',color='blue',marker='o',markerfacecolor='blue',markersize=4)
    plt.plot(xic2,label='last',color='orange',marker='o',markerfacecolor='orange',markersize=4)
    plt.legend()
    plt.title("Initial and final measurement consensus 2")

    plt.show()

if __name__ == "__main__":
    main()
