"""
Author: Muhammad Ahmed
VANILLA KMEANS
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate the Data
n1 = 1000
n2 = 100

# random ellipse 1 centered at 0,0
x = np.random.randn(n1+n2,1)
y = 0.5 * np.random.randn(n1+n2,1)

# random ellipse 2 centered at (1,-2) and rotated by a certain angle theta
x2 = np.random.randn(n1+n2,1) + 1
y2 = 0.2 * np.random.randn(n1+n2,1) -2
theta = np.pi/4

# define a rotation matrix
A = np.array([ [np.cos(theta),-1*np.sin(theta)],[np.sin(theta), np.cos(theta)] ])

# perform the rotation
x3 = A[0,0]*x2 + A[0,1]*y2
y3 = A[1,0]*x2 + A[1,1]*y2

# plotting
f, (ax1) = plt.subplots(1, 1, figsize=(12, 6), sharey=True)

# Scatter plot for cluster 1
# Choose a dark color like 'darkgreen' with a white edgecolor for contrast
ax1.scatter(x, y, color='darkgreen', marker='o', edgecolor='white', s=50)  # Increased size for visibility

# Set title and labels with a default white background
ax1.set_title('Data Cluster', fontsize=14)
ax1.set_xlabel('X-axis', fontsize=12)
ax1.set_ylabel('Y-axis', fontsize=12)

# Scatter plot for cluster 2
# Choose a dark color like 'navy' with a white edgecolor for contrast
ax1.scatter(x3, y3, color='navy', marker='^', edgecolor='white', s=50)  # Increased size for visibility

# Set grid for better visibility and adjust layout
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()



# Training set for Kmeans
X1 = np.hstack ((x3[:n1] , y3[:n1]))
X2 = np.hstack ((x[:n1],y[:n1]))

Y = np.vstack((X1,X2))
Z = np.hstack( (np.ones((n1,1)), 2*np.ones((n1,1))))

# Test data
x1test = np.hstack((x3[n1+1,:],y3[n1+1,:]))
x2test = np.hstack((x[n1+1,:],y[n1+1,:]))

# lets begin with Kmeans
# choose an initial guess
random_val_1 = np.random.randint(100)
g1x = x3[random_val_1]
g1y = y3[random_val_1]
g1 = np.array([g1x, g1y]).flatten()
random_val_2 = np.random.randint(100)
g2x = x[random_val_2]
g2y = y[random_val_2]
g2 = np.array([g2x, g2y]).flatten()


# Plot the points g1 and g2 on the scatter plot with increased size and distinct markers
ax1.plot(g1[0], g1[1], 'r*', markersize=15, label='Initial guess 1')  # Red star for g1
ax1.plot(g2[0], g2[1], 'b*', markersize=15, label='Initial guess 2')  # Blue star for g2

# Add a legend to the plot
ax1.legend(loc='upper right')


# Start comparing each points in the data with the
# inital guess and put the data in two clusters,
# cluster 1 and cluster 2

# data in X1 and X2  with 2 columns one is x and the other is y
# we combined the data to form Y
# Traverse through Y take each row find the norm of that row with the g1 and g2 respectively

max_iterations = 100  # Adjust iteration according to your need
for j in range(max_iterations):
    cluster1 = np.empty((0,2))
    cluster2 = np.empty((0,2))
    for i in range(len(Y)):
        point_1 = Y[i,:]
        # norm is calculated for the distance between the X1 row i and the point g1
        norm_1 = np.linalg.norm(g1 - point_1)
        point_2 = Y[i, :]
        # norm is calculated for the distance between the X2 row i and the point g2
        norm_2 = np.linalg.norm(g2 - point_2)
        if norm_1 < norm_2:
            cluster1 = np.vstack([cluster1, [Y[i, 0], Y[i, 1]]])
        else:
            cluster2 = np.vstack([cluster2, [Y[i, 0], Y[i, 1]]])

    # update initial guess values
    if cluster1.size > 0:
        g1 = np.array([np.mean(cluster1[:, 0]), np.mean(cluster1[:,1])])
    else:
        random_val_1 = np.random.randint(100)
        g1x = x3[random_val_1]
        g1y = y3[random_val_1]
        g1 = np.array([g1x, g1y]).flatten()
    if cluster2.size > 0:
        g2 = np.array([np.mean(cluster2[:, 0]), np.mean(cluster2[:,1])])
    else:
        random_val_2 = np.random.randint(100)
        g2x = x[random_val_2]
        g2y = y[random_val_2]
        g2 = np.array([g2x, g2y]).flatten()
    # Plot the points g1 and g2 on the scatter plot with increased size and distinct markers
    ax1.plot(g1[0], g1[1], 'ro', markersize=7)
    ax1.plot(g2[0], g2[1], 'bo', markersize=7)

    # update on the same plot
    plt.draw()

plt.show()  # Show the final plot


