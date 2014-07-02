#module data_analysis_tools

from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans 

#This function takes a data set, list of centroids, colors[i] = j s.t. centroid[j] closest to data[i]
#Returns average radius of all clusters
#where radius is max distance from a centroid to an element in its cluster
#assuming len(data) = len(colors)
#k is number of clusters. I could figure this out from colors, but easier to pass it
def cluster_analysis(data, centroids, colors, k):
  hold = colors[0]
  centroidsmap = [] #holds the particular centroid closest to corresponding data entry
  radius = [0]*k #holds max radius
  dist = []
  for i in range(len(data)):
  	dist.append(euclidean(data[i], centroids[colors[i]]))
  
  for i in range(len(data)):
    if dist[i] > radius[colors[i]]:
      radius[colors[i]] = dist[i]

  return sum(radius)/k
  
def diameter(data):
	diam = 0
	for m in range(len(data)):
		for n in range(m, len(data)):
			if euclidean(data[n], data[m]) > diam:
				diam = euclidean(data[n], data[m])
				
	return diam
  
#I don't really know how to pick preference, so I'm gonna play around with it like I did with kmeans  
def afprop_analysis(data, centroids, colors):
  k = len(centroids)
  radius = [0]*k #holds max radius
  dist = []
  for i in range(len(data)):
  	dist.append(euclidean(data[i], centroids[colors[i]]))
  
  for i in range(len(data)):
    if dist[i] > radius[colors[i]]:
      radius[colors[i]] = dist[i]

  return sum(radius)/k
  
#This function takes array data and a maximum k =kcheck to look at and prints a plot of k-means clustering errors up to max k
def kcheck(data, kcheck):
  radiuscheck = [0]*(kcheck)
  for k in range(2, kcheck):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    colors = kmeans.labels_
    #centroids,colors = kmeans2(data, k)
    radiuscheck[k] = cluster_analysis(data, centroids, colors, k)
    
    
  plt.plot(radiuscheck)
  plt.ylabel("Avg max radius in a cluster")
  plt.xlabel("Number of clusters")
  plt.title("K-means curve")
  plt.show()
  
  
	
