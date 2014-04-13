import sys
import csv
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.cluster.vq import kmeans2,vq
from numpy import array
import data_analysis_tools as tools
from sklearn.cluster import KMeans, AffinityPropagation
from mpl_toolkits.basemap import Basemap


#This function returns the days in months previous to the current month
def days_in_year(monthnum):
	if monthnum == 1:
		return 0
	elif monthnum in [5, 7, 8, 10, 12]:
	  return 30+days_in_year(monthnum-1)
	elif monthnum in [2, 4, 6, 9, 11]:
	 	 return 31+days_in_year(monthnum - 1)
	elif monthnum == 3:
	 	 return 28 + days_in_year(monthnum - 1)
	else:
	 	 print "Invalid month"
	 	 sys.exit(1)
	
def one_dim_plot(timeinyears, mag, colors):
  fig = plt.figure(figsize=(10, 5), dpi=100)
  if len(timeinyears) != len(mag):
    print "time length != mag length"  	
  plt.scatter(timeinyears, mag, c = colors)
  plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
  plt.title("Earthquakes")
  plt.ylabel("Magnitude")
  plt.xlabel("Year")
  plt.show() 	
	 	
def main():
	
  latperkm = (35.826 - 32.996)/314.7
	
  #IMPORTANT NOTE: THE FOLLOWING CONSTANTs are DEPENDENDENT ON REGION 
  lat1 = 32.996
  lat2 = 35.826
  lon1 = -121.84
  lon2 = -115.402
  lonperkm = (lon2 - lon1)/590.3    #Const for SoCal20002010.csv & 3DSoCal20002010.csv
	
  filename = sys.argv[1]
  x = [] #time data, v. rough format so far
  latitude = [] 
  longitude = []
  mag = []
  index = 0
  
  with open(filename, 'rb') as f:
    readdata = csv.reader(f)
    for row in readdata:
      if index>0:
        x.append(row[0])
        latitude.append(float(row[1]))
        longitude.append(float(row[2]))
        mag.append(float(row[4])) 
      index = index + 1
    
  data = []
  timeinyears = [] #want this for prettier plots
  time = [] 
  temp = [] #for holding things temporarily. Not temperature, this is earthquake stuff, that would be silly
  longdist = [] #dist is in units of km, for running clustering algorithms
  latdist = []
  timefloat = 0.0
  centroids = []
  colors = []
  
 
  for i in range(len(x)):
    timefloat = float(x[i][0:4]) + (days_in_year(float(x[i][5:7])) + float(x[i][8:10]))/365 + float(x[i][11:13])/(24*365) + float(x[i][14:16])/(60*24*365)
    #ok units currently in years, let's change to days
    timeinyears.append(timefloat)
    timefloat = timefloat*365
    time.append(timefloat)
    longdist.append((float(longitude[i]) - lon1)/lonperkm)
    latdist.append((float(latitude[i]) - lat1)/latperkm)
    data.append([longdist[i], latdist[i], time[i]])
	
  time = array(time)
  longdist = array(longdist)
  latdist = array(latdist)
  
  data = array(data)
  
  #preference? 
  af = AffinityPropagation().fit(data)
  cluster_centers_indices = af.cluster_centers_indices_
  labels = af.labels_
  
  print cluster_centers_indices
  print "\n\n"
  print labels
  print "len(cluster_centers_indices is ", len(cluster_centers_indices)
  print "And len(data) is ", len(data)
  
  
  
  

if __name__ == '__main__':
  main()
