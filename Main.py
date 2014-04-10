
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
from sklearn.cluster import KMeans
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
	
	
#lontemp, lattemp, magtemp are arrays with data to map, avggyear is average year, goes in title, m is the map object with correct width, lat/lon, etc. 	
def two_dim_plot(lontemp, lattemp, magtemp, avgyear, m):
  avglon = np.mean(lontemp)
  avglat = np.mean(lattemp)
  m.drawcoastlines()
  m.drawcountries()
  m.drawstates()
  m.fillcontinents(color='coral')
  m.drawmapboundary()
  
  for i in range(len(lontemp)):
    x, y = m(lontemp[i], lattemp[i])
    m.plot(x, y, 'bo', alpha = .6, markersize = 2*magtemp[i])
   
  #Currently just gives the average of the cluster data, rather than the centroid, to minimize things passed 
  plt.title("Earthquakes, clustered around (%d, %d) %d" %(avglat, avglon, avgyear),fontsize=10 )
  plt.show()



def main():
	
  latperkm = (35.826 - 32.996)/314.7
	
  #IMPORTANT NOTE: THE FOLLOWING CONSTANTs are DEPENDENDENT ON REGION 
  lat1 = 32.996
  lat2 = 35.826
  lon1 = -121.84
  lon2 = -115.402
  lonperkm = (lon2 - lon1)/590.3    #Const for SoCal20002010.csv


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
  
  #for i in range(len(data)):
  	#print "time: ", time[i], "data: ", data[i]

  
  #This function now uses sklearns kmeans. It is slooooow. It's faster but less useful with scipy kmeans algorithm
  #tools.kcheck(data, 60)

  #k = int(raw_input('What k looks appropriate? '))

  k = 20
  
  
  #Using sklearn kmeans:
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(data)
  centroids = kmeans.cluster_centers_
  colors = kmeans.labels_
  
  
  lontemp = []
  lattemp = []
  timetemp = []
  magtemp = []
  
  for j in range(0, 3):   #should be range(0, k)
    lontemp = []
    lattemp = []
    timetemp = []
    magtemp = []
    m = []
    for i in range(len(colors)):
      if colors[i] == j:
    	  lontemp.append(longitude[i])
    	  lattemp.append(latitude[i])
    	  timetemp.append(timeinyears[i])
    	  magtemp.append(mag[i])
    avgyear = np.mean(timetemp)
    m = Basemap(width=abs(lon2 - lon1)*1000/lonperkm,height=abs(lat1 - lat2)*1000/latperkm,projection='lcc',resolution='h',lat_0=(lat1 + lat2)/2,lon_0=(lon1 + lon2)/2)
    two_dim_plot(lontemp, lattemp, magtemp, avgyear, m)
    
    #m.drawcoastlines()
    #m.drawcountries()
    #m.drawstates()
    #m.fillcontinents(color='coral')
    #m.drawmapboundary()
  
  #for i in range(len(lontemp)):
    #x, y = m(lontemp[i], lattemp[i])
    #m.plot(x, y, 'bo', markersize = 2*mag[i])
    
  #plt.title("Earthquakes, around year %d" %avgyear,fontsize=10 )
  #plt.show()
  
  #one_dim_plot(timeinyears, mag, colors)
  
  #for i in range(min(colors), max(colors)):
    #m = Basemap(width=abs(lon2 - lon1)*1000/lonperkm,height=abs(lat1 - lat2)*1000/latperkm,projection='lcc',resolution='h',lat_0=(lat1 + lat2)/2,lon_0=(lon1 + lon2)/2)
    #m.drawcoastlines()
    #m.drawcountries()
    #m.drawstates()
    #m.fillcontinents(color='coral')
    #m.drawmapboundary()
    #x, y = m(longitude, latitude)
    #m.plot(x, y, 'bo')
    #plt.title("Earthquakes?",fontsize=10)
    #plt.show()
  

  

if __name__ == '__main__':
  main()
  
  
  
  
  
