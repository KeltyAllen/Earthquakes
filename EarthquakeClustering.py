import sys
import csv
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from numpy import array
import data_analysis_tools as tools
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_pdf import PdfPages



#This function returns the days in months previous to the current month
#Note - I'm ignoring the existence of leap years
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
	 	 


#This returns distance between two points of lat & longitude in km
def distance_on_unit_sphere(lat1, long1, lat2, long2): #Code from Johndcook.com; thanks dude!

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
  degrees_to_radians = math.pi/180.0
        
    # phi = 90 - latitude
  phi1 = (90.0 - lat1)*degrees_to_radians
  phi2 = (90.0 - lat2)*degrees_to_radians
        
    # theta = longitude
  theta1 = long1*degrees_to_radians
  theta2 = long2*degrees_to_radians
    
  cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
         math.cos(phi1)*math.cos(phi2))
  arc = math.acos( cos )
  return arc*6373 #returns distance in km

#Writes the one-dimensional clustering to pdf pp
def one_dim_plot_save(timeinyears, mag, colors,density, pp):
  #density = density*10
  fig = plt.figure(figsize=(10, 5), dpi=100)
  if len(timeinyears) != len(mag):
    print "time length != mag length"  	
  plt.scatter(timeinyears, mag, c = colors)
  plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset = False))
  plt.title("Earthquake cluster with density %f"%density)
  plt.ylabel("Magnitude")
  plt.xlabel("Year")
  plt.savefig(pp, format='pdf')
  
    
#lontemp, lattemp, magtemp are arrays with data to map, avggyear is average year, goes in title, m is the map object with correct width, lat/lon, etc. 	
def two_dim_plot_save(lontemp, lattemp, magtemp, avgyear, m, pp):
  lat1 =min(lattemp)
  lat2 = max(lattemp)
  lon1 = min(lontemp)
  lon2 = max(lontemp)
  avglon = np.mean(lontemp)
  avglat = np.mean(lattemp)
  m.drawcoastlines()
  m.drawcountries()
  m.drawstates()
  m.fillcontinents(color='coral')
  m.drawmapboundary()  
  m.drawparallels(np.arange(lat1,lat2,1),labels=[1,0,0,0])
  m.drawmeridians(np.arange(lon1,lon2,1),labels=[0,0,0,1])
  for i in range(len(lontemp)):
    x, y = m(lontemp[i], lattemp[i])
    m.plot(x, y, 'bo', alpha = .6, markersize = 3*magtemp[i])   
  #Currently just gives the average of the cluster data, rather than the centroid, to minimize things passed 
  plt.title("Earthquakes, clustered around (%d, %d) %d" %(avglat, avglon, avgyear),fontsize=10 )
  plt.ylabel("")
  plt.xlabel("")
  plt.savefig(pp, format='pdf')    
    

def main():
	
	
	################
	#Data input
	################
	
  filename = sys.argv[1]
  
  x = [] #time data, initial format is something like 1900-01-01 00:00:00
  latitude = [] 
  longitude = []
  mag = []
  headerflag = 0
  
  with open(filename, 'rb') as f:
    readdata = csv.reader(f)
    for row in readdata:
      if headerflag == 0:  #getting header info
        xheader = row[0]
        latheader = row[1]
        lonheader = row[2]
        magheader = row[4]
        headerflag = 1
      else:
        x.append(row[0])
        latitude.append(float(row[1]))
        longitude.append(float(row[2]))
        mag.append(float(row[4])) 


  ################
  #Data manipulation 
  ################
  
  data = []
  timeinyears = [] #want this for prettier plots
  time = [] 
  temp = [] #for holding things temporarily. Not temperature, this is earthquake stuff, that would be silly
  longdist = [] #dist is in units of km, for running clustering algorithms
  latdist = []
  timefloat = 0.0
  centroids = []
  colors = []
  
  lat1 = min(latitude)
  lat2 = max(latitude)
  lon1 = min(longitude)
  lon2 = max(longitude)
  
  #The following is an approximate conversion to km from latitude & longitude
  lonperkm = abs((lon2 - lon1)/distance_on_unit_sphere((lat1 + lat2)/2, lon1, (lat1+lat2)/2, lon2))    
  latperkm = abs((lat2 - lat1)/distance_on_unit_sphere(lat1, (lon1 + lon2)/2, lat2, (lon1 + lon2)/2))
  
  #Data cleaning - time put into days, distances are recorded, with (minlat, minlon) as origin, rather than lat & lon
  #We're equating 1 day in time with 1 km (approximate) distance
  #This is super arbitrary
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
  
  #Info for the output file
  #outputpdf = raw_input("Where would you like to store the plots?\nFilename should end in .pdf: ")
  #if(outputpdf[-4:] != ".pdf"):
  	#outputpdf = outputpdf + ".pdf"
  outputpdf = filename[:-4]+".pdf"
  pp = PdfPages(outputpdf)
  metadata = pp.infodict()
  metadata['Title'] = 'Figures for %s'%filename
  
  #Saving the low density clusters to look at if I want
  junkname = outputpdf[:-4] + 'junk.pdf'
  pjunk = PdfPages(junkname)
  metadata = pjunk.infodict()
  metadata['Title'] = 'Figures for %s with low density (not an foreshock/aftershock group)'%filename
  
  print "There are ", len(data), " earthquakes to analyze."
  
  
  ################
  # Running clustering algorithms and plotting one-dimensional color-coded plots to see a big picture of all clusters in time
  ################
  
  #KMeans
  tools.kcheck(data, 70)
  k = int(raw_input('What k looks appropriate for kmeans clustering? '))
  #k = 12
  
  
  #Using sklearn kmeans:
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(data)
  centroids = kmeans.cluster_centers_
  kcolors = kmeans.labels_
  fig = plt.figure(figsize=(10, 5), dpi=100)
  if len(timeinyears) != len(mag):
    print "time length != mag length"  	
  plt.scatter(timeinyears, mag, c = kcolors)
  plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset = False))
  plt.title("Kmeans Earthquake Clustering")
  plt.ylabel("Magnitude")
  plt.xlabel("Year")
  plt.savefig(pp, format='pdf')
  
  

  
  
  #DBSCAN
  db = DBSCAN(eps = 40).fit(data)
  core_samples = db.core_sample_indices_
  labels = db.labels_ 

  
  # Number of clusters in labels, ignoring noise if present.
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  
  print "There are ", n_clusters, " clusters using DBSCAN\n"
  
  #Cluster diameter info; diam[j] = diameter of cluster whose elements are labeled by 'j' in labels
  #diameter is max distance between two elements
  diam = [0]*n_clusters
  
  clusterhold = [] #holds on to one cluster long enough to find its radius
  
  for j in range(0, n_clusters):
    for i in range(len(labels)):
      if labels[i] == j:
      	clusterhold.append(data[i])
    diam[j] = tools.diameter(clusterhold)
    clusterhold = []
  
  #print "Diameters are ", diam
  
  dbcolors = []
  
  for i in range(len(labels)):
    dbcolors.append(labels[i] + 3)
  
  fig = plt.figure(figsize=(10, 5), dpi=100)
  if len(timeinyears) != len(mag):
    print "time length != mag length"  	
  plt.scatter(timeinyears, mag, c = labels)
  plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset = False))
  plt.title("DBSCAN Earthquake Clustering")
  plt.ylabel("Magnitude")
  plt.xlabel("Year")
  plt.savefig(pp, format='pdf')

  density = [] #density is clustersizes/diameter
  
  m = Basemap(width=abs(lon2 - lon1)*1000/lonperkm,height=abs(lat1 - lat2)*1000/latperkm,projection='lcc',resolution='h',lat_0=(lat1 + lat2)/2,lon_0=(lon1 + lon2)/2)
  clustersizes = []
    
  ################  
  # Record the densest clusters
  ################
  
  for j in range(0, n_clusters):   #should be range(0, n_clusters)
    lontemp = []
    lattemp = []
    timetemp = []
    magtemp = []
    colortemp = []
    for i in range(len(labels)):
      if labels[i] == j:
        lontemp.append(longitude[i])
        lattemp.append(latitude[i])
        timetemp.append(timeinyears[i])
        magtemp.append(mag[i])
        colortemp.append(j)
    avgyear = np.mean(timetemp)
    clustersizes.append(len(lontemp)) 
    density.append(clustersizes[j]/diam[j])
    #Bigger clusters get shown and saved
    if (density[j] > .25):
      one_dim_plot_save(timetemp, magtemp, colortemp, density[j], pp)
      two_dim_plot_save(lontemp, lattemp, magtemp, avgyear, m, pp)      
      outfilename = "output_%s_%d.csv"%(filename[:-4], j)
      with open(outfilename, 'wb') as csvfile:
        writing = csv.writer(csvfile, delimiter = ',')
        writing.writerow([xheader, latheader, lonheader, magheader])
        for k in range(len(lontemp)):
          writing.writerow([timetemp[k], lattemp[k], lontemp[k], magtemp[k]])
    else:
      one_dim_plot_save(timetemp, magtemp, colortemp, density[j], pjunk)
      two_dim_plot_save(lontemp, lattemp, magtemp, avgyear, m, pjunk)
            
   
  #Recording the noise, just to see it 
  
  lontemp = []
  lattemp = []
  timetemp = []
  magtemp = []
  colortemp = [] 
  for i in range(len(labels)):
    if labels[i] == -1:
      lontemp.append(longitude[i])
      lattemp.append(latitude[i])
      timetemp.append(timeinyears[i])
      magtemp.append(mag[i])
      colortemp.append(0)
  
  #Saving 1-d plot
  fig = plt.figure(figsize=(10, 5), dpi=100)   
  plt.scatter(timetemp, magtemp, c = colortemp)
  plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset = False))
  plt.title("DBSCAN Noise - unclustered quakes")
  plt.savefig(pp, format='pdf')
  
  
  #Saving 2-d plot
  avglon = np.mean(lontemp)
  avglat = np.mean(lattemp)
  m.drawcoastlines()
  m.drawcountries()
  m.drawstates()
  m.fillcontinents(color='coral')
  m.drawmapboundary()
  m.drawparallels(np.arange(lat1,lat2,1),labels=[1,0,0,0])
  m.drawmeridians(np.arange(lon1,lon2,1),labels=[0,0,0,1])

  for i in range(len(lontemp)):
    x, y = m(lontemp[i], lattemp[i])
    m.plot(x, y, 'bo', alpha = .6, markersize = 3*magtemp[i])
  plt.title("DBSCAN Noise - unclustered earthquakes")
  plt.savefig(pp, format='pdf') 
  #Close the file
  pp.close()
  pjunk.close()
  
  
  
if __name__ == '__main__':
  main() 
