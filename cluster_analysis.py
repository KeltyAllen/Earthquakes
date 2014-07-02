import sys
import csv
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from numpy import array
from scipy.optimize import curve_fit

def omori_utsu_total(x, c, k, p): #this is the integral of normal omori-utsu; gives total quakes by time t.
	return -(k)/(p*(c + x)**(p-1))
	
def gutenberg_richter(x, a, b): #gutenberg-richter law; predicts number of quakes that occur above given magnitude
  return 10**(a-b*x) 


def main():
  filename = sys.argv[1]
  
  x = [] #time data, in years as a float (e.g. 2008.55 is in early July 2008)
  latitude = [] 
  longitude = []
  magnitude = []
  headerflag = 0
  
  with open(filename, 'rb') as f:
    readdata = csv.reader(f)
    for row in readdata:
      if headerflag == 0:  #getting header info
        xheader = row[0]
        latheader = row[1]
        lonheader = row[2]
        magheader = row[3]
        headerflag = 1
      else:
        x.append(float(row[0]))
        latitude.append(float(row[1]))
        longitude.append(float(row[2]))
        magnitude.append(float(row[3]))
     
  ##########
  #omori-utsu comparison
  ##########
     
  #Sort by time:
  x, latitude, longitude, magnitude = (array(t) for t in zip(*sorted(zip(x, latitude, longitude, magnitude))))
  
  #For comparison to omori-utsu theory, want to start our cluster at a large early quake  
  magcheck = []
  maxindex = 0
  for i in range(0, len(x)/8):
    magcheck.append(magnitude[i])
  for i in range(len(magcheck)):
    if max(magcheck) == magcheck[i]:
      maxindex = i
      break
  
  time = x[maxindex:len(x)]
  lat = latitude[maxindex:len(latitude)]
  lon = longitude[maxindex:len(longitude)]
  mag = magnitude[maxindex:len(magnitude)]
  
  starttime = time[0]
  for j in range(len(time)):
    time[j] = time[j] - starttime
  
    
  #Omori - utsu: x-values are time, y-values are number of quakes at that time:
  oux = []
  ouy = []
  for i in range(len(time)):
    oux.append(time[i])
    ouy.append(i+1)
  
 
  
  #print zip(oux, ouy)
  popt,_ = curve_fit(omori_utsu_total, oux, ouy, [.0001, 1, 1])
  
  print "omori-utsu constants for ", filename, " are c, k, p = ", popt
 
  plotdata = []
  for i in range(len(oux)):
  	plotdata.append(omori_utsu_total(oux[i], popt[0], popt[1], popt[2]))
 
  fig = plt.figure(figsize=(10, 5), dpi=100)
  plt.scatter(oux, ouy)
  plt.plot(oux, plotdata)
  plt.title("Number of quakes over time, actual vs. compare to Omori-Utsu Law Prediction")
  plt.ylabel("Number of quakes")
  plt.xlabel("Time after first big quake (years)")
  plt.show()
    
  ##########
  #Gutenberg- Richter Comparison
  ##########
  
  #sort by magnitude
  magnitude, x, latitude, longitude = (array(t) for t in zip(*sorted(zip(magnitude, x, latitude, longitude))))
  
  grx = []
  gry = []
  
  for i in range(len(magnitude)):
  	grx.append(magnitude[i])
  	N = len(magnitude)-i
  	gry.append(float(N)/float(len(magnitude)))
  	
  popt,_ =curve_fit(gutenberg_richter, grx, gry, [3, 1])
  
  print "gutenberg-richter constants for ", filename, " are a, b = ", popt
  
  grdata = []
  for i in range(len(grx)):
  	grdata.append(gutenberg_richter(grx[i], popt[0], popt[1]))
 
  fig = plt.figure(figsize=(10, 5), dpi=100)
  plt.scatter(grx, gry)
  plt.plot(grx, grdata)
  plt.title("Number of quakes above magnitude M, actual vs. compare to Gutenberg-Richter Law Prediction")
  plt.ylabel("# of quakes above magnitude M/total number of quakes")
  plt.xlabel("Magnitude")
  plt.show()      
        
        
if __name__ == '__main__':
  main() 
