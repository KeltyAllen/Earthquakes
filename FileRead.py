
import sys
import csv
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2,vq
from numpy import array
import data_analysis_tools as tools


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
	
		
	


def main():
  #if len(sys.argv) != 2:
  #  print 'usage: ./FileRead.py file'
  #  sys.exit(1)

  filename = sys.argv[1]
  readdata = csv.reader(open(filename))
  data = []
  
  for row in readdata:
    data.append(row)
    
  header = data[0]
  data.pop(0)
  x = [] #time data, v. rough format so far. Don't really need this though?
  year = []
  month = []
  day = []
  hour = []
  minute = []
  time = [] #time in years after start of data collection, (i.e. 1.345 years)
  mag = [] #magnitude
  temp = [] #for holding things temporarily
  a = ""
  timefloat = 0.0
  T = []
  centroids = []
  colors = []
  
 
  for i in range(len(data)):
  	mag.append(data[i][4])
  	x.append(data[i][0])
  	a = x[i]
  	year.append(float(a[0:4]))
  	month.append(float(a[5:7]))
  	day.append(float(a[8:10]))
  	hour.append(float(a[11:13]))
  	minute.append(float(a[14:16]))
		
	
	minyear = min(year)
  for i in range(len(year)):
  	timefloat = year[i] + (days_in_year(month[i]) + day[i])/365 + hour[i]/(24*365) +minute[i]/(60*24*365)
  	time.append(timefloat)
  	
  time = array(time)
  tools.kcheck(time, 20)

  k = int(raw_input('What k looks appropriate? '))

  centroids,colors = kmeans2(time, k)
  plt.scatter(time, mag, c = colors)
  plt.show()
  
  



	
  
  
  
 # readdata.close()
  
  
"""  if option == '--count':
    print_words(filename)
  elif option == '--topcount':
    print_top(filename)
  else:
    print 'unknown option: ' + option
    sys.exit(1)"""

if __name__ == '__main__':
  main()
  
  
  
  
  
