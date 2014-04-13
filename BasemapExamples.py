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


def main(): 
  latperkm = (35.826 - 32.996)/314.7
	
  #IMPORTANT NOTE: THE FOLLOWING CONSTANTs are DEPENDENDENT ON REGION  
  lonperkm = (121.84 - 115.402)/590.3    #Const for SoCal20002010.csv
  
  lat1 = 32.996
  lat2 = 35.826
  lon1 = -121.84
  lon2 = -115.402
  
  m = Basemap(width=abs(lon2 - lon1)*1000/lonperkm,height=abs(lat1 - lat2)*1000/latperkm,projection='lcc',resolution='h',lat_0=(lat1 + lat2)/2,lon_0=(lon1 + lon2)/2)
  m.drawcoastlines()
  m.drawcountries()
  m.drawstates()
  m.fillcontinents(color='coral')
  m.drawmapboundary()
  plt.title("Earthquakes?",fontsize=10)
  plt.show()
  
  
  
if __name__ == '__main__':
  main()
