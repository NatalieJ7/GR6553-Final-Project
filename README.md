# GR6553-Final-Project
This is my final project for GR6553. This repository will contain all data, code, and output files used for this project. 

Project Overview:
This project tells the story of the landfall of Hurricane Idalia on Florida's Gulf Coast. This hurricane is unique because it went through an Eyewall Replacement Cycle (ERC) while making landfall. Since the hurricane was so close to the coast, there are more data types available. This project will showcase gridded model data, radar data, and dropsonde data. These different data types will help tell the story of the landfall of Idalia by generating plots to showcase Idalia's transition onto land and through the ERC. All of the code is contained in one file with different sections to house the different plot types. 

RADAR Overview:
The RADAR data was gathered from the Tallahassee, FL site since this site is closest to where the landfall occurred. The RADAR data allows for the visualization of the ERC and the way that the landfall changes the structure of the storm. The code for these images uses a loop to go through multiple timestamps of RADAR imagery. It pulls out the reflectivity data and sets a map extent to plot the data centered off the Florida coast.

Dropsonde Overview: 
The Dropsonde data was all gathered from the NOAA's Atlantic Oceanic & Meteorological Laboratory. This site houses all of the data from the research flights conducted over the hurricanes. There were many flights conducted for Idalia, but the focus date for this project was August 30th, and the flight on that date was conducted by the U.S. Air Force Weather Reconnaissance Squadron out of Biloxi, MS. The code for these images also uses a loop to run through all the timestamps provided from the flight. This code pulls the data, identifies the temperature and dew point variables, and then plots everything on Skew-T/Log-p diagram. 

Gridded Model Overview: 
The gridded model data run was gathered from the Global Forecast System (GFS). Specifically, the model run is from August 30th at 12Z. This data provides many variables throughout the atmosphere to help explain the conditions present at the time of the landfall. The code for this data is designed to produce a figure showcasing the 850 hPa temperature and geopotential heights. This figure showcases the temperature advection that is occurring in the low levels. 

Surface Analysis Overview: 
The surface data was also gathered from the GFS model. For this plot, however, the goal is to analyze where the lowest pressures are to showcase where Hurricane Idalia is located. The code for this pulls the surface geopotential heights and then finds the local minimums. Once it finds the minimums, a plot is created, and the height contours are plotted. Lastly, the code places a 'L' over the lowest pressure locations to create an easy visualization of the surface pressures. 
