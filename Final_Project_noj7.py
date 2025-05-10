# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:13:34 2025

@author: natal

This code is for creating imagery to tell the story of the landfall of Hurricane Idalia. 
There are four different images created within this code. They code itself is broken into different
sections to allow for individual running as well as the whole code. 
This code is set up to save all of the images created so if you do not wish to save the images make 
sure that you remove those lines or comment them out before running this code. 

"""
#Import all of the packages necessary for each of the sections below

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from metpy.calc import azimuth_range_to_lat_lon
from metpy.io import Level2File
from metpy.plots import USCOUNTIES
from metpy.units import units
from metpy.plots import SkewT
import netCDF4
import os
import pygrib
import cartopy 
import cartopy.feature as cf
from metpy.plots.ctables import registry
from scipy.ndimage import minimum_filter

#%%
# Code for RADAR loops

# List of radar files
radar_files = [
    'KTLH20230830_094403_V06',
    'KTLH20230830_095050_V06',
    'KTLH20230830_095714_V06',
    'KTLH20230830_100339_V06',
    'KTLH20230830_101003_V06',
    'KTLH20230830_101617_V06',
    'KTLH20230830_102231_V06',
    'KTLH20230830_102845_V06',
    'KTLH20230830_103521_V06',
    'KTLH20230830_104210_V06',
    'KTLH20230830_104846_V06',
    'KTLH20230830_105533_V06',
    'KTLH20230830_110148_V06',
    'KTLH20230830_110837_V06'
]

# Loop through each file 
for radar_file in radar_files:
    f = Level2File(radar_file)
    timestamp = f.dt.strftime('%Y-%m-%d %H:%M UTC')
    sweep = 0

    # Extract the azimuths and convert to edge format
    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    diff = np.diff(az)
    crossed = diff < -180
    diff[crossed] += 360.
    avg_spacing = diff.mean()
    az = (az[:-1] + az[1:]) / 2
    az[crossed] += 180.
    az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
    az = units.Quantity(az, 'degrees')

    # Get Reflectivity variable
    ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
    ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
    ref_range = units.Quantity(ref_range, 'kilometers')
    ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

    cent_lon = f.sweeps[0][0][1].lon
    cent_lat = f.sweeps[0][0][1].lat

    # Create a single subplot figure
    fig = plt.figure(figsize=(12, 10), layout='constrained')

    # Only reflectivity (REF) data now
    var_list = [
        (ref, ref_range, 'Reflectivity (dBZ)', mpl.cm.Spectral_r, mpl.colors.Normalize(vmin=0, vmax=60))
    ]

    for data, rng, title, cmap, norm in var_list:
        xlocs, ylocs = azimuth_range_to_lat_lon(az, rng, cent_lon, cent_lat)

        crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
        ax = fig.add_subplot(111, projection=crs)
        ax.add_feature(USCOUNTIES, linewidth=0.5)

        img = ax.pcolormesh(xlocs, ylocs, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        fig.colorbar(img, ax=ax, label=title)

        ax.set_extent([cent_lon - 5, cent_lon + 5, cent_lat - 5, cent_lat + 5])
        ax.set_aspect('equal', 'datalim')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        ax.set_title(f"{radar_file} - {title}")

    # Save the figure 
    output_filename = f"{radar_file}_reflectivity.png"
    plt.savefig(output_filename, dpi=150)
    plt.show()
    plt.close(fig)


#%%
# Dropsonde plots 

# Create list of all the time values
time=[52937,53604,53833,62014,62307,62441,70937,71217,71312,74118,74329,74531,82058,82331,82507,91326,91615,91918,94428,94840,94924,95016,103222,103337,111122,111607]


# Loop to run through all the time values
# Creates individual timestep plots of the dropsonde data
for i in time: 
    hours=str(i).zfill(6)
    filename=f'D20230830_{hours}QC.nc'
    
    try: 
        # Read in dropsonde data
        dropsonde = netCDF4.Dataset(filename)
        # Save dropsonde data
        z = dropsonde.variables['gpsalt']
        z_profile_mask = z[0:]
        z_profile = []
        for i in range(len(z_profile_mask)):
           if z_profile_mask.mask[i] == False:
                z_profile = np.append(z_profile, z_profile_mask[i])
        
        launch_time = dropsonde.variables['time']
        launch_time_mask = launch_time[0:]
        
        i=0
        launch_time_list = []
        for i in range(len(launch_time_mask)):
            if z_profile_mask.mask.data[i] == False:
                launch_time_list = np.append(launch_time_list, launch_time_mask.data[i])
        
        # Create time variable
        flight_start_time = launch_time.units
        
        YYYY = flight_start_time[14:18]
        DD = flight_start_time[22:24]
        MM = flight_start_time[19:21]
        HH = flight_start_time[25:27]
        MIN = flight_start_time[28:30]
        SS = flight_start_time[31:33]
        
        flight_start_trakfile = MM + '/' + DD + '/' +YYYY
        flight_start_trakfile_time = HH + ':' + MIN + ':' + SS
        
        pressure = dropsonde.variables['pres']
        pressure_profile_mask = pressure[0:]
        pressure_profile = []
        for i in range(len(pressure_profile_mask)):
           if pressure_profile_mask.mask[i] == False:
                pressure_profile = np.append(pressure_profile, pressure_profile_mask[i])
                
        dp = dropsonde.variables['dp']
        dp_profile_mask = dp[0:]
        dp_profile = []
        for i in range(len(dp_profile_mask)):
           if dp_profile_mask.mask[i] == False:
                dp_profile = np.append(dp_profile, dp_profile_mask[i])
                
        t = dropsonde.variables['tdry']
        t_profile_mask = t[0:]
        t_profile = []
        for i in range(len(t_profile_mask)):
           if t_profile_mask.mask[i] == False:
                t_profile = np.append(t_profile, t_profile_mask[i])
                
        # Ensure data are the same same size
        p = pressure_profile[0:len(dp_profile)]
        T = t_profile[0:len(dp_profile)]
        Td = dp_profile[0:len(dp_profile)]
        
        # Plot figure
        fig = plt.figure(figsize=(8, 8))
        
        # Initiate the skew-T plot type from MetPy class loaded earlier
        skew = SkewT(fig, rotation=45)
        
        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        skew.plot(p, T, 'r')
        skew.plot(p, Td, 'g')
        #skew.plot_barbs(p[::25], u[::25], v[::25], y_clip_radius=0.03)
        
        # Set some appropriate axes limits for x and y
        skew.ax.set_xlim(0, 40)
        skew.ax.set_ylim(1020, 500)
        
        # Add the relevant special lines to plot throughout the figure
        skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                               alpha=0.25, color='orangered')
        skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                                 alpha=0.25, color='tab:green')
        
        plt.title(f"Flight: {flight_start_trakfile} - {hours}")
        plt.savefig(f'Final_Project_noj7_Sounding_{hours}.png')
        plt.show()
        plt.close()
        
    except Exception as e:
        print('Failed')
        
#%%
#GFS Data (Gridded Data)

# Open the file
file=pygrib.open('gfs_4_20230830_1200_000.grb2')

# Pull the messages needed
gph850mess=file[461]
gph=gph850mess.values/10
tempmess=file[462]
temp=tempmess.values-273
lats,lons=gph850mess.latlons()

# Create the figure parameters
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)
ax.set_extent([-125.,-70.,20.,60.])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightblue')
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray')
ax.add_feature(cf.LAKES,color='lightblue')
grd=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1,linestyle='--',alpha=0.5,color='white')

# Set up GPH range and then plot the contours
gphrange = np.arange(0, 1200, 2)
gphlines=plt.contour(lons,lats,gph,levels=gphrange,colors='black',linewidths=1.5,transform=ccrs.PlateCarree())

# Set up the Temperature range and then plot the contours
temprange=np.arange(-20,40,4)
templines=plt.contourf(lons,lats,temp,levels=temprange,cmap='coolwarm',transform=ccrs.PlateCarree())

ttl = plt.title('850hPa Geopotential Height and Temperature - 2023-08-30 12Z')
cbar=plt.colorbar(templines,orientation='horizontal')
cbar.set_label('Temperature (C)')

plt.tight_layout()
plt.savefig('GFS_Plot_20230830_12z.png', bbox_inches='tight')
plt.show()


#%%
#Surface Analysis Map 

# Load GFS GRIB2 file
grib_file = 'gfs_4_20230830_1200_000.grb2'
GFSdata = pygrib.open(grib_file)

# xtract Surface hPa geopotential height
gphsfcmess = GFSdata[558]
data = gphsfcmess.values / 10  #Coverts the height values
Lats, Lons = gphsfcmess.latlons()
Lons = np.where(Lons > 180, Lons - 360, Lons)  #Fixs issues with the longitudes wrapping around

# Set extent boundaries
lon_min, lon_max, lat_min, lat_max = -130, -60, 20, 60

# Mask data outside of the plot extent
mask = (Lons >= lon_min) & (Lons <= lon_max) & (Lats >= lat_min) & (Lats <= lat_max)
masked_data = np.where(mask, data, np.nan)  # Masks the overlapping contours

# Find local minima (Lows) 
minima = (data == minimum_filter(data, size=15))
l_y, l_x = np.where(minima)

# Set up the plot
fig = plt.figure(figsize=(11, 8), dpi=150)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=-95))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Plot the contours of only the masked data
cs = ax.contour(Lons, Lats, masked_data, levels=np.arange(0, 800, 2),
                colors='black', linewidths=1.25, transform=ccrs.PlateCarree())
ax.clabel(cs, inline=True, fontsize=8)

# Add in all of the plot features
ax.add_feature(cf.LAND, color='wheat')
ax.add_feature(cf.OCEAN, color='lightblue')
ax.add_feature(cf.COASTLINE, edgecolor='gray')
ax.add_feature(cf.STATES, edgecolor='gray')
ax.add_feature(cf.BORDERS, edgecolor='gray')
ax.add_feature(cf.LAKES, color='lightblue')

# Plot Lows only within the plot extent
for y, x in zip(l_y, l_x):
    lon, lat = Lons[y, x], Lats[y, x]
    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
        ax.text(lon, lat, 'L', fontsize=20, color='blue',
                fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree())
        ax.text(lon, lat - 1, f'{data[y, x]:.0f}', fontsize=12, color='blue',
                ha='center', va='top', transform=ccrs.PlateCarree())

ax.set_title('GFS Surface hPa Geopotential Height with Lows')
plt.savefig('GFS_SFC_Plot_20230830_12z.png', bbox_inches='tight')
plt.show()
