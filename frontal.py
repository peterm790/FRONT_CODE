from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import fsspec
import cartopy.crs as ccrs
import math
import skimage.feature
import skimage.segmentation
import scipy.ndimage as ndi
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm
import os
import datetime
import time

def front_watershed(U,V,lon,lat):
    print('front_watershed')
    front = U.copy()
    front['uas'] = front.uas*0
    front = front.rename({'uas':'x'})
    front['U2'] = U.uas #now
    front['V2'] = V.vas #now
    front['U1'] = xr.DataArray(np.concatenate([U.uas.values[:2]*np.nan,U.uas.values[:-2]]),dims=("time","latitude", "longitude"), coords={"time":front.time.values,"longitude":lon ,"latitude": lat})
    front['V1'] = xr.DataArray(np.concatenate([V.vas.values[:2]*np.nan,V.vas.values[:-2]]),dims=("time","latitude", "longitude"), coords={"time":front.time.values,"longitude":lon ,"latitude": lat})
    front = front.sel(time=front.time[2:])
    x = xr.where(front['U1'].values>0,front['U1'].values/front['U1'].values,np.nan)
    x = xr.where(front['V1'].values<0,x,np.nan)
    x = xr.where(front['U2'].values>0,x,np.nan)
    x = xr.where(front['V2'].values>0,x,np.nan)
    x = xr.where(front['V2'].values - front['V1'].values >  2.0 ,x,np.nan)
    front['x'] = xr.DataArray(x,dims=("time","latitude", "longitude"), coords={"time":front.time.values,"longitude":lon ,"latitude": lat}) #were above frontal criteria are met
    front['front'] = xr.DataArray(x*np.nan,dims=("time","latitude", "longitude"), coords={"time":front.time.values,"longitude":lon ,"latitude": lat}) #empty for output
    x = np.nan_to_num(x)
    for d in range(2,len(front.time)): #watershed segmentation to label each front
        local_maxi = skimage.feature.peak_local_max(x[d], indices=False, footprint=np.ones((3, 3)),labels=x[d])
        markers = ndi.label(local_maxi)[0]
        obj = skimage.segmentation.watershed(-x[d], markers, mask=x[d])
        front['front'][d] = xr.DataArray(obj, dims=("lat", "lon"), coords={"lon": lon , "lat":lat})
    front['front'] = front['front'].where(front['front']>0)
    return front



def getfront(frontin,dx='dx',dy='dy',lat='lat',lon='lon'):
    print('getfront')
    front = frontin
    front = front.sel(time=front.time[2:])
    for d in range(len(front.time)):
        winners=[]
        if math.isnan(np.nanmax(front.front[d]))==False:
            for z in range(1,int(np.nanmax(front.front[d]))+1):
                ids =  np.argwhere(front.front[d].values==z)
                latsid = [item[0] for item in ids]
                lonsid = [item[1] for item in ids]
                lats = [lat[i] for i in latsid]
                lons = [lon[i] for i in lonsid]
                mlat = np.mean(lats)
                imlat = np.argmin((lat-mlat)**2) #mean lat
                #area = dy*np.sum(dx[latsi[z]]) #area not sure why though not used again?
                xlen = dx[imlat]*4*np.std(lons)
                ylen = dy*4*np.std(lats)
                length = np.sqrt((ylen**2)+(xlen**2))  #length
                if length>500.0:   #must be greater than 500km
                    winners.append(z)
            front['front'][d] = front.front[d].where(np.isin(front.front[d],winners))
    return front


#U = xr.open_dataset('/media/peter/Storage/data/NOAA/uwnd.10m.2000.nc')
#V = xr.open_dataset('/media/peter/Storage/data/NOAA/vwnd.10m.2000.nc')


U = xr.open_dataset('/media/peter/Storage/data/ERA5/ERA5_ua_950_6hr_2000_2001.nc')
V = xr.open_dataset('/media/peter/Storage/data/ERA5/ERA5_va_950_6hr_2000_2001.nc')
#V = V.rename({'lon':'longitude'})
#V = V.rename({'lat':'latitude'})
#U = U.rename({'lon':'longitude'})
#U = U.rename({'lat':'latitude'})
#U = U.rename({'uwnd':'uas'})
#V = V.rename({'vwnd':'vas'})
U = U.rename({'ua':'uas'})
V = V.rename({'va':'vas'})


if list(V.longitude.values)==list(U.longitude.values): #check if the file is lekker
    print('longitudes are good')
    if list(V.latitude.values)==list(U.latitude.values):
        print('latitudes are good')
        #U = U.sel(time=slice('1950-01-01', '2100-02-01')) #1950 to last available
        #V = V.sel(time=slice('1950-01-01', '2100-02-01'))
        #V = V.assign_coords(longitude=(((V.longitude + 180) % 360) - 180))
        #V = V.sortby(V.longitude)
        #U = U.assign_coords(longitude=(((U.longitude + 180) % 360) - 180))
        #U = U.sortby(U.longitude)
        U = U.sortby(U.latitude)
        V = V.sortby(V.latitude)
        U = U.sel(longitude = slice(-40,30))
        U = U.sel(latitude = slice(-75,-15))
        V = V.sel(longitude = slice(-40,30))
        V = V.sel(latitude = slice(-75,-15))
        U = U.sel(level=U.level[0])
        V = V.sel(level=V.level[0])
        lon = U.longitude.values
        nlon = len(lon)
        lat = U.latitude.values
        nlat = len(lat)
        front = front_watershed(U,V,lon,lat)
        dx = np.cos(lat*math.pi/180.0)*2*math.pi*6370/360*(lon[-1]-lon[1])/(nlon-1)
        dy = ((lat[2]-lat[1])/180.0)*6370*math.pi
        frontout = getfront(front,dx,dy,lat,lon)
        frontout = frontout[['U1','V1','front']]
        frontout = frontout.rename({'U1':'uas'})
        frontout = frontout.rename({'V1':'vas'})
        frontout.to_zarr('ERA5_2000_frontal.zarr')
