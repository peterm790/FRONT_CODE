import numpy as np
import pandas as pd
import xarray as xr
import math
import skimage.feature
import skimage.segmentation
import scipy.ndimage as ndi

def pre_process(U,V):
    try:
        lon = U.longitude.values
        nlon = len(lon)
        lat = U.latitude.values
        nlat = len(lat)
    except:
        V = V.rename({'lon':'longitude'})
        V = V.rename({'lat':'latitude'})
        U = U.rename({'lon':'longitude'})
        U = U.rename({'lat':'latitude'})
        lon = U.longitude.values
        nlon = len(lon)
        lat = U.latitude.values
        nlat = len(lat)
    try:
        test = U.uas
    except:
        pass
    try:
        test = U.ua
        U = U.rename({'ua':'uas'})
        V = V.rename({'va':'vas'})
    except:
        U = U.rename({'uwnd':'uas'})
        V = V.rename({'vwnd':'vas'})
    return U,V

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
