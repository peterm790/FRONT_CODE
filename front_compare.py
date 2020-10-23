import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from matplotlib.cm import get_cmap
import matplotlib
import cmocean

for mod in ['NOAA','ERA5']:
    print(mod)
    front = xr.open_zarr('../FRONT_OUT/'+str(mod)+'_2000_frontal.zarr')
    dic={}
    for seas in ['DJF','MAM','JJA','SON']:
        x = front.where(front.time.dt.season==seas).dropna(dim='time',how='all')
        x = x.front/x.front
        x = x.fillna(0)
        x = x.sum(dim='time')
        dic[seas]=x
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.rcParams['hatch.linewidth']=1
        plt.rcParams['hatch.color']='black'
        ax.coastlines('50m', linewidth=0.8)
        dic[seas].plot()
        ax.set_extent([-40, 30, -15, -75], ccrs.PlateCarree())
        plt.savefig('../FRONT_OUT/'+str(mod)+'_'+str(seas)+'.png')
        plt.close()
        print(seas,float(dic[seas].sel(latitude=-35).sel(longitude = 16)))
