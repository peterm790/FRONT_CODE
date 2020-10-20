import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from matplotlib.cm import get_cmap
import matplotlib
import cmocean

#daily
U = xr.open_dataset('/media/peter/Storage/data/NOAA/uwnd.10m.2000.nc')
V = xr.open_dataset('/media/peter/Storage/data/NOAA/vwnd.10m.2000.nc')
V = V.rename({'lon':'longitude'})
V = V.rename({'lat':'latitude'})
U = U.rename({'lon':'longitude'})
U = U.rename({'lat':'latitude'})
V = V.sel(latitude=slice(-75,-15))
V = V.assign_coords(longitude=(((V.longitude + 180) % 360) - 180))
V = V.sortby(V.longitude)
U = U.sel(latitude=slice(-75,-15))
U = U.assign_coords(longitude=(((U.longitude + 180) % 360) - 180))
U = U.sortby(U.longitude)
U = U.sel(longitude = slice(-40,30))
U = U.sel(latitude = slice(-75,-15))
V = V.sel(longitude = slice(-40,30))
V = V.sel(latitude = slice(-75,-15))


ws = np.sqrt(np.square(U.uwnd) + np.square(V.vwnd))
Dir=np.mod(180+np.rad2deg(np.arctan2(U.uwnd, V.vwnd)),360)


front = xr.open_zarr('../NOAA_2000_frontal.zarr')

front = front.front/front.front
front = front.fillna(0)


front=front.compute()

for i in range(len(front.time)):
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.rcParams['hatch.linewidth']=1
    plt.rcParams['hatch.color']='black'
    ax.coastlines('50m', linewidth=0.8)
    Q = ax.streamplot(U.longitude, U.latitude, U.uwnd.values[i], V.vwnd.values[i],transform = ccrs.PlateCarree(),density = 3, zorder=0)
    W= ax.contourf(U.longitude, U.latitude, ws[i], cmap = cmocean.cm.speed,transform = ccrs.PlateCarree(),alpha=0.6,zorder=1)
    F =ax.contour(U.longitude,U.latitude,front[i],colors='black',levels = [0.99,1.01], transform = ccrs.PlateCarree(),alpha=1,zorder=2)
    F =ax.contourf(U.longitude,U.latitude,front[i],levels = [0.99,1.01], hatches=["x"], transform = ccrs.PlateCarree(),alpha=0,zorder=3)
    ax.set_extent([-40, 30, -15, -75], ccrs.PlateCarree())
    plt.title('ERA5 Front Detection - day\n'+str(U.time.values[i]))
    plt.savefig('demonstration/'+str(i)+'togif.png')
    plt.close()



#cat `ls -v *togif.png` | ffmpeg -framerate 1 -f image2pipe -i - output_daily.mp4
