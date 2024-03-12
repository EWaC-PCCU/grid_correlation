"""
@author: Min Lun Wu, EWaC
calculate grid correlation and draw its range
"""
import os
import time
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from itertools import product
import matplotlib.pyplot as plt

# %%
def corr_continuous(grid):
# check grid continuous by Depth-First-Search
# see Max_Area_of_Island on LeetCode for more info
    ROWS,COLS=len(grid),len(grid[0])
    visit=set()
    def dfs(r,c):
        if (r<0) | (r>=ROWS) | (c<0) | (c>=COLS) | (grid[r,c]==0) | ((r,c) in visit):
            return 0
        visit.add((r,c))
        return 1+dfs(r+1,c)+dfs(r-1,c)+dfs(r,c+1)+dfs(r,c-1)    
    return max(dfs(r,c) for r in range(ROWS) for c in range(COLS))

def base(loc,*,intv=0.125):
# draw a simple map with county lines by given a boundary [loc]
# make sure the environment has geopandas and cartopy module
    import geopandas as gpd
    import cartopy.crs as ccrs

    crs = ccrs.PlateCarree()
    fig,ax = plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':crs})
    county = gpd.read_file('./map_data/twcounty/twcounty.shp')
    county.boundary.plot(ax=ax,lw=0.7,color='k',zorder=999)
    
    ax.set_extent(loc)
    ax.gridlines(draw_labels=False,linewidth=1, color='gray', alpha=0.3, linestyle=':',
                 xlocs=np.arange(110,130+intv,intv),
                 ylocs=np.arange(10,30+intv,intv))

    ax.set_xticks(np.arange(110,130+intv*2,intv*2))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yticks(np.arange(10,30+intv*2,intv*2))
    ax.set_yticklabels(np.round(np.arange(10,30+intv*2,intv*2),2),fontsize=8,rotation=90,va='center')

    ax.set_xlim(loc[0],loc[1])
    ax.set_ylim(loc[2],loc[3])
    return ax

# %%read rain data & prepare
res=1 #unit: km
data = Dataset(f'./rain_1960_2021_daily_0.0{res}deg_taipei.nc','r')
var = data.variables['rain'][:].data # notice that the var name might be different
valid_mask=var[0,:,:]!=-99.9
var[var==-99.9] = np.nan

lat = data.variables['lat'][:].data
lon = data.variables['lon'][:].data
xx,yy = np.meshgrid(lon,lat)

del data

# %% corr cal
var_arr=var.reshape((var.shape[0], -1))
corr_arr=np.corrcoef(var_arr,rowvar=False)

# %% corr threshold
threshhold=0.8
check=corr_arr>=threshhold
corr_thresh=(corr_arr>=threshhold).reshape((len(lat),len(lon),len(lat),len(lon)))

# %% corr continuous calc & sum its area
area=np.full((len(lat),len(lon)),np.nan)
for (r,c) in product(range(len(lat)),range(len(lon))):
    area[r,c]=corr_continuous(corr_thresh[r,c,:,:])

# %% CWB station list
data = pd.read_csv('./map_data/stn_list.csv',encoding='big5')
st,lats,lons = [],[],[]
filt=(data['county'].str.contains('北'))
stn=data[filt].reset_index(drop=True)
CWB_stn=stn['stn_ID'].str[0]=='4'
auto_stn=stn['stn_ID'].str[0]=='C'

del data

# %% draw
area[valid_mask==0]=np.nan
valid_size=np.sum(valid_mask)
loc=[min(lon),max(lon),min(lat),max(lat)]

# corr area
ax=base(loc)
corr_range = ax.pcolor(xx,yy,area/valid_size,cmap='OrRd',vmin=0,vmax=0.4)
cbar = plt.colorbar(corr_range,location='right',pad=0.01,shrink=0.5)

# CWB station
ax.scatter(stn['lon'][CWB_stn],stn['lat'][CWB_stn],s=23,marker='*',c='green',zorder=222,label='CWB stn')
ax.scatter(stn['lon'][auto_stn],stn['lat'][auto_stn],s=30,marker='.',c='yellow',zorder=100,label='auto stn')
leg=plt.legend(loc='upper right',ncol=1,frameon=False,fontsize=10)

plt.title(f'corr > {threshhold:,.1f} range',fontsize=14)
plt.savefig(f'corr_rain_{res}km',dpi=500,bbox_inches='tight') 

