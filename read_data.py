"""Read data functions"""

import sys
import os
import xarray as xr

from config import KSCALEDATA, KSCALEOUTDIR


### FUNC ###

def get_path_season(season='summer'):
    if season == 'summer':
        out = KSCALEDATA + '/outdir_20160801T0000Z'
    elif season == 'winter':
        out = KSCALEDATA + '/outdir_20200120T0000Z'
    return out

def get_path_driving(season='summer', driving='RAL3'):
    path = get_path_season(season)
    out = path + '/DMn1280' + driving
    return out

def get_path_global(season='summer', driving='RAL3', resolution='n1280', physics='RAL3'):
    path = get_path_driving(season, driving)
    out = path + '/global_' + resolution + '_' + physics
    if driving == 'RAL3':
        out = out + 'p2'
    assert os.path.isdir(out)
    return out

def get_path_channel(season='summer', driving='RAL3', resolution='n2560', physics='RAL3'):
    path = get_path_driving(season, driving)
    out = path + '/channel_' + resolution + '_' + physics
    if physics == 'RAL3':
        out = out + 'p2'
    assert os.path.isdir(out)
    return out

def get_path_lam(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3'):
    path = get_path_driving(season, driving)
    out = path + '/lam_' + region + '_' + resolution + '_' + physics
    if physics == 'RAL3':
        out = out + 'p2'
    assert os.path.isdir(out)
    return out

def get_varfiles_global_single(season='summer', driving='RAL3', resolution='n1280', physics='RAL3', variable='shfx'):
    path = get_path_global(season, driving, resolution, physics)
    varpath = path + '/single_' + variable
    out = os.listdir(varpath)
    return out

def get_varfiles_global_precip(season='summer', driving='RAL3', resolution='n1280', physics='RAL3'):
    path = get_path_global(season, driving, resolution, physics)
    varpath = path + '/precip'
    out = os.listdir(varpath)
    return out

def get_varfiles_global_profile(season='summer', driving='RAL3', resolution='n1280', physics='RAL3', level=800):
    path = get_path_global(season, driving, resolution, physics)
    varpath = path + '/profile_' + str(level)
    out = os.listdir(varpath)
    return out

def get_varfiles_channel_single(season='summer', driving='RAL3', resolution='n2560', physics='RAL3', variable='shfx'):
    path = get_path_channel(season, driving, resolution, physics)
    varpath = path + '/single_' + variable
    out = os.listdir(varpath)
    return out

def get_varfiles_channel_precip(season='summer', driving='RAL3', resolution='n2560', physics='RAL3'):
    path = get_path_channel(season, driving, resolution, physics)
    varpath = path + '/precip'
    out = os.listdir(varpath)
    return out

def get_varfiles_channel_profile(season='summer', driving='RAL3', resolution='n2560', physics='RAL3', level=800):
    path = get_path_channel(season, driving, resolution, physics)
    varpath = path + '/profile_' + str(level)
    out = os.listdir(varpath)
    return out

def get_varfiles_lam_single(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', variable='shfx'):
    path = get_path_lam(season, driving, region, resolution, physics)
    varpath = path + '/single_' + variable
    out = os.listdir(varpath)
    return out

def get_varfiles_lam_precip(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3'):
    path = get_path_lam(season, driving, region, resolution, physics)
    varpath = path + '/precip'
    out = os.listdir(varpath)
    return out

def get_varfiles_lam_profile(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', level=800):
    path = get_path_lam(season, driving, region, resolution, physics)
    varpath = path + '/profile_' + str(level)
    out = os.listdir(varpath)
    return out


############################
#                          #
#      LOAD RAW DATA       #
#                          #
############################

#~~~ GLOBAL ~~~#

def load_ds_var_global_single(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1):
    path = get_path_global(season, driving, resolution, physics)
    files = get_varfiles_global_single(season, driving, resolution, physics, variable)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/single_' + variable + '/' + f
    out = xr.open_dataset(datafile)
    return out

def load_data_global_precip(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    path = get_path_global(season, driving, resolution, physics)
    files = get_varfiles_global_precip(season, driving, resolution, physics)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/precip/' + f
    ds = xr.open_dataset(datafile)
    out = ds.precipitation_rate.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out

def load_data_global_smc(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.), depth=0):
    ds = load_ds_var_global_single(season, driving, resolution, physics, 'smc', year, month, day)
    out = ds.moisture_content_of_soil_layer.isel(depth=depth).sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    out = out.where(abs(out) < 1000)   # remove inf
    return out

def load_data_global_single_var(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    ds = load_ds_var_global_single(season, driving, resolution, physics, variable, year, month, day)
    varname = list(ds.variables)[0]
    out = ds[varname].sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out


#~~~ CHANNEL ~~~#

def load_ds_var_channel_single(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1):
    path = get_path_channel(season, driving, resolution, physics)
    files = get_varfiles_channel_single(season, driving, resolution, physics, variable)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/single_' + variable + '/' + f
    out = xr.open_dataset(datafile)
    return out

def load_data_channel_precip(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    path = get_path_channel(season, driving, resolution, physics)
    files = get_varfiles_channel_precip(season, driving, resolution, physics)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/precip/' + f
    ds = xr.open_dataset(datafile)
    out = ds.precipitation_rate.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out

def load_data_channel_smc(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.), depth=0):
    ds = load_ds_var_channel_single(season, driving, resolution, physics, 'smc', year, month, day)
    out = ds.moisture_content_of_soil_layer.isel(depth=depth).sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    out = out.where(abs(out) < 1000)   # remove inf
    return out

def load_data_channel_single_var(season='summer', driving='RAL3', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    ds = load_ds_var_channel_single(season, driving, resolution, physics, variable, year, month, day)
    varname = list(ds.variables)[0]
    out = ds[varname].sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out


#~~~ LAM ~~~#

def load_ds_var_lam_single(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1):
    path = get_path_lam(season, driving, region, resolution, physics)
    files = get_varfiles_lam_single(season, driving, region, resolution, physics, variable)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/single_' + variable + '/' + f
    out = xr.open_dataset(datafile)
    return out

def load_data_lam_precip(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    path = get_path_lam(season, driving, region, resolution, physics)
    files = get_varfiles_lam_precip(season, driving, region, resolution, physics)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    f = [f for f in files if date in f][0]
    datafile = path + '/precip/' + f
    ds = xr.open_dataset(datafile)
    out = ds.precipitation_rate.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out

def load_data_lam_smc(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.), depth=0):
    ds = load_ds_var_lam_single(season, driving, region, resolution, physics, 'smc', year, month, day)
    out = ds.moisture_content_of_soil_layer.isel(depth=depth).sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    out = out.where(abs(out) < 1000)   # remove inf
    return out

def load_data_lam_single_var(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', variable='shfx', year=2016, month=8, day=1, lat_range=(-90., 90.), lon_range=(-180., 180.)):
    ds = load_ds_var_lam_single(season, driving, region, resolution, physics, variable, year, month, day)
    varname = list(ds.variables)[0]
    out = ds[varname].sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out



############################
#                          #
#      LOAD EF DATA        #
#                          #
############################

def get_var_path_season(season='summer'):
    if season == 'summer':
        out = KSCALEOUTDIR + '/outdir_20160801T0000Z'
    elif season == 'winter':
        out = KSCALEOUTDIR + '/outdir_20200120T0000Z'
    return out

def load_var_global(season='summer', driving='RAL3', resolution='n1280', physics='RAL3', variable='ef', lat_range=(-90., 90.), lon_range=(-180., 180.)):
    datapath = get_var_path_season(season)
    datapath = datapath + '/' + variable + '/DMn1280' + driving + '/global_' + resolution + '_' + physics
    datafile = datapath + '/' + variable + '_daily.nc'
    da = xr.open_dataarray(datafile)
    out = da.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out


def load_var_channel(season='summer', driving='RAL3', resolution='n2560', physics='RAL3', variable='ef', lat_range=(-90., 90.), lon_range=(-180., 180.)):
    datapath = get_var_path_season(season)
    datapath = datapath + '/' + variable + '/DMn1280' + driving + '/channel_' + resolution + '_' + physics
    datafile = datapath + '/' + variable  + '_daily.nc'
    da = xr.open_dataarray(datafile)
    out = da.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out


def load_var_lam(season='summer', driving='RAL3', region='africa', resolution='km2p2', physics='RAL3', variable='ef', lat_range=(-90., 90.), lon_range=(-180., 180.)):
    datapath = get_var_path_season(season)
    datapath = datapath + '/' + variable  + '/DMn1280' + driving + '/lam_' + resolution + '_' + physics
    datafile = datapath + '/' + variable + '_daily.nc'
    da = xr.open_dataarray(datafile)
    out = da.sel(latitude=slice(lat_range[0], lat_range[1]), longitude=slice(lon_range[0], lon_range[1]))
    return out
