"""Read data functions"""

import sys
import os
import intake
import xarray as xr

from config import KSCALEDATA, KSCALEOUTDIR
from KSCALE.read_data.get_coords import get_nn_lon_lat_index


### CST ###

cat = 'https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml'
current_location = "online"


### FUNC ###

def get_cat():
    out = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')[current_location]

    return out


def get_ds_name_global(resolution='n1280', driving='GAL9'):
    """values: (n1280, GAL9), (n2560, RAL3p3)"""
    out = 'um_glm_' + resolution + '_' + driving

    return out


def get_ds_name_regional(region='CTC', resolution='km4p4', physics='RAL3P3', driving='GAL9', reso_nest='n1280'):
    out = 'um_' + region + '_' + resolution + '_' + physics + '_' + reso_nest + '_' + driving + '_nest'

    return out


def get_dataset_global(resolution='n1280', driving='GAL9', zoom=7):
    cat = get_cat()
    ds_name = get_ds_name_global(resolution, driving)
    out = cat[ds_name](time='PT1H', zoom=zoom).to_dask()

    return out


def get_dataset_regional(region='CTC', resolution='km4p4', physics='RAL3P3', driving='GAL9', reso_nest='n1280', zoom=7):
    cat = get_cat()
    ds_name = get_ds_name_regional(region, resolution, physics, driving, reso_nest)
    out = cat[ds_name](time='PT1H', zoom=zoom).to_dask()

    return out


def get_2d_variable_global(resolution='n1280', driving='GAL9', zoom=7, variable='pr', lat_range=(-30., 30.), lon_range=(-180., 180.)):
    """Get variable space-time data"""
    idx = get_nn_lon_lat_index(zoom, lat_range, lon_range)
    ds = get_dataset_global(resolution, driving, zoom)
    vardata = ds[variable]
    out = vardata.isel(cell=idx)

    return out


def get_2d_variable_regional(region='CTC', resolution='km4p4', physics='RAL3P3', driving='GAL9', reso_nest='n1280', zoom=7, variable='pr', lat_range=(-30., 30.), lon_range=(-180., 180.)):
    """Get variable space-time data"""
    idx = get_nn_lon_lat_index(zoom, lat_range, lon_range)
    ds = get_dataset_regional(region, resolution, physics, driving, reso_nest, zoom)
    vardata = ds[variable]
    out = vardata.isel(cell=idx)

    return out


def load_daily_data_global(resolution='n1280', driving='GAL9', zoom=7, variable='hflsd', lat_range=(-30.,30), lon_range=(-180.,180.), year=2020):
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    datapath = KSCALEOUTDIR + '/' + variable + '/' + driving + '/' + resolution + '/z' + str(zoom) + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)
    datafile = datapath + '/' + str(year) + '_daily.nc'
    out = xr.open_dataarray(datafile)

    return out

