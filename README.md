# UPSCALE

Jupyter notebooks to analyze land surface states (soil moisture, turbulent fluxes) and precipitation for various simulation setups and regions (S. America, Africa, SEA)

config.py: define the input and output paths\
p_config.py: boundaries of sub-regions\
~/read_data: functions to access the simulation outputs\
make_data_<global/regional>.ipynb: extract data on specific domain and resample to monthly/daily\
compute_ef_<global/regional>.ipynb: compute evaporative fraction
plot_<ef/precip>.ipynb: seasonal mean/sum plots
