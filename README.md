# UPSCALE

Jupyter notebooks to analyze and plot soil moisture (SM) states and precipitation statistics for various simulation setups and regions (S. America, Africa, SEA)

config.py: define the input and output paths\
read_data.py: functions to access the simulation outputs\
p_config.py: boundaries of sub-regions\
plot_SM.ipynb: plot daily SM mean and variability\
plot_precip.ipynb: plot daily precipitation mean and variability\
compute_ef.ipynb: compute evaporative fraction (EF) from daily surface sensible and latent heat flux values\
compute_ef_sm_models.ipynb: model the SM-EF relationship based on daily values\
plot_ef_sm_models.ipynb: plots parameters of the fitted SM-EF models\
landatmospherecoupling.py: functions and class to fit SM-EF models and retrieve parameter values
