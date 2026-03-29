# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run temperature and ocean heat based on updated scenarios

# %%
import json
import numpy as np
from fair.constants import molwt
from fair.forcing.ghg import etminan, meinshausen
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.stats as st
import random
import pandas as pd
from tqdm.notebook import tqdm
import random
import os
from climateforcing.twolayermodel import TwoLayerModel

# from ar6.utils import check_and_download, mkdir_p
# from ar6.utils.statistics import weighted_percentile
# from ar6.utils.h5 import *
# from ar6.utils import mkdir_p
# from ar6.constants.gases import ghg_to_rcmip_names
from aerosol_forcing import ghan, aerocom_n
# from ar6.forcing.ozone import eesc
# from ar6.constants.gases import rcmip_to_ghg_names, ghg_to_rcmip_names, ods_species, radeff
from multiprocessing import Pool

import matplotlib.pyplot as pl
import json

# %%
NINETY_TO_ONESIGMA = st.norm.ppf(0.95)
NINETY_TO_ONESIGMA

# %%
with open('../data_input/random_seeds.json', 'r') as filehandle:
    SEEDS = json.load(filehandle)

# %%
forcing = {}
scenarios = [
    'Low-to-Negative - SSP2 (Marker)',
    'Medium-to-Low - SSP2 (Marker)',
    'High - SSP3 (Marker)',
    'Medium - SSP2 (Marker)',
    'Very Low - SSP1 (Marker)',
    'High-to-Low - SSP5 (Marker)',
    'Low - SSP2 (Marker)'
]
for scenario in scenarios:
    forcing[scenario] = {}

# %%
emissions = pd.read_csv('../data_input/emissions/rcmip-emissions-annual-means-v5-1-0.csv')
df_emissions = pd.concat([emissions.loc[(
        (emissions.Variable=='Emissions|BC')|
        (emissions.Variable=='Emissions|OC')|
        (emissions.Variable=='Emissions|Sulfur')|
        (emissions.Variable=='Emissions|NOx')|
        (emissions.Variable=='Emissions|NH3')|
        (emissions.Variable=='Emissions|VOC')|
        (emissions.Variable=='Emissions|CO')
    ) & (emissions.Scenario=='ssp245') & (emissions.Region=='World'), 'Variable'], emissions.loc[(
        (emissions.Variable=='Emissions|BC')|
        (emissions.Variable=='Emissions|OC')|
        (emissions.Variable=='Emissions|Sulfur')|
        (emissions.Variable=='Emissions|NOx')|
        (emissions.Variable=='Emissions|NH3')|
        (emissions.Variable=='Emissions|VOC')|
        (emissions.Variable=='Emissions|CO')
    ) & (emissions.Scenario=='ssp245') & (emissions.Region=='World'), '1750':'2100']], axis=1)#.interpolate(axis=1).T
df_emissions.set_index('Variable', inplace=True)
df_emissions = df_emissions.interpolate(axis=1).T
df_emissions.rename(
    columns={
        'Emissions|BC': 'BC',
        'Emissions|OC': 'OC',
        'Emissions|Sulfur': 'SO2',
        'Emissions|NOx': 'NOx',
        'Emissions|NH3': 'NH3',
        'Emissions|VOC': 'VOC',
        'Emissions|CO': 'CO'
    }, inplace=True
)
# only keep cols we want
emissions = df_emissions[['SO2', 'BC', 'OC', 'NH3', 'NOx', 'VOC', 'CO']]
emissions.index = emissions.index.astype('int')
emissions.index.name='year'
emissions.columns.name=None

emissions_ceds_update = emissions.copy()

emissions_old = pd.read_csv('../data_input/emissions/rcmip-emissions-annual-means-v5-1-0.csv')
df_emissions = pd.concat([emissions_old.loc[(
        (emissions_old.Variable=='Emissions|BC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|OC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|Sulfur|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|NOx|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|NH3|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|VOC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|CO|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|BC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|OC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|Sulfur|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|NOx|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|NH3|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|VOC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|CO|MAGICC AFOLU|Agriculture')
    ) & (emissions_old.Scenario=='ssp245') & (emissions_old.Region=='World'), 'Variable'], emissions_old.loc[(
        (emissions_old.Variable=='Emissions|BC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|OC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|Sulfur|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|NOx|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|NH3|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|VOC|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|CO|MAGICC Fossil and Industrial')|
        (emissions_old.Variable=='Emissions|BC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|OC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|Sulfur|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|NOx|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|NH3|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|VOC|MAGICC AFOLU|Agriculture')|
        (emissions_old.Variable=='Emissions|CO|MAGICC AFOLU|Agriculture')
    ) & (emissions_old.Scenario=='ssp245') & (emissions_old.Region=='World'), '1750':'2100']], axis=1)#.interpolate(axis=1).T
df_emissions.set_index('Variable', inplace=True)
df_emissions = df_emissions.interpolate(axis=1).T
for species in ['BC', 'OC', 'Sulfur', 'NOx', 'NH3', 'VOC', 'CO']:
    df_emissions[species] = df_emissions['Emissions|{}|MAGICC Fossil and Industrial'.format(species)] + df_emissions['Emissions|{}|MAGICC AFOLU|Agriculture'.format(species)]
df_emissions.rename(columns = {'Sulfur': 'SO2'}, inplace=True)
df_emissions.drop(columns=[
        'Emissions|BC|MAGICC Fossil and Industrial',
        'Emissions|OC|MAGICC Fossil and Industrial',
        'Emissions|Sulfur|MAGICC Fossil and Industrial',
        'Emissions|NOx|MAGICC Fossil and Industrial',
        'Emissions|NH3|MAGICC Fossil and Industrial',
        'Emissions|VOC|MAGICC Fossil and Industrial',
        'Emissions|CO|MAGICC Fossil and Industrial',
        'Emissions|BC|MAGICC AFOLU|Agriculture',
        'Emissions|OC|MAGICC AFOLU|Agriculture',
        'Emissions|Sulfur|MAGICC AFOLU|Agriculture',
        'Emissions|NOx|MAGICC AFOLU|Agriculture',
        'Emissions|NH3|MAGICC AFOLU|Agriculture',
        'Emissions|VOC|MAGICC AFOLU|Agriculture',
        'Emissions|CO|MAGICC AFOLU|Agriculture',
    ],
    inplace=True
)
df_emissions.index = emissions.index.astype('int')
df_emissions.index.name='year'
df_emissions.columns.name=None

global_total = {}
for species in ['BC', 'OC', 'SO2', 'NH3', 'NOx', 'NMVOC', 'CO']:
    df = pd.read_csv('../data_input/emissions/CEDS/{}_global_CEDS_emissions_by_sector_2020_09_11.csv'.format(species))
    global_total[species] = df.sum(axis=0).values[3:].astype(float) / 1000 # yes could get openscm on this
    #unit = df.units[0]
    #print(unit)
global_total['VOC'] = global_total['NMVOC']
new_ceds = pd.DataFrame(global_total)
new_ceds.index = np.arange(1750,2020)
new_ceds.index = new_ceds.index.astype('int')
new_ceds.index.name='year'
new_ceds.columns.name=None
emissions_ceds_update = new_ceds.loc[1750:2020] + emissions - df_emissions
emissions_ceds_update.drop(index=range(2020,2101), inplace=True)
emissions_ceds_update
#new_ceds

emissions = pd.read_csv('../data_input/emissions/emissions_for_scm_v2.csv')
new_emissions = {}
for scenario in tqdm(scenarios):
    bc = np.zeros(351)
    oc = np.zeros(351)
    so2 = np.zeros(351)
    nh3 = np.zeros(351)
    nox = np.zeros(351)
    nmvoc = np.zeros(351)
    co = np.zeros(351)
    bc[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|BC'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    oc[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|OC'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    so2[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|Sulfur'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    nh3[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|NH3'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    nox[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|NOx'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    nmvoc[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|VOC'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    co[265:] = emissions.loc[(emissions['scenario']==scenario)&(emissions['region']=='World')&(emissions['variable']=='AR6 climate diagnostics|Infilled|Emissions|CO'),'2015':'2100'].interpolate(axis=1, pad=True).values.squeeze()
    bc[:265] = emissions_ceds_update.loc[1750:2014,'BC'].values
    oc[:265] = emissions_ceds_update.loc[1750:2014,'OC'].values
    so2[:265] = emissions_ceds_update.loc[1750:2014,'SO2'].values
    nh3[:265] = emissions_ceds_update.loc[1750:2014,'NH3'].values
    nox[:265] = emissions_ceds_update.loc[1750:2014,'NOx'].values
    nmvoc[:265] = emissions_ceds_update.loc[1750:2014,'VOC'].values
    co[:265] = emissions_ceds_update.loc[1750:2014,'CO'].values
    bc[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'BC'].values + np.linspace(0,0.8,5) * bc[265:270]
    oc[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'OC'].values + np.linspace(0,0.8,5) * oc[265:270]
    so2[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'SO2'].values + np.linspace(0,0.8,5) * so2[265:270]
    nh3[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'NH3'].values + np.linspace(0,0.8,5) * nh3[265:270]
    nox[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'NOx'].values + np.linspace(0,0.8,5) * nox[265:270]
    nmvoc[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'VOC'].values + np.linspace(0,0.8,5) * nmvoc[265:270]
    co[265:270] = np.linspace(1,0.2,5) * emissions_ceds_update.loc[2015:2019,'CO'].values + np.linspace(0,0.8,5) * co[265:270]
    
    new_emissions[scenario] = pd.DataFrame(
    {
        'BC': bc,
        'OC': oc,
        'SO2': so2,
        'NH3': nh3,
        'NOx': nox,
        'VOC': nmvoc,
        'CO': co
    })

# %% [markdown]
# ## Load unconstrained parameters

# %%
# ozone
ozone_feedback = np.load('../data_input/fair-samples/ozone_feedback_unconstrained.npy')
beta_ch4 = np.load('../data_input/fair-samples/beta_ch4_unconstrained.npy')
beta_n2o = np.load('../data_input/fair-samples/beta_n2o_unconstrained.npy')
beta_ods = np.load('../data_input/fair-samples/beta_ods_unconstrained.npy')
beta_co = np.load('../data_input/fair-samples/beta_co_unconstrained.npy')
beta_voc = np.load('../data_input/fair-samples/beta_voc_unconstrained.npy')
beta_nox = np.load('../data_input/fair-samples/beta_nox_unconstrained.npy')

# carbon cycle
r0 = np.load('../data_input/fair-samples/r0_unconstrained.npy')
rC = np.load('../data_input/fair-samples/rC_unconstrained.npy')
rT = np.load('../data_input/fair-samples/rT_unconstrained.npy')
pre_ind_co2 = np.load('../data_input/fair-samples/pre_ind_co2_unconstrained.npy')

# aerosol
beta_so2 = np.load('../data_input/fair-samples/beta_so2_unconstrained.npy')
beta_bc = np.load('../data_input/fair-samples/beta_bc_unconstrained.npy')
beta_oc = np.load('../data_input/fair-samples/beta_oc_unconstrained.npy')
beta_nh3 = np.load('../data_input/fair-samples/beta_nh3_unconstrained.npy')
beta = np.load('../data_input/fair-samples/beta_unconstrained.npy')
aci_coeffs = np.load('../data_input/fair-samples/aci_coeffs.npy')

# forcing
scale_normals = np.load('../data_input/fair-samples/scale_normals.npy')
trend_solar = np.load('../data_input/fair-samples/scale_trend_solar.npy')

# climate response
geoff_sample_df = pd.read_csv('../data_input/fair-samples/geoff_sample.csv', index_col=0)
f2x = np.load('../data_input/fair-samples/f2x_unconstrained.npy')
ecs = np.load('../data_input/fair-samples/ecs_unconstrained.npy')
tcr = np.load('../data_input/fair-samples/tcr_unconstrained.npy')

# accepted ensemble
accept_inds = np.loadtxt('../data_input/fair-samples/accept_inds.csv', dtype=int)

# %% [markdown]
# ## apply the constraint

# %%
df_solar = pd.read_csv('../data_input/forcings/solar_erf.csv', index_col='year')
solar_forcing = df_solar.solar_erf.loc[1750:2019].values

df_volcanic = pd.read_csv('../data_input/forcings/volcanic_erf.csv', index_col='year')
volcanic_forcing = np.zeros((351))
volcanic_forcing[:269] = df_volcanic.volcanic_erf.loc[1750:2018].values
# ramp down last 10 years to zero according to https://www.geosci-model-dev.net/9/3461/2016/gmd-9-3461-2016.html
volcanic_forcing[269] = volcanic_forcing[268]

# %%
# ozone
ozone_feedback = ozone_feedback[accept_inds]
beta_ch4 = beta_ch4[accept_inds]
beta_n2o = beta_n2o[accept_inds]
beta_ods = beta_ods[accept_inds]
beta_co = beta_co[accept_inds]
beta_voc = beta_voc[accept_inds]
beta_nox = beta_nox[accept_inds]

# aerosol
beta_so2 = beta_so2[accept_inds]
beta_bc = beta_bc[accept_inds]
beta_oc = beta_oc[accept_inds]
beta_nh3 = beta_nh3[accept_inds]
beta = beta[accept_inds]
aci_coeffs = aci_coeffs[accept_inds]

# forcing
scale_normals = scale_normals[accept_inds]
trend_solar = trend_solar[accept_inds]

# climate response
geoff_sample_df = geoff_sample_df.loc[accept_inds]
f2x = f2x[accept_inds]
ecs = ecs[accept_inds]
tcr = tcr[accept_inds]

# %%
f2x_median = np.median(f2x)
ecs_median = np.median(ecs)
tcr_median = np.median(tcr)

# %%
kappa = f2x/tcr - f2x/ecs
# kappa = efficacy * eta
pl.hist(kappa)

# %%
lamg = geoff_sample_df['lamg'].values
eff = geoff_sample_df['eff'].values
gamma_2l = geoff_sample_df['gamma_2l'].values
cdeep = geoff_sample_df['cdeep'].values
cmix = geoff_sample_df['cmix'].values

# %% [markdown]
# ## Aerosol emissions

# %%
for scenario in tqdm(scenarios):
    bc = new_emissions[scenario]['BC'].values.squeeze()
    oc = new_emissions[scenario]['OC'].values.squeeze()
    so2 = new_emissions[scenario]['SO2'].values.squeeze()
    nh3 = new_emissions[scenario]['NH3'].values.squeeze()
    
    forcing[scenario]['aerosol-radiation_interactions'] = np.zeros((351, len(accept_inds)))
    forcing[scenario]['aerosol-cloud_interactions'] = np.zeros((351, len(accept_inds)))
    
    for i in tqdm(range(len(accept_inds)), leave=False):
        forcing[scenario]['aerosol-radiation_interactions'][:, i] = (
            (so2 - so2[0]) * beta_so2[i] * 32/64 +
            (bc - bc[0]) * beta_bc[i] +
            (oc - oc[0]) * beta_oc[i] +
            (nh3 - nh3[0]) * beta_nh3[i]
        )

        forcing[scenario]['aerosol-cloud_interactions'][:,i] = ghan([so2 * 32/64, bc+oc], beta[i], aci_coeffs[i,0], aci_coeffs[i,1]) - ghan([so2[0] * 32/64, bc[0]+oc[0]], beta[i], aci_coeffs[i,0], aci_coeffs[i,1])

# %%
pl.fill_between(np.arange(1750, 2101), np.percentile(forcing[scenario]['aerosol-radiation_interactions'], 5, axis=1), np.percentile(forcing[scenario]['aerosol-radiation_interactions'], 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750,2101), np.percentile(forcing[scenario]['aerosol-radiation_interactions'], 50, axis=1), color='k')
pl.grid()

# %%
pl.fill_between(np.arange(1750, 2101), np.percentile(forcing[scenario]['aerosol-cloud_interactions'], 5, axis=1), np.percentile(forcing[scenario]['aerosol-cloud_interactions'], 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750,2101), np.percentile(forcing[scenario]['aerosol-cloud_interactions'], 50, axis=1), color='k')
pl.grid()

# %%
pl.fill_between(np.arange(1750, 2101), np.percentile(forcing[scenario]['aerosol-radiation_interactions']+forcing[scenario]['aerosol-cloud_interactions'], 5, axis=1), np.percentile(forcing[scenario]['aerosol-radiation_interactions']+forcing[scenario]['aerosol-cloud_interactions'], 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750, 2101), np.percentile(forcing[scenario]['aerosol-radiation_interactions']+forcing[scenario]['aerosol-cloud_interactions'], 50, axis=1), color='k')
pl.grid()

# %%
