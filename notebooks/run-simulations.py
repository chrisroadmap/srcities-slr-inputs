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
# # Run temperature and ocean heat based on AR6 forcing and two layer model
#
# Based on https://github.com/chrisroadmap/ar6/blob/main/notebooks/215_chapter9_projections_AR6-historical.ipynb
#
# - Run two layer model with AR6 forcing, but using constrained-correlated parameter setups
# - Use constrained FaIR setup from 160
# - This also gets passed on to chapter 9

# %%
import json
import sys
import os
import random
from multiprocessing import Pool

import fair
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.stats as st
from climateforcing.twolayermodel import TwoLayerModel
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from h5_utils import *
from aerosol_forcing import aerocom_n, ghan

# %%
NINETY_TO_ONESIGMA = st.norm.ppf(0.95)
NINETY_TO_ONESIGMA

# %% [markdown]
# ## Ensemble generation
#
# We want to ensure reproducible results that don't change when this script is re-run. Grab list of pre-generated random seeds.

# %%
with open('../data_input/random_seeds.json', 'r') as filehandle:
    SEEDS = json.load(filehandle)

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
    global_total[species] = df.sum(axis=0).values[3:].astype(float) / 1000 
    #unit = df.units[0]
    #print(unit)
global_total['VOC'] = global_total.pop('NMVOC')
new_ceds = pd.DataFrame(global_total)
new_ceds.index = np.arange(1750,2020)
new_ceds.index = new_ceds.index.astype('int')
new_ceds.index.name='year'
new_ceds.columns.name=None
emissions_ceds_update = new_ceds.loc[1750:2020] + emissions - df_emissions
emissions_ceds_update.drop(index=range(2020,2101), inplace=True)
emissions_ceds_update

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
# ## Apply the constraint

# %%
accept_inds

# %%
geoff_sample_df.loc[accept_inds]

# %%
df_solar = pd.read_csv('../data_input/forcings/solar_erf.csv', index_col='year')
solar_forcing = df_solar.solar_erf.loc[1750:2019].values

df_volcanic = pd.read_csv('../data_input/forcings/volcanic_erf.csv', index_col='year')
volcanic_forcing = np.zeros((270))
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

# carbon cycle
pre_ind_co2 = pre_ind_co2[accept_inds]
r0 = r0[accept_inds]
rC = rC[accept_inds]
rT = rT[accept_inds]

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

# %%
ERFari = np.zeros((270, len(accept_inds)))
ERFaci = np.zeros((270, len(accept_inds)))

so2 = emissions_ceds_update.loc[:,'SO2']
bc = emissions_ceds_update.loc[:,'BC']
oc = emissions_ceds_update.loc[:,'OC']
nh3 = emissions_ceds_update.loc[:,'NH3']

for i in tqdm(range(len(accept_inds))):
    ERFari[:, i] = (
        (emissions_ceds_update.loc[:,'SO2']-emissions_ceds_update.loc[1750,'SO2']) * beta_so2[i] * 32/64 +
        (emissions_ceds_update.loc[:,'BC']-emissions_ceds_update.loc[1750,'BC']) * beta_bc[i] +
        (emissions_ceds_update.loc[:,'OC']-emissions_ceds_update.loc[1750,'OC']) * beta_oc[i] +
        (emissions_ceds_update.loc[:,'NH3']-emissions_ceds_update.loc[1750,'NH3']) * beta_nh3[i]
    )
    
    ERFaci[:,i] = ghan([so2 * 32/64, bc+oc], beta[i], aci_coeffs[i,0], aci_coeffs[i,1]) - ghan([so2[1750] * 32/64, bc[1750]+oc[1750]], beta[i], aci_coeffs[i,0], aci_coeffs[i,1])

# %%
pl.fill_between(np.arange(1750, 2020), np.percentile(ERFari, 5, axis=1), np.percentile(ERFari, 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750,2020), np.percentile(ERFari, 50, axis=1), color='k')
pl.grid()

# %%
pl.fill_between(np.arange(1750, 2020), np.percentile(ERFaci, 5, axis=1), np.percentile(ERFaci, 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750,2020), np.percentile(ERFaci, 50, axis=1), color='k')
pl.grid()

# %%
pl.fill_between(np.arange(1750, 2020), np.percentile(ERFari+ERFaci, 5, axis=1), np.percentile(ERFari+ERFaci, 95, axis=1), color='k', alpha=0.5)
pl.plot(np.arange(1750,2020), np.percentile(ERFari+ERFaci, 50, axis=1), color='k')
pl.grid()

# %% [markdown]
# ## Non-aerosol forcings and aggregated categories

# %%
df = pd.read_csv('../data_input/forcings/AR6_ERF_1750-2019.csv')
forcing_ensemble = {}
for cat in ['co2', 'ch4', 'n2o', 'other_wmghg', 'o3', 'h2o_stratospheric', 'contrails', 'aerosol-radiation_interactions',
           'aerosol-cloud_interactions', 'bc_on_snow', 'land_use', 'solar', 'volcanic', 'wmghgs', 'aerosol', 'albedo',
           'anthro', 'natural']:
    forcing_ensemble[cat] = np.zeros((8775, len(accept_inds)))
    
forcing_ensemble['co2'][8505:,:] = df['co2'].values[:,None] * scale_normals[:,0]
forcing_ensemble['ch4'][8505:,:] = df['ch4'].values[:,None] * scale_normals[:,1]
forcing_ensemble['n2o'][8505:,:] = df['n2o'].values[:,None] * scale_normals[:,2]
forcing_ensemble['other_wmghg'][8505:,:] = df['other_wmghg'].values[:,None] * scale_normals[:,3]
forcing_ensemble['o3'][8505:,:] = df['o3'].values[:,None] * scale_normals[:,4]
forcing_ensemble['h2o_stratospheric'][8505:,:] = df['h2o_stratospheric'].values[:,None] * scale_normals[:,5]
forcing_ensemble['contrails'][8505:,:] = df['contrails'].values[:,None] * scale_normals[:,6]
forcing_ensemble['aerosol-radiation_interactions'][8505:,:] = ERFari
forcing_ensemble['aerosol-cloud_interactions'][8505:,:] = ERFaci
forcing_ensemble['bc_on_snow'][8505:,:] = df['bc_on_snow'].values[:,None] * scale_normals[:,7]
forcing_ensemble['land_use'][8505:,:] = df['land_use'].values[:,None] * scale_normals[:,8]
forcing_ensemble['volcanic'][6255:8505,:] = df_volcanic.volcanic_erf.loc[:1749].values[:,None] * scale_normals[:,9]
forcing_ensemble['volcanic'][8505:,:] = df['volcanic'].values[:,None] * scale_normals[:,9]
forcing_ensemble['solar'][:8505,:] = df_solar.solar_erf.loc[:1749].values[:,None] * scale_normals[:,10]
forcing_ensemble['solar'][8505:,:] = np.linspace(0,trend_solar,270) + df['solar'].values[:,None] * scale_normals[:,10]
forcing_ensemble['wmghgs'] = (
    forcing_ensemble['co2'] +
    forcing_ensemble['ch4'] +
    forcing_ensemble['n2o'] +
    forcing_ensemble['other_wmghg']
)
forcing_ensemble['aerosol'] = (
    forcing_ensemble['aerosol-radiation_interactions'] +
    forcing_ensemble['aerosol-cloud_interactions']
)
forcing_ensemble['albedo'] = (
    forcing_ensemble['bc_on_snow'] +
    forcing_ensemble['land_use']
)
forcing_ensemble['natural'] = (
    forcing_ensemble['volcanic'] +
    forcing_ensemble['solar']
)
forcing_ensemble['anthro'] = (
    forcing_ensemble['co2'] +
    forcing_ensemble['ch4'] +
    forcing_ensemble['n2o'] +
    forcing_ensemble['other_wmghg'] +
    forcing_ensemble['o3'] +
    forcing_ensemble['h2o_stratospheric'] +
    forcing_ensemble['contrails'] +
    forcing_ensemble['aerosol-radiation_interactions'] +
    forcing_ensemble['aerosol-cloud_interactions'] +
    forcing_ensemble['bc_on_snow'] +
    forcing_ensemble['land_use']
)
forcing_ensemble['total'] = (
    forcing_ensemble['co2'] +
    forcing_ensemble['ch4'] +
    forcing_ensemble['n2o'] +
    forcing_ensemble['other_wmghg'] +
    forcing_ensemble['o3'] +
    forcing_ensemble['h2o_stratospheric'] +
    forcing_ensemble['contrails'] +
    forcing_ensemble['aerosol-radiation_interactions'] +
    forcing_ensemble['aerosol-cloud_interactions'] +
    forcing_ensemble['bc_on_snow'] +
    forcing_ensemble['land_use'] +
    forcing_ensemble['volcanic'] +
    forcing_ensemble['solar']
)

# %% [markdown]
# ## Run climate model

# %%
with open('../data_input/fair-samples/cmip6_twolayer_tuning_params.json', 'r') as filehandle:
    cmip6_models = json.load(filehandle)

# %%
cmix_mean = cmip6_models['cmix']['mean']['EBM-epsilon']
cdeep_mean = cmip6_models['cdeep']['mean']['EBM-epsilon']
eff_mean = cmip6_models['eff']['mean']['EBM-epsilon']

lamg_median = f2x_median/ecs_median
kappa_median = -(f2x_median/ecs_median - f2x_median/tcr_median)
gamma_2l_median = kappa_median/eff_mean

# %%
gamma_2l_median, kappa_median, lamg_median, eff_mean

# %%
results_ch9 = {}

# %%
os.makedirs('../data_output/', exist_ok=True)

# %%
arglist = []

lamg = -geoff_sample_df['lamg'].values
eff = geoff_sample_df['eff'].values
gamma_2l = geoff_sample_df['gamma_2l'].values
cdeep = geoff_sample_df['cdeep'].values
cmix = geoff_sample_df['cmix'].values


for i in range(len(accept_inds)):
    arglist.append(
        {
            'cmix': cmix[i],
            'cdeep': cdeep[i],
            'gamma_2l': gamma_2l[i],
            'lamg': lamg[i],
            'eff': eff[i],
            'in_forcing' : forcing_ensemble['total'][:,i],
            'firstyear': -6755
        }
    )

def run_tlm(args):
    in_forcing = args['in_forcing']
    driver = TwoLayerModel(
        extforce=in_forcing,
        exttime=np.arange(args['firstyear'],2020),
        tbeg=args['firstyear'],
        tend=2020,
        lamg=args['lamg'],
        t2x=None,
        eff=args['eff'],
        cmix=args['cmix'],
        cdeep=args['cdeep'],
        gamma_2l=args['gamma_2l'],
        outtime=np.arange(1750,2020),
        dt=0.2
    )
    output = driver.run()
    return(
        output.tg,
        output.tlev[:,1],
        output.hflux,
        output.ohc
    )
    
    
if __name__ == '__main__':
    with Pool(14) as pool:
        result = list(tqdm(pool.imap(run_tlm, arglist), total=len(accept_inds)))
    output = np.array(result)

# %%
output_t = output.transpose(1,2,0)

# %%
# this is separate to the headline results, for bob kopp
results_ch9['historical-AR6'] = {}
results_ch9['historical-AR6']['effective_radiative_forcing'] = forcing_ensemble['total'][8505:,...]
results_ch9['historical-AR6']['surface_temperature'] = output_t[0,...]
results_ch9['historical-AR6']['deep_ocean_temperature'] = output_t[1,...]
results_ch9['historical-AR6']['net_energy_imbalance'] = output_t[2,...]
results_ch9['historical-AR6']['ocean_heat_content'] = output_t[3,...] * 1e22
results_ch9['historical-AR6']['ECS'] = ecs
results_ch9['historical-AR6']['TCR'] = tcr
results_ch9['historical-AR6']['lambda0'] = lamg
results_ch9['historical-AR6']['cmix'] = cmix
results_ch9['historical-AR6']['cdeep'] = cdeep
results_ch9['historical-AR6']['efficacy'] = eff
results_ch9['historical-AR6']['eta'] = gamma_2l
results_ch9['year'] = np.arange(1750, 2020)

# %%
save_dict_to_hdf5(results_ch9, '../data_output_large/chapter9/twolayer_historical-AR6.h5')

# %%
