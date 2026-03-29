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
# Based on https://github.com/chrisroadmap/cop26-pathways/blob/main/notebooks/3_run-emissions-scenarios.ipynb

# %%
import os
import json
from multiprocessing import Pool
import platform

import fair
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from h5_utils import save_dict_to_hdf5

# %%
# load in the configs and extend params to 2500
with open('../data_input/fair-1.6.2-wg3-params.json') as f:
    config_list = json.load(f)

    for data in config_list:
        for key in data:
            val = data[key]
            # if the value is a list of length 401
            if type(val) == list and len(val) == 361:
                for i in range(390):
                    # add the final value 400 - n times
                    val.append(val[-1])

# %%
emissions_in = {}
results_out = {}
WORKERS = 11  # set this based on your individual machine - allows parallelisation. nprocessors-1 is a sensible shout.

# %%
scenarios = ["H", "HL", "M", "ML", "L", "LN", "VL"]

# %%
for scenario in scenarios:
    emissions_in[scenario] = np.loadtxt('../data_input/emissions/fair-1.6.2-inputs/{}.csv'.format(scenario), delimiter=',')


# %%
def run_fair(args):
    _, thisF, thisT, _, thisOHU, _, thisAF = fair.forward.fair_scm(**args)
    return (thisT, thisOHU, np.sum(thisF, axis=1), np.sum(thisF[:,35:41], axis=1))

def fair_process(emissions):
    updated_config = []
    for i, cfg in enumerate(config_list):
        updated_config.append({})
        for key, value in cfg.items():
            if isinstance(value, list):
                updated_config[i][key] = np.asarray(value)
            else:
                updated_config[i][key] = value
        updated_config[i]['emissions'] = emissions
        updated_config[i]['diagnostics'] = 'AR6'
        updated_config[i]["efficacy"] = np.ones(45)
        updated_config[i]["gir_carbon_cycle"] = True
        updated_config[i]["temperature_function"] = "Geoffroy"
        updated_config[i]["aerosol_forcing"] = "aerocom+ghan2"
        updated_config[i]["fixPre1850RCP"] = False
    #    updated_config[i]["scale"][43] = 0.6
        updated_config[i]["F_solar"][270:] = 0
        
    # multiprocessing is not working for me on Windows
    if platform.system() == 'Windows':
        shape = (751, len(updated_config))
        ohu = np.ones(shape) * np.nan
        t = np.ones(shape) * np.nan
        f_aer = np.ones(shape) * np.nan
        f_tot = np.ones(shape) * np.nan
        for i, cfg in tqdm(enumerate(updated_config), total=len(updated_config), position=0, leave=True):
            t[:,i], ohu[:,i], f_tot[:,i], f_aer[:,i] = run_fair(updated_config[i])
    
    else:
        if __name__ == '__main__':
            with Pool(WORKERS) as pool:
                result = list(tqdm(pool.imap(run_fair, updated_config), total=len(updated_config), position=0, leave=True))

        result_t = np.array(result).transpose(1,2,0)
        t, ohc, f_tot, f_aer = result_t
    #temp_rebase = t - t[100:151,:].mean(axis=0)
    
    return t, ohu, f_tot, f_aer


# %%
for scenario in tqdm(scenarios, position=0, leave=True):
    results_out[scenario] = {}
    (
        results_out[scenario]['temperature'],
        results_out[scenario]['ocean_heat_content'],
        results_out[scenario]['effective_radiative_forcing'],
        results_out[scenario]['aerosol_effective_radiative_forcing'],
    ) = fair_process(emissions_in[scenario])

# %%
pl.plot(results_out[scenario]['temperature'])

# %%
pl.plot(results_out[scenario]['ocean_heat_content'])

# %%
pl.plot(results_out[scenario]['aerosol_effective_radiative_forcing'])

# %%
pl.plot(results_out[scenario]['effective_radiative_forcing'])

# %%
os.makedirs('../data_output', exist_ok=True)

# %%
save_dict_to_hdf5(results_out, '../data_output/fair_scenarios.h5')

# %%
