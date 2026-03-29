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
import json
from multiprocessing import Pool
import platform

import fair
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
with open('../data_input/fair-1.6.2-wg3-params.json') as f:
    config_list = json.load(f)

# %%
emissions_in = {}
results_out = {}
WORKERS = 11  # set this based on your individual machine - allows parallelisation. nprocessors-1 is a sensible shout.

# %%
scenarios = ["ssp245_constant-2020-ch4", "ch4_40"]

# %%
for scenario in scenarios:
    emissions_in[scenario] = np.loadtxt('../data_input/emissions/{}.csv'.format(scenario), delimiter=',')


# %%
def run_fair(args):
    thisC, thisF, thisT, _, thisOHU, _, thisAF = fair.forward.fair_scm(**args)
    return (thisC[:,0], thisC[:,1], thisT, thisF[:,1], np.sum(thisF, axis=1))

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
        shape = (361, len(updated_config))
        c_co2 = np.ones(shape) * np.nan
        c_ch4 = np.ones(shape) * np.nan
        t = np.ones(shape) * np.nan
        f_ch4 = np.ones(shape) * np.nan
        f_tot = np.ones(shape) * np.nan
        for i, cfg in tqdm(enumerate(updated_config), total=len(updated_config), position=0, leave=True):
            c_co2[:,i], c_ch4[:,i], t[:,i], f_ch4[:,i], f_tot[:,i] = run_fair(updated_config[i])
    
    else:
        if __name__ == '__main__':
            with Pool(WORKERS) as pool:
                result = list(tqdm(pool.imap(run_fair, updated_config), total=len(updated_config), position=0, leave=True))

        result_t = np.array(result).transpose(1,2,0)
        c_co2, c_ch4, t, f_ch4, f_tot = result_t
    temp_rebase = t - t[100:151,:].mean(axis=0)
    
    return c_co2, c_ch4, temp_rebase, f_ch4, f_tot


# %%
for scenario in tqdm(scenarios, position=0, leave=True):
    results_out[scenario] = {}
    (
        results_out[scenario]['co2_concentrations'],
        results_out[scenario]['ch4_concentrations'],
        results_out[scenario]['temperatures'],
        results_out[scenario]['ch4_effective_radiative_forcing'],
        results_out[scenario]['effective_radiative_forcing']
    ) = fair_process(emissions_in[scenario])

# %%
emissions_in[scenario]

# %%
config_list[0]

# %%
import numpy as np

# %%
np.__version__

# %%
