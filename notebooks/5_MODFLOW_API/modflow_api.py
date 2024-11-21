#!/usr/bin/env python
# coding: utf-8

# # Using the API to Dynamically Adjust River Conductance
# 
# In this example, we are going to use the API to modify the river package so that the conductance is greater when the groundwater system is discharging to the river and smaller when the river is discharging to the aquifer. This example is inspired by [Zaadnoordijk (2009)](https://doi.org/10.1111/j.1745-6584.2009.00582.x), which used a combination of general head and drain boundaries to increase the conductance when the aquifer is discharging to the river.
# 
# <div>
# <img src="attachment:image.png" width="1000"/>
# </div>
# 
# ## Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as pl
import platform
import time

import flopy.mf6 as fp
import flopy.plot.styles as styles

import modflowapi
from modflowapi import Callbacks
from modflowapi.extensions import ApiSimulation


# ### Set path to the MODFLOW shared library

# In[ ]:


_platform = platform.system()
_DLL_PATH = (pl.Path(os.getenv('CONDA_PREFIX')) / "bin").resolve()
print(f"libmf6 path: {_DLL_PATH}")

if _platform == "Linux":
    _ext = ".so"
elif _platform == "Darwin":
    _ext = ".dylib"
else:
    _ext = ".dll"

libmf6 = _DLL_PATH / f"libmf6{_ext}"


# In[ ]:


print(f"libmf6 exists {libmf6.is_file()}")
print(f"Path to libmf6 '{libmf6}'")


# ## Create a sinusoidal groundwater level
# 
# This will be used to define the groundwater level in the MODFLOW model. We will define a constant head in the lower model layer.

# In[ ]:


h_mean, h_min, h_max = 320.0, 315.0, 325.0


# In[ ]:


amplitude = (h_min - h_max) / 2.0


# In[ ]:


ntimesteps = 401
ihalf = int(ntimesteps / 2) + 1
x = np.linspace(-np.pi, np.pi, ntimesteps)
chd_head = amplitude * np.sin(x) + h_mean


# Plot the sinusoidal groundwater level

# In[ ]:


plt.axhline(h_mean, color="black", ls=":")
ax = plt.gca()
ax.axvline(ihalf, color="black", ls=":")
ax.plot(chd_head, color="blue")
ax.set_xlabel("Simulation Time, days")
ax.set_ylabel("Groundwater Head, m")
ax.set_xlim(0,400);


# ## Create a MODFLOW model with a river stress package
# 
# We will create a simple MODFLOW model with 2 layers, 1 row, and 1 column. A constant head will be specified in the lower layer (layer 2) and the river boundary cell will be specified in model layer 1.

# In[ ]:


ws = pl.Path("temp")
name = "model"


# In[ ]:


sim = fp.MFSimulation(sim_name=name, sim_ws=ws, memory_print_option="all")
pd = [(1, 1, 1.0)] * chd_head.shape[0]
tdis = fp.ModflowTdis(sim, nper=len(pd), perioddata=pd)
ims = fp.ModflowIms(
    sim, complexity="simple", outer_dvclose=1e-6, inner_dvclose=1e-6
)
gwf = fp.ModflowGwf(
    sim,
    modelname=name,
    print_input=True,
    save_flows=True,
)
dis = fp.ModflowGwfdis(
    gwf,
    nlay=2,
    nrow=1,
    ncol=1,
    delr=1.0,
    delc=1.0,
    top=360,
    botm=[220, 200],
)
npf = fp.ModflowGwfnpf(
    gwf,
    k=50.0,
    k33=10.0,
)
ic = fp.ModflowGwfic(gwf, strt=chd_head[0])
condref = 1.0
spd = [((0, 0, 0), h_mean, condref, 319.0)]
riv = fp.ModflowGwfriv(
    gwf, stress_period_data=spd, pname="RIVER", print_flows=True
)
spd = {idx: [((1, 0, 0), h)] for idx, h in enumerate(chd_head)}
chd = fp.ModflowGwfchd(gwf, stress_period_data=spd, print_flows=True)
oc = fp.ModflowGwfoc(
    gwf,
    head_filerecord=f"{name}.hds",
    budget_filerecord=f"{name}.cbc",
    printrecord=[("budget", "all")],
    saverecord=[("head", "all"), ("budget", "all")],
)
sim.write_simulation()


# ## Use the modflowapi package to modify the river conductance at run time
# 
# We will be using the modflowapi python package to work with the MODFLOW API. A simple class that will plot simulated results during the simulation and defines the callback functions to initialize, update, and finalize the model simulation. XMI functionality will be used to update the river conductance during each Picard (outer) iteration based on the river/aquifer exchange direction.

# In[ ]:


class FluxMonitor:
    """
    An example class that sets the river conductance based
    on the gradient between the river cell and the head
    in the groundwater cell. A reduced river conductance 
    value is used when the flow is from the river cell to
    the aquifer. This class could be adapted to modify other 
    stress packages or monitor other simulated stress 
    packages fluxes by modifying this callback class.

    Parameters
    ----------
    vmin : float
        minimum head value for color scaling on the plot
    vmax : float
        maximum head value for color scaling on the plot
    ntimes : int
        number of time steps
    h_mean : float
        mean water level during the simulation
    """

    def __init__(self, vmin, vmax, ntimes, h_mean):
        self.vmin = vmin
        self.vmax = vmax
        self.ntimes = ntimes
        self.h_mean = h_mean
        
        self.time = np.linspace(0, self.ntimes, num=self.ntimes+1)
        self.flux = [0.0]
        
        self.sim_flux = None

    def _plot_idx(self, idx):
        self.ax.set_xlim(0, 400.0)
        self.ax.set_ylim(self.vmin, self.vmax)
        self.ax.set_xlabel("Simulation Time, days")
        self.ax.set_ylabel("Flow Rate, m$^3$/d")

        self.ax.axhline(0, lw=0.5, ls="-.", color="black")
        self.flux_line, = self.ax.plot(
            self.time[:idx],
            self.flux[:idx],
            color="black",
            lw=1.0,
            label="River flux",
        )
        self.ax.plot(idx, self.flux[-1], marker="o", ms=5, mfc="green", mec="green", clip_on=False) 
        styles.graph_legend(ax=self.ax)        
        
    def initialize_plot(self):
        """
        Method to initalize a matplotlib plot using flopy
        """
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            layout="constrained",
            figsize=(4.5, 5),
            )
        self.fig = fig
        self.ax = ax

        self._plot_idx(1)

    def update_plot(self, ml):
        """
        Method to update the matplotlib plot

        Parameters
        ----------
        ml : ApiModel
            modflow-api ApiModel object
        """
        
        self.ax.cla()

        idx = len(self.flux)
        self._plot_idx(idx)

        display.display(self.fig, clear=True)

    def callback(self, sim, callback_step):
        """
        A demonstration function that dynamically adjusts the CHD
        boundary conditions each stress period in a MODFLOW 6 model
        through the MODFLOW-API and then updates heads on a matplotlib
        plot for each timestep.

        Parameters
        ----------
        sim : modflowapi.Simulation
            A simulation object for the solution group that is
            currently being solved
        callback_step : enumeration
            modflowapi.Callbacks enumeration object that indicates
            the part of the solution modflow is currently in.
        """
        if callback_step == Callbacks.initialize:
            ml = sim.get_model()
            tag = ml.mf6.get_var_address("SIMVALS", "MODEL", "RIVER")
            self.sim_flux = ml.mf6.get_value_ptr(tag)
            
            self.initialize_plot()

        if callback_step == Callbacks.iteration_start:
            riv = sim.model.river
            spd = sim.model.river.stress_period_data.values
            if sim.model.X[0, 0, 0] > self.h_mean:
                cond = condref
            else:
                cond = condref * 0.10
            spd[0] = ((0, 0, 0), h_mean, cond, 319.0)
            sim.model.river.stress_period_data.values = spd        

        if callback_step == Callbacks.timestep_end:
            ml = sim.get_model()
            self.flux.append(float(self.sim_flux.sum()))
            self.update_plot(ml)

        if callback_step == Callbacks.finalize:
            plt.close()


# ## Run the API with the modflowapi python package
# 
# Create an instance of the FluxMonitor class and run the simulation using the API.

# In[ ]:


fmon = FluxMonitor(-0.6, 0.6, ntimesteps, h_mean)

modflowapi.run_simulation(libmf6, ws, fmon.callback, verbose=False)


# ## Extract the simulated river boundary flows
# 
# We will make a plot of the simulated results save to the cell-by-cell output file for comparison with the plot generated at runtime

# In[ ]:


cobj = gwf.output.budget()
riv_q_api = []
for totim in cobj.get_times():
    riv_q_api.append(cobj.get_data(totim=totim, text="riv")[0]["q"].sum())
riv_q_api = np.array(riv_q_api)


# ### Plot the simulated results

# In[ ]:


with styles.USGSPlot():
    axd = plt.figure(
        layout="constrained",
        figsize=(9, 5),
    ).subplot_mosaic(
        """
        ab
        """,
        empty_sentinel="X",
    )
    for key in axd.keys():
        axd[key].set_xlim(0, ntimesteps)
        axd[key].set_xlabel("Simulation Time, days")

    ax = axd["a"]
    ax.fill_between(
        cobj.get_times()[ihalf:],
        y1=h_mean,
        y2=chd_head[ihalf:],
        color="blue",
        ec="blue",
        label="Recharge",
    )
    ax.fill_between(
        cobj.get_times()[:ihalf],
        y1=h_mean,
        y2=chd_head[:ihalf],
        color="red",
        ec="red",
        label="Discharge",
    )
    ax.axhline(y=h_mean, color="black", lw=1.25, ls="--", label="River Stage")
    ax.axhline(y=319, color="black", lw=1.25, ls=":", label="River Bottom")
    ax.set_ylabel("Groundwater Head, m")
    styles.graph_legend(ax=ax)

    ax = axd["b"]
    ax.set_ylim(-0.6, 0.6)
    ax.axhline(y=0.0, color="black", lw=1.25, ls="-.", label=None)
    ax.plot(
        cobj.get_times(), riv_q_api, color="green", lw=0.75, label="RIVER API"
    )
    ax.set_ylabel("Flow Rate, m$^3$/d")
    styles.graph_legend(ax=ax)


# In[ ]:




