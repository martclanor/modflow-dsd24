#!/usr/bin/env python
# coding: utf-8

# # Building and post-processing a MODFLOW 6 model with FloPy

# ![ex01a.png](attachment:ex01a.png)

# A MODFLOW 6 model will be developed of the domain shown above. This model simulation is based on example 1 in [Pollock, D.W., 2016, User guide for MODPATH Version 7—A particle-tracking model for MODFLOW: U.S. Geological Survey Open-File Report 2016–1086, 35 p., http://dx.doi.org/10.3133/ofr20161086](https://doi.org/10.3133/ofr20161086).
# 
# The model domain will be discretized into 3 layers, 21 rows, and 20 columns. A constant value of 500 ft will be specified for `delr` and `delc`. The top (`TOP`) of the model should be set to 400 ft and the bottom of the three layers (`BOTM`) should be set to 220 ft, 200 ft, and 0 ft, respectively. The model has one steady-state stress period. 
# 
# MODFLOW does not require that input data be provided in specific units (for example, SI units) instead it only requires that consistent units be used. As a result all input data should be specified with a length unit of feet and a time unit of days.

# In[ ]:


import pathlib as pl
import platform

import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Before creating any of the MODFLOW 6 FloPy objects you should define the simulation workspace (`ws`) where the model files are and the simulation name (`name`). The `ws` should be set to `"./temp"` and `name` should be set to `"intro"`.

# In[ ]:


ws = pl.Path("./temp")
name = "intro"


# ## Build the model
# 
# Create a simulation object, a temporal discretization object, and a iterative model solution object using `flopy.mf6.MFSimulation()`, `flopy.mf6.ModflowTdis()`, and `flopy.mf6.ModflowIms()`, respectively. Set the `sim_name` to `name` and `sim_ws` to `ws` in the simulation object. Use default values for all temporal discretization and iterative model solution variables. Make sure to include the simulation object (`sim`) as the first variable in the temporal discretization and iterative model solution objects. 
# 
# **NOTE:** Variables with default values (for example, `time_units=None` in `flopy.mf6.ModflowTdis()`).

# In[ ]:


# create simulation (sim = flopy.mf6.MFSimulation())


# create tdis package (tdis = flopy.mf6.ModflowTdis(sim))


# create iterative model solution (ims = flopy.mf6.ModflowIms(sim))



# Create the groundwater flow model object (`gwf`) using `flopy.mf6.ModflowGwf()`. Make sure to include the simulation object (`sim`) as the first variable in the groundwater flow model object and set `modelname` to `name`. Use `Shift-Tab` to see the optional variables that can be specified.

# In[ ]:





# Create the discretization package using `flopy.mf6.ModflowGwfdis()`. Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `DIS` package (`flopy.mf6.ModflowGwfdis()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-dis.html).
# 
# FloPy can accomodate all of the options for specifying array data for a model. `CONSTANT` values for a variable can be specified by using a `float` or `int` python variable (as is done below for `DELR`, `DELC`, and `TOP`). `LAYERED` data can be specified by using a list or `CONSTANT` values for each layer (as is done below for `BOTM` data) or a list of numpy arrays or lists. Three-Dimensional data can be specified using a three-dimensional numpy array (with a shape of `(nlay, nrow, ncol)`) for this example. More information on how to specify array data can be found in the [FloPy ReadTheDocs](https://flopy.readthedocs.io/en/latest/Notebooks/mf6_data_tutorial07.html#MODFLOW-6:-Working-with-MODFLOW-Grid-Array-Data). 

# In[ ]:


# dis data


# In[ ]:





# `pmv = flopy.plot.PlotMapView(model=gwf)` and `pxs = flopy.plot.PlotCrossSection(model=gwf, line={"row": 20})` can be used to confirm that the discretization is correctly defined.

# In[ ]:





# In[ ]:





# ### Create the initial conditions (IC) package
# 
# Create the initial conditions package (`IC`) using `flopy.mf6.ModflowGwfic()` and set the initial head (`strt`) to 320. Default values can be used for the rest of the initial conditions package input. Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `IC` package (`flopy.mf6.ModflowGwfic()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-ic.html).

# In[ ]:





# ### Create the node property flow (NPF) package
# 
# The hydraulic properties for the model are defined in the image above and are specified in the node property flow package (`NPF`) using `flopy.mf6.ModflowGwfnpf()`. The first layer should be convertible (unconfined) and the remaining two layers will be non-convertible so `icelltype` should be `[1, 0, 0]`. The horizontal (`k`) and vertical (`k33`) conductivity should also be assigned as python lists (`[,,]`) and the values shown in the image above. The variable `save_specific_discharge` should be set to `True` so that specific discharge data are saved to the cell-by-cell file and can be used to plot discharge. Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `NPF` package (`flopy.mf6.ModflowGwfic()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-npf.html).

# In[ ]:





# In[ ]:





# ### Create the recharge package
# 
# The recharge rate is defined in the image above. Use the `flopy.mf6.ModflowGwfrcha()` method to specify recharge data using arrays. Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `RCH` package (`flopy.mf6.ModflowGwfrcha()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-rcha.html). 
# 
# **NOTE:** For stress packages, if you rerun a cell another version of the package will be created since MODFLOW 6 allows multiple versions of stress packages. To avoid this you can specify `pname=uniquename` which will replace the stress package if you rerun the cell. 

# In[ ]:





# ### Create the well package
# 
# The well is located in layer 3, row 11, column 10. The pumping rate is defined in the image above. Use the `flopy.mf6.ModflowGwfwel()` method to specify well data for the well package (`WEL`). Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `WEL` package (`flopy.mf6.ModflowGwfwel()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-wel.html).
# 
# `stress_period_data` for list-based stress packages (for example, `WEL`, `DRN`, `RIV`, and `GHB`) is specified as a dictionary with the zero-based stress-period number as the key and a list of tuples, with the tuple containing the data required for each stress entry. For example, each tuple for the `WEL` package includes a zero-based cellid and the well rate `(cellid, q)`. For this example, the zero-based cellid for `WEL` package can be a tuple with the `(layer, row, column)` for the well or three integers separated by a comma `layer, row, column`. More information on how to specify `stress_period_data` for list based stress packages can be found in the [FloPy ReadTheDocs](https://flopy.readthedocs.io/en/latest/Notebooks/mf6_data_tutorial06.html#Adding-Stress-Period-List-Data). 
# 
# An example of a `stress_period_data` tuple for the `WEL` package is
# 
# ```python
# # (layer, row, column, q)
# (0, 0, 0, -1e5)
# ```
# 
# **NOTE:** Remember the `pname="well"` tip to prevent creating multiple versions of the well package if you rerun the cell.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# ### Create the river package
# 
# The river is located in layer 1 and column 20 in every row in the model. The river stage stage and bottom are at 320 and 318, respectively; the river conductance is 1e5. Use the `flopy.mf6.ModflowGwfriv()` method to specify well data for the river package (`RIV`). Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `RIV` package (`flopy.mf6.ModflowGwfriv()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-riv.html).
# 
# An example of a `stress_period_data` tuple for the `RIV` package is
# 
# ```python
# # (layer, row, column, stage, cond, rbot)
# (0, 0, 0, 320., 1e5, 318.)
# ```
# 
# **HINT**: list comprehension is an easy way to create a river cell in every row in column 20 of the model.
# 
# **NOTE:** Remember the `pname="river"` tip to prevent creating multiple versions of the river package if you rerun the cell.
# 

# In[ ]:





# In[ ]:





# ### Build output control
# 
# Define the output control package (`OC`) for the model using the `flopy.mf6.ModflowGwfoc()` method to `[('HEAD', 'ALL'), ('BUDGET', 'ALL')]` to save the head and flow for the model. Also the head (`head_filerecord`) and cell-by-cell flow (`budget_filerecord`) files should be set to `f"{name}.hds"` and `f"{name}.cbc"`, respectively. Use `Shift-Tab` to see the optional variables that can be specified. A description of the data required by the `OC` package (`flopy.mf6.ModflowGwfoc()`) can be found in the MODFLOW 6 [ReadTheDocs document](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-oc.html).

# In[ ]:


oc = flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord=f"{name}.hds",
    budget_filerecord=f"{name}.cbc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)


# Because we haven't set `SAVE_FLOWS` to `True` in all of the packages we can set `.name_file.save_flows` to `True` for the groundwater flow model (`gwf`) to save flows for all packages that can save flows. 

# In[ ]:


gwf.name_file.save_flows = True


# ### Write the model files and run the model
# 
# Write the MODFLOW 6 model files using `sim.write_simulation()`. Use `Shift-Tab` to see the optional variables that can be specified for `.write_simulation()`.

# In[ ]:


sim.write_simulation()


# Run the model using `sim.run_simulation()`, which will run the MODFLOW 6 executable installed in the Miniforge class environment (`pyclass`) and the MODFLOW 6 model files created with `.write_simulation()`. Use `Shift-Tab` to see the optional variables that can be specified for `.run_simulation()`.

# In[ ]:


sim.run_simulation()


# ## Post-process the results
# 
# Load the heads and face flows from the hds and cbc files. The head file can be loaded with the `gwf.output.head().get_data()` method. The cell-by-cell file can be loaded with the `gwf.output.budget().get_data()` method. 
# 
# Name the heads data `hds`.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# The entries in the cell-by-cell file can be determined with the `.headers` property on the cell budget file object.

# In[ ]:





# The unique records in the cell budget file can be determined using `.headers[["text", "imeth"]].drop_duplicates()`.

# In[ ]:





# Retrieve the `'DATA-SPDIS'` data type from the cell-by-cell file. Name the specific discharge data `spd`.
# 
# Cell-by-cell data is returned as a list so access the data by using `spd = gwf.output.budget().get_data(text="DATA-SPDIS")[0]`.

# In[ ]:





# ### Plot the results
# 
# Plot the results using `flopy.plot.PlotMapView()`. The head results can be plotted using the `.plot_array()` method. The discharge results can be plotted using the `plot_specific_discharge()` method. Boundary conditions can be plotted using the `.plot_bc()` method.

# In[ ]:


mm = flopy.plot.PlotMapView(model=gwf, layer=2, extent=gwf.modelgrid.extent)
cbv = mm.plot_array(hds)
q = mm.plot_vector(spd["qx"], spd["qy"])
mm.plot_bc("RIV", color="blue")
mm.plot_bc("WEL", plotAll=True)
mm.plot_grid(lw=0.5, color="black")

# create data outside of plot limits for legend data
mm.ax.plot(
    -100,
    -100,
    marker="s",
    lw=0,
    ms=4,
    mfc="red",
    mec="black",
    mew=0.5,
    label="Well",
)
mm.ax.plot(
    -100,
    -100,
    marker="s",
    lw=0,
    ms=4,
    mfc="blue",
    mec="black",
    mew=0.5,
    label="River cell",
)

# plot legend
plt.legend()

# plot colorbar
cb = plt.colorbar(cbv, ax=mm.ax, shrink=0.5)
cb.set_label(label="Head, ft", weight="bold")
plt.show(block=False)


# In[ ]:





# In[ ]:





# In[ ]:




