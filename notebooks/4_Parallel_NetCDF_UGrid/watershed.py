#!/usr/bin/env python
# coding: utf-8

# # Parallelization, NetCDF, and UGrid on a watershed
# This example is based on the watershed from the [FloPy paper](https://doi.org/10.1111/gwat.13327) published in Groundwater. The following figure shows the lay of the land with a plan view of the watershed and two cross-sections. Although the figure shows Voronoi type (DISV) grid, this notebook will use a regularly structured (DIS) grid for the spatial discretization. 
# <div>
# <img src="./data/watershed.png" width="800"/>
# </div>

# ## Outline
# * Construct the grid, generate boundary data
# * Use KDTree technique to mark area near the rivers as 'valley', and away as 'mountain'
# * Build the groundwater flow (GWF) model with FloPy
# * Add two transport models (GWT) for the tracers: 'mountain' and 'valley'
# * Run the simulation and plot Heads, Flow pattern, Groundwater origin, Distribution over time
# * Enable NetCDF output and use the ModelSplitter to partition the model
# * Run parallel simulation
# * Use `xugrid` to open the NetCDF output, merge output data, and plot
# 

# ### Imports

# In[ ]:


import os
import matplotlib.pyplot as plt
import flopy
from flopy.discretization import StructuredGrid

# import some ready made items for your convenience
from defaults import *


# In[ ]:


model_dir = get_base_dir()


# ### Boundary data
# 
# Load the boundary data from `defaults.py`, containing rivers and the domain boundary

# In[ ]:


boundary_polygon = string2geom(geometry["boundary"])
bp = np.array(boundary_polygon)

stream_segs = (
    geometry["streamseg1"],
    geometry["streamseg2"],
    geometry["streamseg3"],
    geometry["streamseg4"],
)
sgs = [string2geom(sg) for sg in stream_segs]


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
ax.set_aspect("equal")

riv_colors = ("blue", "cyan", "green", "orange", "red")

ax.plot(bp[:, 0], bp[:, 1], "ko-")
for idx, sg in enumerate(sgs):
    sa = np.array(sg)
    _ = ax.plot(sa[:, 0], sa[:, 1], color=riv_colors[idx], lw=0.75, marker="o")


# ### Construct the regular (DIS) grid

# Set the cell dimensions. This will determine the number of cells in the grid. For example, setting dx = dy = 2500.0 will result in 9595 active cells

# In[ ]:


dx = 2500.0
dy = 2500.0
nrow = int(Ly / dy) + 1
ncol = int(Lx / dx) + 1


# Create a structured grid to work with (NB: this is not the simulation grid object)

# In[ ]:


working_grid = StructuredGrid(
    nlay=1,
    delr=np.full(ncol, dx),
    delc=np.full(nrow, dy),
    xoff=0.0,
    yoff=0.0,
    top=np.full((nrow, ncol), 1000.0),
    botm=np.full((1, nrow, ncol), -100.0),
)

set_structured_idomain(working_grid, boundary_polygon)
print("grid data: ", Lx, Ly, nrow, ncol)


# Load the topographic data from file

# In[ ]:


fine_topo = flopy.utils.Raster.load("./data/fine_topo.asc")
ax = fine_topo.plot()


# and resample the elevation onto the working grid

# In[ ]:


top_wg = fine_topo.resample_to_grid(
    working_grid,
    band=fine_topo.bands[0],
    method="linear",
    extrapolate_edges=True,
)


# ### Intersect river segments with grid
# Use a utility function to determine the grid cells that have a RIV segment, and generate an array to mark the river intersections.

# In[ ]:


ixs, cellids, lengths = intersect_segments(working_grid, sgs)

intersection_rg = np.zeros(working_grid.shape[1:])
for loc in cellids:
    intersection_rg[loc] = 1


# and plot the topology, the domain boundary, the RIV segments, and the grid nodes that have a RIV element in a single plot

# In[ ]:


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
pmv = flopy.plot.PlotMapView(modelgrid=working_grid)
ax.set_aspect("equal")
pmv.plot_array(top_wg)
pmv.plot_array(
    intersection_rg,
    masked_values=[
        0,
    ],
    alpha=0.2,
    cmap="Reds_r",
)
pmv.plot_inactive(color_noflow="white")
ax.plot(bp[:, 0], bp[:, 1], "k", linestyle="dashed")
for sg in sgs:
    sa = np.array(sg)
    ax.plot(sa[:, 0], sa[:, 1], "b-")


# ### Generate TOP and BOT coordinates from the topology

# In[ ]:


nlay = 5
dv0 = 5.0

topc = np.zeros((nlay, nrow, ncol), dtype=float)
botm = np.zeros((nlay, nrow, ncol), dtype=float)
dv = dv0
topc[0] = top_wg.copy()
botm[0] = topc[0] - dv
for idx in range(1, nlay):
    dv *= 1.5
    topc[idx] = botm[idx - 1]
    botm[idx] = topc[idx] - dv

for k in range(nlay):
    print(f"<z> for layer {k+1}: {(topc[k] - botm[k]).mean()}")


# ### Hydraulic conductivity
# Set uniform  hydraulic conductivity except for the two aquitards

# In[ ]:


hyd_cond = 10.0
hk = hyd_cond * np.ones((nlay, nrow, ncol), dtype=float)
hk[1, :, 25:] = hyd_cond * 0.001
hk[3, :, 10:] = hyd_cond * 0.00005


# ### Create the drain data for the river segments

# In[ ]:


leakance = hyd_cond / (0.5 * dv0)  # kv / b
drn_data = build_drain_data(
    working_grid,
    cellids,
    lengths,
    leakance,
    top_wg,
)
drn_data[:3]


# ### Create the groundwater discharge drain data

# In[ ]:


gw_discharge_data = build_groundwater_discharge_data(
    working_grid,
    leakance,
    top_wg,
)
gw_discharge_data[:3]


# ### Create idomain and starting head data
# Replicate the idomain from the working grid (= 1 layer) to the other layers. The starting heads are equal in each column and determined from the topology

# In[ ]:


idomain = np.array([working_grid.idomain[0, :, :].copy() for k in range(nlay)])
strt = np.array([top_wg.copy() for k in range(nlay)], dtype=float)


# ### Recharge data for the mountains and the valley
# Here we use a KDTree technique to set up two sources of recharge: one with clean mountain water and the other with potentially contaminated valley water.

# In[ ]:


from scipy.spatial import KDTree

# get grid x and y
grid_xx = working_grid.xcellcenters
grid_yy = working_grid.ycellcenters

# the river x and y from the indexes
riv_idxs = np.array(cellids)
riv_xx = grid_xx[riv_idxs[:,0],riv_idxs[:,1]]
riv_yy = grid_yy[riv_idxs[:,0],riv_idxs[:,1]]

# stack 2 arrays into single array of 2D coordinates
river_xy = np.column_stack((riv_xx, riv_yy))
grid_xy = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))

grid_xy[:3], river_xy[-3:]


# Now we create a KDTree from the river coordinates and then query with all grid coordinates for their closest distance to a river:

# In[ ]:


tree = KDTree(river_xy)
distance, index = tree.query(grid_xy)

index2d = index.reshape(nrow, ncol)
distance2d = distance.reshape(nrow, ncol)


# Plot the result

# In[ ]:


# cut on the distance to the closest RIV element to discriminate valley and mountain water
dist_from_riv = 10000.0


# In[ ]:


d2d_copy = distance2d.copy()
d2d_copy[d2d_copy < dist_from_riv] = 0.0
plt.imshow(d2d_copy)
plt.colorbar(shrink=0.6)


# Generate the cell indexes for the mountain recharge (away from rivers) and valley recharge (near rivers) 

# In[ ]:


# numpy.nonzero: Return the indices of the elements that are non-zero.
mountain_array = np.asarray(distance2d > dist_from_riv).nonzero()
mountain_idxs = np.array(list(zip(mountain_array[0], mountain_array[1])))

valley_array = np.asarray(distance2d <= dist_from_riv).nonzero()
valley_idxs = np.array(list(zip(valley_array[0], valley_array[1])))


# Both sources will have equal rates but are coupled to different tracers (GWT model)

# In[ ]:


max_recharge = 0.0001
rch_orig = max_recharge * np.ones((nrow, ncol))

# mountain recharge
rch_mnt = np.zeros((nrow, ncol))
for idx in mountain_idxs:
  rch_mnt[idx[0], idx[1]] = max_recharge

# valley recharge
rch_val = np.zeros((nrow, ncol))
for idx in valley_idxs:
  rch_val[idx[0], idx[1]] = max_recharge


# ## Build the FloPy simulation

# In[ ]:


sim = flopy.mf6.MFSimulation(
    sim_ws=model_dir,
    exe_name="mf6",
    memory_print_option="summary",
)


# ### Set up time discretization TDIS data

# In[ ]:


nper = 10
nsteps = 1
year = 365.25 # days
dt = 1000 * year
tdis = flopy.mf6.ModflowTdis(sim, 
                             nper=nper, 
                             perioddata=nper * [(nsteps*dt, nsteps, 1.0)])


# ### Setup the groundwater flow (GWF) model

# In[ ]:


gwfname = "gwf"

imsgwf = flopy.mf6.ModflowIms(
    sim,
    complexity="simple",
    print_option="SUMMARY",
    linear_acceleration="bicgstab",
    outer_maximum=1000,
    inner_maximum=100,
    outer_dvclose=1e-4,
    inner_dvclose=1e-5,
    filename=f"{gwfname}.ims",
)

gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=gwfname,
    print_input=False,
    save_flows=True,
    newtonoptions="NEWTON UNDER_RELAXATION",
)

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dx,
    delc=dy,
    idomain=idomain,
    top=top_wg,
    botm=botm,
    xorigin=0.0,
    yorigin=0.0,
)

ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=hk,
)
# storage
sto = flopy.mf6.ModflowGwfsto(
    gwf,
    save_flows=True,
    iconvert=1,
    ss=0.00001,
    sy=0.35,
    steady_state={0: True},
    transient={1 : True},
)

rch = flopy.mf6.ModflowGwfrcha(
    gwf,
    pname="rch_original",
    recharge={0 : rch_orig, 1 : 0.0},
    filename="gwf_original.rch",
)

rch = flopy.mf6.ModflowGwfrcha(
    gwf,
    pname="rch_mountain",
    recharge={1 : rch_mnt},
    auxiliary="CONCENTRATION",
    aux={1 : 1.0},
    filename="gwf_mountain.rch",
)

rch = flopy.mf6.ModflowGwfrcha(
    gwf,
    pname="rch_valley",
    recharge={1 : rch_val},
    auxiliary="CONCENTRATION",
    aux={1 : 1.0},
    filename="gwf_valley.rch",
)

drn = flopy.mf6.ModflowGwfdrn(
    gwf,
    stress_period_data=drn_data,
    pname="river",
    filename=f"{gwfname}_riv.drn",
)
drn_gwd = flopy.mf6.ModflowGwfdrn(
    gwf,
    auxiliary=["depth"],
    auxdepthname="depth",
    stress_period_data=gw_discharge_data,
    pname="gwd",
    filename=f"{gwfname}_gwd.drn",
)

oc = flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord=f"{gwf.name}.hds",
    budget_filerecord=f"{gwf.name}.cbc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("BUDGET", "ALL")],
)

sim.register_ims_package(imsgwf, [gwf.name])


# ### Setup two groundwater transport models

# In[ ]:


conc_start = 0.0

diffc = 0.0
alphal = 0.1

porosity = 0.35


# In[ ]:


def build_gwt_model(sim, gwtname, rch_package):

    gwt = flopy.mf6.ModflowGwt(
        sim,
        modelname=gwtname,
        print_input=False,
        save_flows=True,
    )

    dis = flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=dx,
        delc=dy,
        idomain=idomain,
        top=top_wg,
        botm=botm,
        xorigin=0.0,
        yorigin=0.0,
    )

    # initial conditions
    ic = flopy.mf6.ModflowGwtic(gwt, strt=conc_start, filename=f"{gwtname}.ic")

    # advection
    adv = flopy.mf6.ModflowGwtadv(gwt, scheme="tvd", filename=f"{gwtname}.adv")

    # dispersion
    dsp = flopy.mf6.ModflowGwtdsp(
        gwt,
        diffc=diffc,
        alh=alphal,
        alv=alphal,
        ath1=0.0,
        atv=0.0,
        filename=f"{gwtname}.dsp",
    )

    # mobile storage and transfer
    mst = flopy.mf6.ModflowGwtmst(
        gwt, 
        porosity=porosity,
        filename=f"{gwtname}.mst"
    )

    # sources and mixing
    sourcerecarray = [
        (rch_package, "AUX", "CONCENTRATION"),
    ]
    ssm = flopy.mf6.ModflowGwtssm(
        gwt, sources=sourcerecarray, filename=f"{gwtname}.ssm"
    )

    # output control
    oc = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbc",
            concentration_filerecord=f"{gwtname}.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
        )
    
    return gwt


# In[ ]:


imsgwt = flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        print_option="SUMMARY",
        linear_acceleration="bicgstab",
        outer_maximum=1000,
        inner_maximum=100,
        outer_dvclose=1e-4,
        inner_dvclose=1e-5,
        filename=f"gwt.ims",
    )

gwt_mnt = build_gwt_model(sim, "gwt_mnt", "rch_mountain")
sim.register_ims_package(imsgwt, [gwt_mnt.name])

gwt_val = build_gwt_model(sim, "gwt_val", "rch_valley")
sim.register_ims_package(imsgwt, [gwt_val.name])


# ### The GWF-GWT exchanges
# We need two of these here because both tracers, mountain and valley, need to be connected to the flow model.

# In[ ]:


gwfgwt = flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=gwfname,
    exgmnameb=gwt_mnt.name,
    filename="gwfgwt_mnt.exg",
)

gwfgwt = flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=gwfname,
    exgmnameb=gwt_val.name,
    filename="gwfgwt_val.exg",
)


# ### Count the number of active cells
# This should already give you an idea of the parallel performance of the model. Large models generally have better parallel performance.

# In[ ]:


ncells, nactive = get_model_cell_count(gwf)
print("nr. of cells:", ncells, ", active:", nactive)


# ### Write the model files

# In[ ]:


sim.write_simulation()


# ### Run the model
# 
# (NB: passing in the 'processors=1' argument here forces MODFLOW to use the PETSc parallel solver)

# In[ ]:


sim.run_simulation(processors=1)


# ### Plot Conductivities

# In[ ]:


fig = plt.figure(figsize=(10,8))

ax = plt.subplot(2,1,1)
pxs = flopy.plot.PlotCrossSection(model=gwf, line={"row": 20})
pa = pxs.plot_array(np.log10(hk))
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()

cb = plt.colorbar(pa)
cb.set_label("log(K)")
plt.title("conductivities (x-z)")

ax = plt.subplot(2,1,2)
pxs = flopy.plot.PlotCrossSection(model=gwf, line={"column": 25})
pa = pxs.plot_array(np.log10(hk))
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()

cb = plt.colorbar(pa)
cb.set_label("log(K)")
plt.title("conductivities (y-z)")


# ### Plot Results

# In[ ]:


times = gwf.output.head().get_times()
base_head = np.squeeze(gwf.output.head().get_data(totim=times[-1]))


# Create a top view of hydraulic head in the watershed. The red dashed lines show where the cross sections are taken to generate the results below.

# In[ ]:


fig = plt.figure(figsize=(8,4))
pmv = flopy.plot.PlotMapView(model=gwf, layer=0)
pa = pmv.plot_array(base_head)

# draw rivers
for sg in sgs:
    sa = np.array(sg)
    pmv.ax.plot(sa[:, 0], sa[:, 1], "b-")

# indicate cross sections (used further down)
xs_row = 20
xs_col = 42
xmin, xmax = pmv.ax.get_xlim()
ymin, ymax = pmv.ax.get_ylim()
plt.hlines(gwf.modelgrid.ycellcenters[xs_row][0],
           xmin, xmax, color="red", linestyles="dotted")
plt.vlines(gwf.modelgrid.xcellcenters[0][xs_col], 
           ymin, ymax, color="red", linestyles="dotted")

plt.colorbar(pa)

gwf.modelgrid.ycellcenters[xs_row][0], 


# To learn more about the global flow system, we plot the specific discharge. Note that the vectors are normalized to illustrate the principal flow direction. Generally, the vertical component is very small which shows when to set `normalize=False` in the `plot_vector` call.

# In[ ]:


plt.figure(figsize=(10,4))
pxs = flopy.plot.PlotCrossSection(model=gwf, line={"row": 20})
pa = pxs.plot_array(base_head)
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()

spdis = gwf.output.budget().get_data(text="DATA-SPDIS", totim=times[-1])[0]
nodes = nlay * nrow * ncol
qx = np.ones((nodes), dtype=float) * 1.0e30
qy = np.ones((nodes), dtype=float) * 1.0e30
qz = np.ones((nodes), dtype=float) * 1.0e30
n0 = spdis["node"] - 1
qx[n0] = spdis["qx"]
qy[n0] = spdis["qy"]
qz[n0] = spdis["qz"]
qx = qx.reshape(nlay, nrow, ncol)
qy = qy.reshape(nlay, nrow, ncol)
qz = qz.reshape(nlay, nrow, ncol)
qx = np.ma.masked_equal(qx, 1.0e30)
qy = np.ma.masked_equal(qy, 1.0e30)
qz = np.ma.masked_equal(qz, 1.0e30)
pxs.plot_vector(qx, qy, qz, normalize=True)

plt.title("Head and *normalized* specific discharge")
plt.colorbar(pa)


# ## Distribution of water origins after 10000 years

# In[ ]:


t = times[-1]
gwt_mnt = sim.get_model(model_name="gwt_mnt")
conc_mnt = np.squeeze(gwt_mnt.output.concentration().get_data(totim=t))
gwt_val = sim.get_model(model_name="gwt_val")
conc_val = np.squeeze(gwt_val.output.concentration().get_data(totim=t))
conc_orig = 1.0 - conc_mnt - conc_val
conc_orig[conc_orig == -1e+30] = 1.e+30

fig = plt.figure(figsize=(14,8))
fig.suptitle(f"Distribution after {int(t/365.25)} years")

plt.subplot(2,2,1)
pxs = flopy.plot.PlotCrossSection(model=gwt_mnt, line={"row": xs_row})
pa = pxs.plot_array(conc_mnt, vmin=0.0, vmax=1.0)
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()
plt.title(f"Mountain water (y = {grid_yy[xs_row,0]})")
plt.colorbar(pa, shrink=1.0)

plt.subplot(2,2,2)
pxs = flopy.plot.PlotCrossSection(model=gwt_mnt, line={"column": xs_col})
pa = pxs.plot_array(conc_mnt, vmin=0.0, vmax=1.0)
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()
plt.title(f"Mountain water (x = {grid_xx[0,xs_col]})")
plt.colorbar(pa, shrink=1.0)

plt.subplot(2,2,3)
pxs = flopy.plot.PlotCrossSection(model=gwt_val, line={"row": xs_row})
pa = pxs.plot_array(conc_val, vmin=0.0, vmax=1.0)
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()
plt.title("Valley water")
plt.colorbar(pa, shrink=1.0)

plt.subplot(2,2,4)
pxs = flopy.plot.PlotCrossSection(model=gwt_mnt, line={"column": xs_col})
pa = pxs.plot_array(conc_val, vmin=0.0, vmax=1.0)
pxs.plot_ibound(color_noflow="lightgrey")
pxs.plot_grid()
plt.title("Valley water")
plt.colorbar(pa, shrink=1.0)


# ## Distribution of mountain water over time

# In[ ]:


fig = plt.figure(figsize=(14,8))
fig.suptitle("Distribution of mountain water over time")

gwt_mnt = sim.get_model(model_name="gwt_mnt")

times = gwt_mnt.output.concentration().get_times()
plt_idxs = [0, 1, 4, 9]

for idx, plt_idx in enumerate(plt_idxs):
  t = times[plt_idxs[idx]]
  ax = plt.subplot(2, 2, idx + 1)
  pxs = flopy.plot.PlotCrossSection(model=gwt_mnt, line={"row": xs_row})
  conc = np.squeeze(gwt_mnt.output.concentration().get_data(totim=t))
  pa = pxs.plot_array(conc, vmin=0.0, vmax=1.0)

  ax.set_title(f"t = {int(t/365.25)} years")

  pxs.plot_ibound(color_noflow="lightgrey")
  pxs.plot_grid()

  plt.colorbar(pa, shrink=1.0)


# The following activates the writing of NetCDF output:

# In[ ]:


gwf = sim.get_model(gwfname)
gwf.export_netcdf = "EXPORT_NETCDF UGRID"

tdis = sim.get_package("tdis")
tdis.start_date_time = "1980-01-01"


# ### Model splitter to prepare for parallel run

# In[ ]:


split_dir = "simsplit"
splitter = flopy.mf6.utils.Mf6Splitter(sim)
split_mask = splitter.optimize_splitting_mask(nparts=2)
split_sim = splitter.split_multi_model(split_mask)
split_sim.set_sim_path(split_dir)
split_sim.write_simulation()

splitter.save_node_mapping(split_dir + "/" + "mfsplit_node_mapping.json")


# In[ ]:


split_sim.run_simulation(processors=2)


# ### Using NetCDF and `xugrid`

# In[ ]:


import xugrid as xu

fpth = os.path.join(split_dir, "gwf_0.nc")
nc_ds0 = xu.open_dataset(fpth)
fpth = os.path.join(split_dir, "gwf_1.nc")
nc_ds1 = xu.open_dataset(fpth)
nc_ds1


# Plot the partitions

# In[ ]:


args = {"vmin" : 0.0, "vmax" : 110.0}
var_name = "head_l1" # 

plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)
nc_ds1[var_name].isel(time=-1).ugrid.plot(**args)

plt.subplot(1, 2, 2)
nc_ds0[var_name].isel(time=-1).ugrid.plot(**args)


# ### Merge the partitions with `xugrid`

# In[ ]:


partitions = []
for ds in [nc_ds0, nc_ds1]:
    keep = ds["head_l1"].isel(time=-1).notnull()
    partitions.append(ds["head_l1"].isel(time=-1).where(keep, drop=True))

nc_merge = xu.merge_partitions(partitions)


# and plot

# In[ ]:


plt.figure(figsize=(8,4))
nc_merge["head_l1"].ugrid.plot(**args)


# In[ ]:




