#!/usr/bin/env python
# coding: utf-8

# # Aquifer Storage and Recovery in a saline aquifer
# ### Exercise 3

# In[ ]:


# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 3) # set default figure size
from ipywidgets import interact, FloatSlider, Layout
import flopy as fp  # import flopy and call it fp


# The parameters for the base case are given in the parameter block below.
# First, compute the recovery efficiency of the base case for the first cycle. 
# 
# Next, answer the following questions and demonstrate your answer by computing the recovery efficiency of the first cycle for the altered parameter while all other parameters are for the base case. 
# 1. Does the recovery efficiency increase or decrease when the dispersivity is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $\alpha_L=1$ m.
# 3. Does the recovery efficiency increase or decrease when the horizontal hydraulic conductivity is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $k=40$ m/d.
# 3. Does the recovery efficiency increase or decrease when the vertical hydraulic conductivity  is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $k_v=10$ m/d.
# 4. Does the recovery efficiency increase or decrease when the injected volume is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $Q=1000$ m$^3$/d.
# 5. Does the recovery efficiency increase or decrease when the salinity of the aquifer is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $c_s=30$ g/L. 

# ### Parameters

# In[ ]:


# domain size and boundary conditions
rw = 0.2 # radius of well, m
R = 200 # length of domain, m. large enough so that injected water does not reach boundary.
hR = 0 # head at right side of domain

# aquifer parameters
k = 20 # hydraulic conductivity, m/d
kv = k / 10 # vertical hydraulic conductivity, m/d
H = 10 # aquifer thickness, m
npor = 0.35 # porosity, -

# flow
Q = 500 # injection and extraction rate, m^3/d

# transport
alphaL = 0.1 # longitudinal dispersivity in horizontal direction
alphaT = alphaL / 10
diffusion_coef = 0

# concentration
cs = 20 # initial concentration, g/L (=kg/m^3)
cf = 0 # concentration injected water, g/L
climit = 1 # maximum concentration of extracted water, g/L

# buoyancy
rhoref = 1000 # reference density, kg/m^3
cref = 0 # reference concentration, kg/m^3
drhodc = 0.7143  # Slope of the density-concentration line

# space discretization
delr = 0.5 # length of cell along row (in x-direction), m
delc = 1 # width of cells normal to plane of flow (in y-direction), m
nlay = 20 # number of layers
nrow = 1 # number of rows
ncol = round((R - rw) / delr) # number of columns
z = np.linspace(0, -H, nlay + 1) # top and bottom(s) of layers
zc = 0.5 * (z[:-1] + z[1:]) # center of cells, used for contouring
r = rw + np.cumsum(delr * np.ones(ncol)) - 0.5 * delr # radial coordinates of center of cells, m
rb = np.hstack((rw, rw + np.cumsum(delr * np.ones(ncol)))) # radial coordinates of boundaries of cells, m

# radialize parameters:
theta = 2 * np.pi
krad = k * r * theta * np.ones((nlay, nrow, ncol))
kvertrad = kv * r * theta * np.ones((nlay, nrow, ncol))
nporrad = npor * r * theta * np.ones((nlay, nrow, ncol))

# time discretization
tin = 50 # injection time, d
delt = 0.5 # time step, d
nstepin = round(tin / delt) # computed number of steps during injection, integer
tout = 50 # extraction time, d
delt = 0.5 # time step, d
nstepout = round(tout / delt) # computed number of steps during extraction, integer

# model name and workspace
modelname = 'modelbuoy' # name of model
gwfname = modelname + 'f' # name of flow model
gwtname = modelname + 't' # name of transport model
modelws = './' + modelname # model workspace to be used


# In[ ]:


def asrmodel(simtime=tin, nstep=nstepin, Qwell=Q, c_init=cs):
    # simulation
    sim = fp.mf6.MFSimulation(sim_name=modelname, # name of simulation
                              version='mf6', # version of MODFLOW
                              exe_name='mf6', # absolute path to MODFLOW executable
                              sim_ws=modelws, # path to workspace where all files are stored
                             )
    
    # time discretization
    tdis = fp.mf6.ModflowTdis(simulation=sim, # add to the simulation called sim (defined above)
                              time_units="DAYS", 
                              nper=1, # number of stress periods 
                              perioddata=[[simtime, nstep, 1]], # period length, number of steps, timestep multiplier
                             )
    
    # groundwater flow model
    gwf = fp.mf6.ModflowGwf(simulation=sim, # add to simulation called sim
                            modelname=gwfname, # name of gwf model
                            save_flows=True, # make sure all flows are stored in binary output file
                           )
    
    # iterative model solver
    gwf_ims  = fp.mf6.ModflowIms(simulation=sim, # add to simulation called sim
                                 filename=gwf.name + '.ims', # file name to store ims
                                 linear_acceleration="BICGSTAB", # use BIConjuGantGradientSTABalized method
                                 inner_dvclose=1e-6, # get slightly more accurate solution than the default option
                                )                                                                                                
    # register solver
    sim.register_ims_package(solution_file=gwf_ims, # name of iterative model solver instance
                             model_list=[gwf.name], # list with name of groundwater flow model
                            )   
    
    # discretization
    gwf_dis = fp.mf6.ModflowGwfdis(model=gwf, # add to groundwater flow model called gwf
                                   nlay=nlay, 
                                   nrow=nrow, 
                                   ncol=ncol, 
                                   delr=delr, 
                                   delc=delc, 
                                   top=z[0], 
                                   botm=z[1:], 
                                  )
    
    # aquifer properties
    krad = k * r * theta * np.ones((nlay, nrow, ncol))
    kvertrad = kv * r * theta * np.ones((nlay, nrow, ncol))
    gwf_npf  = fp.mf6.ModflowGwfnpf(model=gwf, 
                                    k=krad, # horizontal k value
                                    k33=kvertrad, # vertical k value
                                    alternative_cell_averaging="LOGARITHMIC", # logarithmic averaging
                                    save_flows=True, # save the flow for all cells
                                   )
        
    # initial condition
    gwf_ic = fp.mf6.ModflowGwfic(model=gwf, 
                                 strt=hR, # initial head used for iterative solution
                                )
    
    # wells
    wells = []
    for ilay in range(nlay):
        wells.append([(ilay, 0, 0),  Qwell / nlay, cf])  # [(layer, row, col), U, concentration]
    wel_spd = {0: wells} # stress period data for periods 0 and 1
    gwf_wel = fp.mf6.ModflowGwfwel(model=gwf, 
                                   stress_period_data=wel_spd, 
                                   auxiliary=['CONCENTRATION'],
                                   pname='WEL1', # package name
                                  )
    
    # constant head 
    chd = []
    for ilay in range(nlay):
        chd.append([(ilay,  0,  ncol-1), hR, cs]) # [(layer, row, col), head, concentration]
    chd_spd  = {0: chd, 1: chd}    # Stress period data
    gwf_chd = fp.mf6.ModflowGwfchd(model=gwf, 
                                   stress_period_data=chd_spd, 
                                   auxiliary=['CONCENTRATION'],
                                   pname='CHD1', # package name
                                  )
    
    # buoyancy
    buy = fp.mf6.ModflowGwfbuy(model=gwf,
                               packagedata=[0, drhodc, cref, gwtname, 'CONCENTRATION'],
                               denseref=rhoref, # reference concentration
                               nrhospecies=1, # number of species
                               density_filerecord=f"{gwf.name}.dst", # file name
                               pname='BUY1', 
                              )
        
    # output control
    oc = fp.mf6.ModflowGwfoc(model=gwf, 
                             saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")], # what to save
                             budget_filerecord=f"{gwfname}.cbc", # file name where all budget output is stored
                             head_filerecord=f"{gwfname}.hds", # file name where all head output is stored
                            )
    
    # groundwater transport model
    gwt = fp.mf6.ModflowGwt(simulation=sim, 
                            modelname=gwtname, # name of groundwater transport model
                           )
    
    # iterative model solver
    gwt_ims  = fp.mf6.ModflowIms(simulation=sim,
                                 filename=gwt.name + '.ims', # must be different than file name of gwf model ims
                                 linear_acceleration="BICGSTAB",
                                 inner_dvclose=1e-6,
                                ) 
    sim.register_ims_package(solution_file=gwt_ims, 
                             model_list=[gwt.name],
                            )
    
    # discretization
    gwt_dis = fp.mf6.ModflowGwtdis(model=gwt, 
                                   nlay=nlay, 
                                   nrow=nrow, 
                                   ncol=ncol, 
                                   delr=delr, 
                                   delc=delc, 
                                   top=z[0], 
                                   botm=z[1:], 
                                  )
    
    # mobile storage and transfer
    gwt_sto = fp.mf6.ModflowGwtmst(model=gwt, 
                                   porosity=nporrad, # porosity
                                   save_flows=True,
                                  )
    
    # initial condition
    gwt_ic = fp.mf6.ModflowGwtic(model=gwt, 
                                 strt=c_init, # initial concentration
                                ) 
    
    # source sink mixing
    sourcelist = [("WEL1", "AUX", "CONCENTRATION"), ("CHD1", "AUX", "CONCENTRATION")]
    ssm = fp.mf6.ModflowGwtssm(model=gwt, 
                               sources=sourcelist, 
                               save_flows=True,
                               pname='SSM1', 
                              )
    
    # advection
    adv = fp.mf6.ModflowGwtadv(model=gwt,  
                               scheme="TVD",  # use Total Variation Diminishing (TVD)
                               pname='ADV1',
                              )
    
    # dispersion
    dsp = fp.mf6.ModflowGwtdsp(model=gwt, 
                               alh=alphaL,
                               ath1=alphaT, 
                               diffc=diffusion_coef,
                               pname='DSP1', 
                              )
    
    # output control
    oc = fp.mf6.ModflowGwtoc(model=gwt,
                             saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")], # what to save
                             budget_filerecord=f"{gwtname}.cbc", # file name where all budget output is stored
                             concentration_filerecord=f"{gwtname}.ucn", # file name where all concentration output is stored
                            )
    
    # interaction between gwf and gwt
    fp.mf6.ModflowGwfgwt(simulation=sim, 
                         exgtype="GWF6-GWT6", 
                         exgmnamea=gwf.name , 
                         exgmnameb=gwt.name , 
                         filename=f"{modelname}.gwfgwt",
                        );
    
    # write input files and solve model
    print('Writing and running model, please wait. ', end='')
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True) 
    if success == 1:
        print('Model solved successfully')
    else:
        print('Solve failed')
        
    # read concentration output
    cobj = gwt.output.concentration() # get handle to binary concentration file
    c = cobj.get_alldata().squeeze() # get the concentration data from the file
    times = np.array(cobj.get_times()) # get the times and convert to array

    return c, times


# #### Base case

# In[ ]:


c1, times1 = asrmodel(simtime=tin, nstep=nstepin, Qwell=Q, c_init=cs)
c2, times2 = asrmodel(simtime=tout, nstep=nstepout, Qwell=-Q, c_init=c1[-1])
for itime in range(nstepout):
    if np.mean(c2[itime, :, 0]) > climit:
        break 
Vextracted = times2[itime - 1] * Q # at itime c was above limit
Vinjected = tin * Q
RE = Vextracted / Vinjected
print(f'recovery efficiency base case: {RE:.2f}')

