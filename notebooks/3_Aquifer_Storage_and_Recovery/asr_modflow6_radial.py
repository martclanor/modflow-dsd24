#!/usr/bin/env python
# coding: utf-8

# # Aquifer Storage and Recovery in a saline aquifer - radial flow

# In[ ]:


# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 3) # set default figure size
from ipywidgets import interact, FloatSlider, Layout
import flopy as fp  # import flopy and call it fp


# ## Description of the flow problem
# Consider radial flow in a confined, initially saline aquifer. The aquifer extends in the $r$ direction from $r=0$ to $r=R$, where $R$ is chosen far enough away not to effect the solution. Water is injected by the well at a rate $Q$. The head is fixed at $r=R$ to $h_R$. Flow is considered to be at steady state instantaneously. 
# 
# The initial salt concentration is equal to $c_s$ everywhere. Injection of fresh water with concentration $c_f$ starts at $t=0$ and last for $t_\text{in}$ days, after which water is extracted at the same rate $Q$ for $t_\text{out}$ days. Injection and extraction are distributed equally across the layers, which is an approximation when buoyancy is taken into account.
# 
# The density $\rho$ of the water is approximated using the following linear relation:
# \begin{equation}
# \rho = \frac{\text{d}\rho}{\text{d}c}(c-c_\text{ref}) + \rho_\text{ref}
# \end{equation}
# where the reference concentration is $c_\text{ref}=0$ and the reference density is $\rho_\text{ref}=1000$ kg/m$^3$ (i.e., freshwater) and $\frac{\text{d}\rho}{\text{d}c}$ is the specified gradient such that a salinity of $c=35$ kg/m$^3$ (seawater) results in a density of 1025 kg/m$^3$. Here, the aquifer that is considered is brackish and has a salinity $c_s=20$ g/L, which is lower than seawater.
# 
# Radial flow is simulated with a MODFLOW 6 model with one row (and multiple layers and columns, of course) using the approach of Langevin (2008). This means that the hydraulic conductivity $k$ and porosity $n$ are replaced by radialized versions $k_r$ and $n_r$ as
# \begin{equation}
# k_r = 2\pi r k
# \end{equation}
# \begin{equation}
# n_r = 2\pi r n
# \end{equation}
# where $r$ is the radial coordinate of the center of the cell. Furthermore, the inter-cell hydraulic conductivities must be computed using the logarithmic averaging option.

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
alphaL = 0.5 # longitudinal dispersivity in horizontal direction
alphaT = alphaL / 10
diffusion_coef = 0

# concentration
cs = 20 # initial concentration, g/L (=kg/m^3)
cf = 0 # concentration injected water, g/L

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


# ## Function to create and run the model
# The function to run the model is defined below. The model is for one stress period and takes as input the total time, number of steps, discharge of the well (positive for injection) and initial salinity concentration. The function returns the computed concentration and corresponding times. Compared to the 1D case of the previous notebook (linear flow), the following changes are made to the groundwater flow model (no changes are made to the simulation and the groundwater transport model):
# * Use the converted (radialized) values of $k$ in the npf package and use the logarithmic averaging option to compute cell-by-cell hydraulic conductivities.
# * Add the (converted) vertical hydraulic conductivity
# * Use the discharge $Q$ in the well package.
# * Take buoyancy into account using the buoyancy package

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


# ## Run model and make contour plot of the concentration
# The model is run twice. First for injection starting with an initial concentration of `cs`. Second for extraction starting with an initial concentration equal to the computed concentration at the end of the injection period. All concentrations are gathered in one array `c`. A contour plot with a time slider is created in the second code cell below.

# In[ ]:


c = cs * np.ones((1, nlay, ncol))
c1, times1 = asrmodel(simtime=tin, nstep=nstepin, Qwell=Q, c_init=cs)
c2, times2 = asrmodel(simtime=tout, nstep=nstepout, Qwell=-Q, c_init=c1[-1])
c = np.append(c, c1, axis=0)
c = np.append(c, c2, axis=0)


# In[ ]:


def contour(t=0):
    tstep = round(t / delt)
    plt.subplot(111, xlim=(0, 80), ylim=(-10, 0), xlabel='x (m)', ylabel='z (m)', aspect=1)
    cset = plt.contour(r, zc, c[tstep], [1, 4, 8, 12, 16, 19], cmap='coolwarm')
    plt.colorbar(cset, shrink=0.35, aspect=4)

interact(contour, t=FloatSlider(value=0, min=0, max=(tin + tout), 
                                step=delt, layout=Layout(width='60%')));


# ### Recovery efficiency
# Recovery is terminated when the average concentration in the well is above the limit (recall that the discharge is distributed equally over the well screen, which is an approximation for variable density flow). The total salinity of drinking water should be less than 1 g/L (maximum salinity values are much lower in many countries, but 1 g/L can still be considered drinkable). Here, the recovery efficiency is computed using the 1 g/L limit. A plot of the salinity in the first cell is shown below. The vertical orange line is the time that extraction starts.

# In[ ]:


times = np.hstack((0, times1, tin + times2))
plt.plot(times, np.mean(c[:, :, 0], axis=1))
plt.xlabel('time (d)')
plt.ylabel('average conc. at $x=0$ (g/L)')
plt.axvline(50, color='C1')
plt.text(45, 0.25, 'start of extraction', rotation=90, va='center', color='C1')
plt.grid()


# The recovery efficiency is

# In[ ]:


climit = 1
for itime in range(nstepout):
    if np.mean(c2[itime, :, 0]) > climit:
        break
Vextracted = times2[itime - 1] * Q # at itime c was above limit
Vinjected = tin * Q
RE = Vextracted / Vinjected
print(f'recovery efficiency of first cycle: {RE:.2f}')


# ### Add a rest period
# When the maximum concentration is reached, extraction is stopped for the remainder of the extraction time. During this time, no water is injected or extracted, but the transition zone will very slowly rotate. This rest period is simulated by running the model for a third time using a well discharge of zero and an initial concentration equal to the concentration in the aquifer when extraction was stopped. 

# In[ ]:


c = cs * np.ones((1, nlay, ncol))
c1, times1 = asrmodel(simtime=tin, nstep=nstepin, Qwell=Q)
c2, times2 = asrmodel(simtime=tout, nstep=nstepout, Qwell=-Q, c_init=c1[-1])
for itime in range(nstepout):
    if np.mean(c2[itime, :, 0]) > climit:
        break
c3, times3 = asrmodel(simtime=tout - itime * delt, nstep=nstepout - itime, 
                      Qwell=0, c_init=c2[itime - 1])
c = np.append(c, c1, axis=0)
c = np.append(c, c2[:itime], axis=0)
c = np.append(c, c3, axis=0)


# In[ ]:


def contour(t=0):
    tstep = round(t / delt)
    plt.subplot(111, xlim=(0, 80), ylim=(-10, 0), xlabel='x (m)', ylabel='z (m)', aspect=1)
    cset = plt.contour(r, zc, c[tstep], [1, 4, 8, 12, 16, 19], cmap='coolwarm')
    plt.colorbar(cset, shrink=0.35, aspect=4)

interact(contour, t=FloatSlider(value=0, min=0, max=(tin + tout), 
                                step=delt, layout=Layout(width='60%')));


# ### Multiple cycles
# The performance of ASR systems commonly increases during successive cycles. 
# The process described above is repeated for 5 cycles below. The concentration is gathered in one large array `c` for all 5 cycles; `c[0]` contains the initial condition. The recovery efficiency of each cycle is stored in the list `RElist`. This takes a little while to compile as the model needs to be run three times for each cycle. Note that the recovery efficiency increases slowly from 54% in the first cycle to 83% in the fifth cycle. 

# In[ ]:


c = cs * np.ones((1, nlay, ncol))
c_init = cs
ncycle = 5
RElist = []
for icycle in range(ncycle):
    c1, times1 = asrmodel(simtime=tin, nstep=nstepin, Qwell=Q, c_init=c_init)
    c2, times2 = asrmodel(simtime=tout, nstep=nstepout, Qwell=-Q, c_init=c1[-1])
    for itime in range(nstepout):
        if np.mean(c2[itime, :, 0]) > climit:
            break
    Vextracted = times2[itime - 1] * Q # at itime c was above limit
    Vinjected = tin * Q
    RE = Vextracted / Vinjected
    RElist.append(RE)
    print(f'RE for cycle {icycle + 1} is {RE:.2}')
    c3, times3 = asrmodel(simtime=tout - itime * delt, nstep=nstepout - itime, 
                          Qwell=0, c_init=c2[itime - 1])
    c_init = c3[-1]
    c = np.append(c, c1, axis=0)
    c = np.append(c, c2[:itime], axis=0)
    c = np.append(c, c3, axis=0)


# In[ ]:


plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, 6), RElist, '.', ls='-')
plt.xlabel('cycle')
plt.ylabel('recovery efficiency')
plt.xticks(np.arange(1, 6))
plt.grid()


# In[ ]:


def contour(t=0):
    tstep = round(t / delt)
    plt.subplot(111, xlim=(0, 100), ylim=(-10, 0), xlabel='x (m)', ylabel='z (m)', aspect=1)
    cset = plt.contour(r, zc, c[tstep], [1, 4, 8, 12, 16, 19], cmap='coolwarm')
    plt.colorbar(cset, shrink=0.35, aspect=4)
    
interact(contour, t=FloatSlider(value=0, min=0, max=ncycle * (tin + tout), 
                                step=delt, layout=Layout(width='100%')));


# ### Mass balance
# The mass balance is checked here for two time steps in the first cycle. 
# During an injection time step, freshwater with concentration $c_f$ flows into the system at a rate $Q$, while saltwater with concentration $c_s$ flows out of the system at a rate $Q$ (recall that flow is stationary). Hence, the net increase in mass (negative for a decrease) caused by injection during a time step of length $\Delta t$ is 
# \begin{equation}
# \Delta M_\text{injected} = (Q c_f - Q c_s)\Delta t
# \end{equation}
# The mass increase in the system may be computing as the difference in the total mass in the system at the end of the time step (time $t$) and the total mass in the system at the beginning of the time step (time $t-\Delta t$). 
# \begin{equation}
# \Delta M_\text{increase} = \sum_i{(c_i(t)-c_i(t-\Delta t)) (r_{i+1}^2 - r_{i}^2) \pi H n}
# \end{equation}
# During an extraction time step, saltwater with concentration $c_s$ flows into the system at a rate $Q$ at the right boundary, while water with concentration $\bar{c}_0$ is extracted from the system at a rate $Q$, where $\bar{c}_0$ is the mean concentration in the first cell at the end of the time step. Hence, the net increase in mass (negative for a decrease) caused by extraction during a time step is 
# \begin{equation}
# \Delta M_\text{extraction} = (-Q \bar{c}_0 + Q c_s)\Delta t
# \end{equation}
# If the mass balance is met, the change in mass caused by injection or extraction must equal the change in mass in the system. Below, the mass balance is computed for one time step during injection and one type step during extaction for the first cycle. The mass balance is met for 3 significant digits. 

# In[ ]:


# compute concentration during first cycle
c1, times1 = asrmodel(simtime=tin, nstep=nstepin, Qwell=Q)
c2, times2 = asrmodel(simtime=tout, nstep=nstepout, Qwell=-Q, c_init=c1[-1])
for itime in range(nstepout):
    if np.mean(c2[itime, :, 0]) > climit:
        break
c3, times3 = asrmodel(simtime=tout - itime * delt, nstep=nstepout - itime, 
                      Qwell=0, c_init=c2[itime - 1])


# In[ ]:


# during injection
itime = 20
print(f'mass balance injection during time step {itime}')
delMinj = (cf - cs) * Q * delt
print(f'delM injected: {delMinj:.2f} kg')
delconc = (c1[itime, :, :] - c1[itime - 1, :, :])
area = (rb[1:] ** 2 - rb[:-1] ** 2) * np.pi
delMinc = np.sum(delconc * area) * H / nlay * npor
print(f'delM increase: {delMinc:.2f} kg')


# In[ ]:


itime = 50
print(f'mass balance extraction during time step {itime}')
delM1ext = (-np.mean(c2[itime, :, 0]) + cs) * Q * delt
print(f'delM extraction: {delM1ext:.2f} kg')
delconc = (c2[itime, :, :] - c2[itime - 1, :, :])
area = (rb[1:] ** 2 - rb[:-1] ** 2) * np.pi
delMinc = np.sum(delconc * area) * H / nlay * npor
print(f'delM increase:   {delMinc:.2f} kg')


# ### Exercise 3
# Use the Exercise3 notebook to investigate the effect on the recovery efficiency of changing the following parameters:
# 1. Does the recovery efficiency increase or decrease when the dispersivity is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $\alpha_L=1$ m.
# 3. Does the recovery efficiency increase or decrease when the horizontal hydraulic conductivity is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $k=40$ m/d.
# 3. Does the recovery efficiency increase or decrease when the vertical hydraulic conductivity  is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $k_v=10$ m/d.
# 4. Does the recovery efficiency increase or decrease when the injected volume is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $Q=1000$ m$^3$/d.
# 5. Does the recovery efficiency increase or decrease when the salinity of the aquifer is larger? Demonstrate your answer by computing the recovery efficiency for the first cycle for the case that $c_s=30$ g/L. 

# #### References
# Langevin, C.D., 2008. Modeling axisymmetric flow and transport. Groundwater, 46(4), pp.579-590.
