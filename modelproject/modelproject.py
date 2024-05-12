from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result

class OLGModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # Model parameters
        par.alpha = 0.35
        par.n = 0.02
        par.rho = 0.03
        par.beta = 1/(1+par.rho)

        # Others
        par.K_lag_ini = 0.1
        par.L_lag_ini = 1.0 
        par.simT = 50 
    
    def allocate(self):
        """ allocate arrays for simulation """
        par = self.par
        sim = self.sim

        # Initialize arrays to zero instead of NaN to prevent propagation of NaNs
        allvarnames = ['C1', 'C2', 'k', 'K_lag', 'L_lag', 'Y', 'K', 'L', 'r', 'w']
        for varname in allvarnames:
            sim.__dict__[varname] = np.zeros(par.simT)

    def simulate(self,do_print=True):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.L_lag[0] = par.L_lag_ini

        # Set an initial value for s
        s = 0.41

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par, sim, t, s)

            if t == par.simT-1: continue          

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def simulate_before_s(par,sim,t,s):
        """ simulate forward """

        if t > 0:
            sim.K_lag[t] = sim.k[t-1]
            

        # ii. factor prices
        sim.r[t] = par.alpha * (sim.K_lag[t] ** (par.alpha - 1)) 
        sim.w[t] = (1 - par.alpha) * (sim.K_lag[t] ** par.alpha)

        #capital
        sim.k[t] = ((1 - par.alpha) * sim.K_lag[t]**par.alpha) / ((1 + par.n) * (2 + par.rho))
       

        # c. consumption
        sim.C2[t] = (1+sim.r[t])*(sim.K_lag[t])  


def simulate_after_s(par,sim,t,s):
        """ simulate forward """

        # a. consumption of young
        sim.C1[t] = sim.w[t]*(1.0-s)


class OLGModelClass2():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # Model parameters
        par.alpha = 0.35
        par.n = 0.02
        par.rho = 0.03
        par.tau = 0.00
        par.beta = 1/(1+par.rho)
        par.prod = 'cobb-douglas'

        # Others
        par.K_lag_ini = 0.1
        par.L_lag_ini = 1.0 
        par.simT = 50 
    
    def allocate(self):
        """ allocate arrays for simulation """
        par = self.par
        sim = self.sim

        # Initialize arrays to zero instead of NaN to prevent propagation of NaNs
        allvarnames = ['C1', 'C2', 'k', 'K_lag', 'L_lag', 'Y', 'K', 'L', 'r', 'w']
        for varname in allvarnames:
            sim.__dict__[varname] = np.zeros(par.simT)

    def simulate(self,do_print=True):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.L_lag[0] = par.L_lag_ini

        # Set an initial value for s
        s = 0.41

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par, sim, t, s)

            if t == par.simT-1: continue          

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def simulate_before_s(par,sim,t,s):
        """ simulate forward """

        if t > 0:
            sim.K_lag[t] = sim.k[t-1]
            

        # ii. factor prices
        sim.r[t] = par.alpha * (sim.K_lag[t] ** (par.alpha - 1)) 
        sim.w[t] = (1 - par.alpha) * (sim.K_lag[t] ** par.alpha)

        #capital
        sim.k[t] = ((1 - par.alpha) * sim.K_lag[t]**par.alpha) / ((1 + par.n) * (2 + par.rho))
       

        # c. consumption
        sim.C2[t] = (1+sim.r[t])*(sim.K_lag[t])  


def simulate_after_s(par,sim,t,s):
        """ simulate forward """

        # a. consumption of young
        sim.C1[t] = sim.w[t]*(1.0-s)