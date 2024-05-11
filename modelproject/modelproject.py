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


class modelproject():

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

         # a. list of variables
        household = ['C1','C2']
        firm = ['k', 'K_lag', 'L_lag', 'Y', 'K', 'L' ]
        prices = ['r','w']

        # b. allocate
        allvarnames = household + firm + prices
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)


    def simulate(self,do_print=True):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.L_lag[0] = par.L_lag_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            self.simulate_before_s(par,sim,t)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = self.find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: self.calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            self.simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')


    def find_s_bracket(self,par,sim,t,maxiter=500,do_print=False):
        """ find bracket for s to search in """

        # a. maximum bracket
        s_min = 0.0 + 1e-8 # save almost nothing
        s_max = 1.0 - 1e-8 # save almost everything

        # b. saving a lot is always possible 
        value = self.calc_euler_error(s_max,par,sim,t)
        sign_max = np.sign(value)
        if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

        # c. find bracket      
        lower = s_min
        upper = s_max

        it = 0
        while it < maxiter:
                    
            # i. midpoint and value
            s = (lower+upper)/2 # midpoint
            value = self.calc_euler_error(s,par,sim,t)

            if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

            # ii. check conditions
            valid = not np.isnan(value)
            correct_sign = np.sign(value)*sign_max < 0
            
            # iii. next step
            if valid and correct_sign: # found!
                s_min = s
                s_max = upper
                if do_print: 
                    print(f'bracket to search in with opposite signed errors:')
                    print(f'[{s_min:12.8f}-{s_max:12.8f}]')
                return s_min,s_max
            elif not valid: # too low s -> increase lower bound
                lower = s
            else: # too high s -> increase upper bound
                upper = s

            # iv. increment
            it += 1

        raise Exception('cannot find bracket for s')


    def calc_euler_error(self, s, par, sim, t):
        # Make sure to pass 's' when calling simulate_after_s
        self.simulate_after_s(par, sim, t, s)
        self.simulate_before_s(par, sim, t + 1)

        # Now continue with your Euler equation calculation
        LHS = sim.C1[t]**(-1)
        RHS = (1 + sim.r[t+1]) * par.beta * sim.C2[t+1]**(-1)
        return LHS - RHS


    def simulate_before_s(self,par,sim,t):
            """ simulate forward """

        
            if t > 0:
                sim.K_lag[t] = sim.K[t-1]
                sim.L_lag[t] = sim.L[t-1]

            # a. production and factor prices
            if par.prod == 'cobb-douglas':

                # i. production
                sim.Y[t] = ((sim.K_lag[t]**par.alpha)*(sim.L_lag[t]**(1-par.alpha)))

                #ii. 
                sim.k[t] = sim.K_lag[t] / sim.L_lag[t]


                # iii. factor prices
                sim.r[t] = par.alpha*sim.k[t]**(par.alpha-1)
                sim.w[t] = (1-par.alpha)*sim.k[t]**par.alpha

            # b. consumption
            sim.C2[t] = (1+sim.r[t])*sim.K_lag[t]

    def simulate_after_s(self, par, sim, t, s):
        # Calculate consumption of the young
        sim.C1[t] = sim.w[t] * (1.0 - s)

        # Calculate end-of-period capital stocks
        I = sim.Y[t] - sim.C1[t] - sim.C2[t]
        sim.K[t+1] = sim.K[t] + I

        # Debugging to check what is happening with I and K[t+1]
        print(f"Time {t}, Investment I: {I}, K[t+1]: {sim.K[t+1]}")

