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

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

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

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = sim.C1[t]**(-1)
    RHS = (1+sim.r[t+1]) * par.beta * sim.C2[t+1]**(-1)

    return LHS-RHS

def simulate_before_s(par,sim,t):
    """ simulate forward """

    if t > 0:
        sim.K_lag[t] = sim.K_lag[t-1]
        sim.L_lag[t] = sim.L_lag[t-1]*(1-par.n)

    if par.prod == 'cobb-douglas':

        sim.Y[t] = sim.K_lag[t] ** par.alpha * sim.L_lag[t] ** (1 - par.alpha)
        sim.r[t] = par.alpha * (sim.K_lag[t] ** (par.alpha - 1)) * (sim.L_lag[t] ** (1 - par.alpha))
        sim.w[t] = (1 - par.alpha) * (sim.K_lag[t] ** par.alpha) * (sim.L_lag[t] ** (-par.alpha))


    # c. consumption
    sim.C2[t] = (1+sim.r[t])*(sim.K_lag[t])


def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    sim.k[t]=sim.K_lag[t]/sim.L_lag[t]
    # a. consumption of young
    sim.C1[t] = sim.w[t]*(1.0-s)


    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t]
    sim.K[t] = sim.K_lag[t] + I


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

        for t in range(par.simT):
            self.simulate_before_s(par, sim, t)
            if t == par.simT - 1:
                continue

        # Ensure 't' is passed to find_s_bracket
        s_min, s_max = self.find_s_bracket(par, sim, t)

        # Setting up the lambda for the optimizer
        obj = lambda s: self.calc_euler_error(s, par, sim, t)
        result = optimize.root_scalar(obj, bracket=(s_min, s_max), method='bisect')
        s = result.root

        # Simulate after 's' is found
        self.simulate_after_s(par, sim, t, s)

        if do_print:
            print(f"Simulation completed in {time.time() - t0:.2f} seconds")


    def find_s_bracket(self, par, sim, t, maxiter=500, do_print=False):
        s_min = 0.0 + 1e-8  # save almost nothing
        s_max = 1.0 - 1e-8  # save almost everything

        # Calculate euler error at s_max
        value = self.calc_euler_error(s_max, par, sim, t)
        sign_max = np.sign(value)
        if do_print:
            print(f"Euler error for s={s_max:.4f} is {value:.4f}")

        lower = s_min
        upper = s_max
        it = 0
        while it < maxiter:
            s = (lower + upper) / 2
            value = self.calc_euler_error(s, par, sim, t)
            if do_print:
                print(f"Euler error for s={s:.4f} is {value:.4f}")

            valid = not np.isnan(value)
            correct_sign = np.sign(value) * sign_max < 0

            if valid and correct_sign:
                return s, upper  # Found a valid bracket
            elif not valid:
                lower = s  # Increase lower bound
            else:
                upper = s  # Decrease upper bound
            it += 1

        raise Exception("Cannot find a valid bracket for 's'")


    def calc_euler_error(self, s, par, sim, t):
        # Simulate the state of the system after savings 's' at time 't'
        self.simulate_after_s(par, sim, t, s)

        # Update the system state to time 't+1' before savings decision
        if t + 1 < par.simT:
            self.simulate_before_s(par, sim, t + 1)

        # Calculate the left-hand side (LHS) of the Euler equation
        LHS = sim.C1[t] ** (-1)  # Utility function derivative w.r.t. C1

        # Calculate the right-hand side (RHS) of the Euler equation, ensuring t+1 is within bounds
        RHS = (1 + sim.r[t + 1]) * par.beta * sim.C2[t + 1] ** (-1)
        

        # Euler error is the difference between the LHS and the RHS
        euler_error = LHS - RHS

        # Debugging print statements to show the calculation
        print(f"Time {t}: s={s:.4f}, LHS={LHS:.4f}, RHS={RHS:.4f}, Euler Error={euler_error:.4f}")

        return euler_error

    def simulate_before_s(self, par, sim, t):
        if t == 0:
            sim.K_lag[t] = par.K_lag_ini
            sim.L_lag[t] = par.L_lag_ini
        else:
            sim.L_lag[t] = sim.L_lag[t-1] * (1 + par.n)
            sim.K_lag[t] = sim.K_lag[t-1]

        # Assuming Cobb-Douglas production function
        sim.Y[t] = sim.K_lag[t] ** par.alpha * sim.L_lag[t] ** (1 - par.alpha)
        sim.r[t] = par.alpha * (sim.K_lag[t] ** (par.alpha - 1)) * (sim.L_lag[t] ** (1 - par.alpha))
        sim.w[t] = (1 - par.alpha) * (sim.K_lag[t] ** par.alpha) * (sim.L_lag[t] ** (-par.alpha))

        print(f"Time {t}: K_lag={sim.K_lag[t]}, L_lag={sim.L_lag[t]}, Y={sim.Y[t]}, r={sim.r[t]}, w={sim.w[t]}")


        # d. Consumption before s 
        sim.C2[t] = (1+sim.r[t])*(sim.K_lag[t]) 
        print(f"Time {t}: Pre-savings C2={sim.C2[t]:.4f}, r={sim.r[t]:.4f}")

    def simulate_after_s(self, par, sim, t, s):

        sim.k[t]=sim.K_lag[t]/sim.L_lag[t]

        print(f"Debug at time {t}: s={s}, w[t]={sim.w[t]}, K[t]={sim.K[t]}, Y[t]={sim.Y[t]}")
        # Calculate consumption of the young
        sim.C1[t] = sim.w[t] * (1.0 - s)

        # Calculate end-of-period capital stocks
        I = sim.Y[t] - sim.C1[t] - sim.C2[t]
        sim.K[t] = sim.K_lag[t] + I
        print(f"Time {t}: s={s:.4f}, w={sim.w[t]:.4f}, C1={sim.C1[t]:.4f}, Pre-C2={sim.C2[t]:.4f}, Post-C2={(1+sim.r[t])*sim.K[t]:.4f}, K={sim.K[t]:.4f}, I={I:.4f}")


