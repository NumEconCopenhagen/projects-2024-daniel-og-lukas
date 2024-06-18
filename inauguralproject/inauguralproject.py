import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import brentq


class ExchangeEconomyClass:
    #par = self.par = SimpleNamespace()
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        par.w1B = 1 - par.w1A 
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        return x1A**self.par.alpha*x2A**(1-self.par.alpha)

    def utility_B(self,x1B,x2B):
        return x1B**self.par.beta*x2B**(1-self.par.beta)

    def demand_A(self,p1):
        return self.par.alpha*(p1*self.par.w1A+self.par.w2A)/p1, (1-self.par.alpha)*(p1*self.par.w1A+self.par.w2A)

    def demand_B(self,p1):
        return self.par.beta*(p1*self.par.w1B+self.par.w2B)/p1, (1-self.par.beta)*(p1*self.par.w1B+self.par.w2B)

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def edgeworth(self, x_values, y_values):
        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0
        
        # b. figure set up
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel(r"$x_1^A$")
        ax_A.set_ylabel(r"$x_2^A$")

        temp = ax_A.twiny()
        temp.set_ylabel(r"$x_2^B$")
        ax_B = temp.twinx()
        ax_B.set_xlabel(r"$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # A
        ax_A.scatter(self.par.w1A, self.par.w2A, marker='s', color='black', label='endowment')
        ax_A.scatter(x_values,y_values,marker='o',color='black',s=0.1,label='Pareto improvements')

        # limits
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))

        plt.show()

class ExchangeEconomyClass2:
    def __init__(self, w1A, w2A):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = w1A
        par.w2A = w2A
        par.w1B = 1 - par.w1A 
        par.w2B = 1 - par.w2A

    def utility_A(self, x1A, x2A):
        return x1A**self.par.alpha * x2A**(1-self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B**self.par.beta * x2B**(1-self.par.beta)

    def demand_A(self, p1):
        budget = p1 * self.par.w1A + self.par.w2A
        return self.par.alpha * budget / p1, (1 - self.par.alpha) * budget

    def demand_B(self, p1):
        budget = p1 * self.par.w1B + self.par.w2B
        return self.par.beta * budget / p1, (1 - self.par.beta) * budget

    def check_market_clearing(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        eps1 = x1A + x1B - 1
        eps2 = x2A + x2B - 1

        return eps1, eps2

def find_market_equilibrium_price(economy):
    # We define a function to find the root of the excess demand
    def excess_demand(p1):
        eps1, _ = economy.check_market_clearing(p1)
        return eps1

    # We use a root finding algorithm to find the market-clearing price
    p1_equilibrium = brentq(excess_demand, 0.01, 10)
    return p1_equilibrium

