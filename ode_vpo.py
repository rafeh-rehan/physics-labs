import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy.signal import argrelmax


"""
This program solves and visualizes the solutions to the differential equations
describing the Unforced and Forced Van der Pol oscillators using scipy's odeint
method.

https://en.wikipedia.org/wiki/Van_der_Pol_oscillator

"""



""" Unforced Van der Pol Oscillator """

#Initial conditions
x0 = 1
v0 = 2
mu=20  # Choose the strength of the non-linear damping effect

#time step
dt=0.001
t= np.arange(0.0,50.0,dt)

#solving ODE using odeint
#define a function for our system of first order odes and initial conditions
ics = [x0,v0]
def pend(y,t,mu):
    x,v = y
    dydt = [v , mu*(1-x**2)*v - x]
    return dydt

#store the solution into a new variable
sol = scint.odeint(pend, ics, t, args=(mu,))
x_ode = sol[:,0]
v_ode = sol[:,1]

# Plot numerical solutions and
plt.figure(1, figsize=(10,10))

#Position plot
plt.subplot(3,1,1)
plt.plot(t,x_ode)
plt.legend((r'Position',), loc='lower right')
plt.ylabel('x(t)')
plt.xlabel('t')
plt.grid('on')

#Velocity plot
plt.subplot(3,1,2)
plt.plot(t,v_ode)
plt.legend((r'Velocity',), loc='lower right')
plt.ylabel('v(t)')
plt.xlabel('t')
plt.grid('on')

#2D phase space
plt.subplot(3,1,3)
plt.plot(x_ode,v_ode)
plt.legend((r'Position vs Velocity',), loc='lower right')
plt.ylabel('v(t)')
plt.xlabel('x(t)')
plt.grid('on')

plt.suptitle('Unforced van der Pol for mu=' + ('%.2f' % mu) + ' dt=' +('%.3f' % dt))



## Analyzing the period T of the oscillation as a function of the damping parameter \mu


#time step
dt=0.01
t= np.arange(0.0,300.01,dt)

#\mu step for iteration over mu when calculating periods for each value
dmu=0.01
mu=np.arange(0.01,50.01,dmu)

#initialize empty period array
T=[]
for i in range(len(mu)):
    #store the solution into a new variable
    sol = scint.odeint(pend, ics, t, args=(mu[i],))
    x_ode = sol[:,0]

    #Estimating the period of the oscillator
    #store array of indices of local maxima in x_ode after its settled into limit cycle
    maxima = argrelmax(x_ode[200:])[0]

    #if the maxima array is non-empty, calculate the period
    if len(maxima) > 0:
        num_of_periods = len(maxima) - 1
        delta_t = (maxima[-1] - maxima[0])*dt
        T.append(delta_t/num_of_periods)

#Theoretical model in the large \mu approximation
def T_mu(u):
    a = 2.33810741
    b = 1.3246
    return (3-2*np.log(2))*u + 3*a/(u**(1/3)) - 2*np.log(u)/(3*u) - b/(u**2)

# Split mu array into small and large mu
smol_mu = np.array([u for u in mu if u <= .5])
big_mu  = np.array([u for u in mu if u >= 40])


plt.figure(2, figsize=(10,10))
#Period vs \mu plot for 0.1 <= mu <= 15
plt.subplot(2,1,1)
plt.plot(smol_mu,T[:len(smol_mu)], smol_mu, T_mu(smol_mu))
plt.legend((r'Numerical',r'Theoretical'), loc='lower right')
plt.ylabel('Period (sec)')
plt.xlabel(r'$\mu$')
plt.grid('on')

#Period vs mu plot for \mu > 15
plt.subplot(2,1,2)
plt.plot(big_mu, T[-len(big_mu):], big_mu, T_mu(big_mu))
plt.legend((r'Numerical',r'Theoretical',r'Large $\mu$ theoretical',), loc='lower right')
plt.ylabel('Period (sec)')
plt.xlabel(r'$\mu$')
plt.grid('on')

plt.suptitle(r'Period vs $\mu$ for Unforced Van der Pol')



""" Forced/Driven Van der Pol Oscillator"""



#Initial conditions, change what you like :)
x0 = 2
v0 = 0
phi0 = 0
A = 15
mu = 3
w = 3.98
#

#time step
dt=0.001
t= np.arange(0.0,100.0,dt)

#solving ODE using odeint
#define a function for our system of first order odes and initial conditions
ics = [x0,v0,phi0]
def pend(y,t):
    x,v,phi = y
    dydt = [v , A*np.cos(w*t) + mu*(1-x**2)*v - x, w]
    return dydt

#storing solutions in new variables
sol = scint.odeint(pend, ics, t)
x_ode = sol[:,0]
v_ode = sol[:,1]
phi_ode = sol[:,2]

#All plots are done after the motion is settled in its limit cycle
plt.figure(3,figsize=(10,10))
#Position plot
plt.subplot(3,1,1)
plt.plot(t[3000:],x_ode[3000:])
plt.ylabel('x(t)')
plt.xlabel('t')
plt.grid('on')

#Velocity plot
plt.subplot(3,1,2)
plt.plot(t[3000:],v_ode[3000:])
plt.ylabel('v(t)')
plt.xlabel('t')
plt.grid('on')

#3D phase space projection onto (x,v)
plt.subplot(3,1,3)
plt.plot(x_ode[3000:],v_ode[3000:])
plt.ylabel('v(t)')
plt.xlabel('x(t)')
plt.grid('on')

plt.suptitle('Forced Van der Pol for w=' + ('%.3f' % w))
