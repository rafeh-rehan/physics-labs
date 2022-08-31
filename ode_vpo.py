import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint


"""
This program solves the differential equation describing the Van der Pol oscillator
using scipy's odeint method.

https://en.wikipedia.org/wiki/Van_der_Pol_oscillator

"""

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
