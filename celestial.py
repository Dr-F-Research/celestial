#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:10:48 2023

@author: anthonyferrar
"""

# This code solves the 2d 2-body problem in rectangular coordinates

#import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Constants (SI Units)
G = 6.6743e-11

M_earth = 5.97219e24
r_earth_sun = 1.5e11
v_earth = 29800

M_moon = 7.34767309e22
r_moon_earth = 3.85e8
v_moon = 1.022e3

M_sun = 1.9891e30



# Solution Controls
# time step size (seconds)
deltat = 3600
# number of time steps to run
#n = 32000
days = 1*365
n = 24*days

# Initial Conditions

# Object 1
m1 = M_earth

x1_0 = 0
y1_0 = r_earth_sun

u1_0 = v_earth
v1_0 = 0

# Object 2
m2 = M_sun

x2_0 = 0
y2_0 = 0

u2_0 = 0
v2_0 = 0

# Object 3
m3 = M_moon

x3_0 = 0
y3_0 = r_moon_earth + r_earth_sun

u3_0 = v_earth + v_moon
v3_0 = 0

# Initialize Solution Values

# Object 1
ax1 = [0] * (n+1)
ay1 = [0] * (n+1)
A1  = [0] * (n+1)
u1  = [0] * (n+1)
v1  = [0] * (n+1)
V1  = [0] * (n+1)
x1  = [0] * (n+1)
y1  = [0] * (n+1)
R1  = [0] * (n+1)
Fx1 = [0] * (n+1)
Fy1 = [0] * (n+1)
F1  = [0] * (n+1)

x1[0] = x1_0
y1[0] = y1_0
u1[0] = u1_0
v1[0] = v1_0

R1[0] = (x1[0]**2 + y1[0]**2)**.5
V1[0] = (u1[0]**2 + v1[0]**2)**.5
A1[0] = (ax1[0]**2 + ay1[0]**2)**.5

# Object 2
ax2 = [0] * (n+1)
ay2 = [0] * (n+1)
A2  = [0] * (n+1)
u2  = [0] * (n+1)
v2  = [0] * (n+1)
V2  = [0] * (n+1)
x2  = [0] * (n+1)
y2  = [0] * (n+1)
R2  = [0] * (n+1)
Fx2 = [0] * (n+1)
Fy2 = [0] * (n+1)
F2  = [0] * (n+1)

x2[0] = x2_0
y2[0] = y2_0
u2[0] = u2_0
v2[0] = v2_0

R2[0] = (x2[0]**2 + y2[0]**2)**.5
V2[0] = (u2[0]**2 + v2[0]**2)**.5
A2[0] = (ax2[0]**2 + ay2[0]**2)**.5

# Object 3
ax3 = [0] * (n+1)
ay3 = [0] * (n+1)
A3  = [0] * (n+1)
u3  = [0] * (n+1)
v3  = [0] * (n+1)
V3  = [0] * (n+1)
x3  = [0] * (n+1)
y3  = [0] * (n+1)
R3  = [0] * (n+1)
Fx3 = [0] * (n+1)
Fy3 = [0] * (n+1)
F3  = [0] * (n+1)

x3[0] = x3_0
y3[0] = y3_0
u3[0] = u3_0
v3[0] = v3_0

R3[0] = (x3[0]**2 + y3[0]**2)**.5
V3[0] = (u3[0]**2 + v3[0]**2)**.5
A3[0] = (ax3[0]**2 + ay3[0]**2)**.5


# Loop Time!
for i in range(n):
    #update the force between objects 1 + 2
    r12 = ((x2[i]-x1[i])**2 + (y2[i]-y1[i])**2)**.5
    u12x = (x2[i]-x1[i])/r12
    u12y = (y2[i]-y1[i])/r12
    F12 =  G*m1*m2/(r12**2)

    #update the force between objects 1 + 3
    r13 = ((x3[i]-x1[i])**2 + (y3[i]-y1[i])**2)**.5
    u13x = (x3[i]-x1[i])/r13
    u13y = (y3[i]-y1[i])/r13
    F13 =  G*m1*m3/(r13**2)

    #update the force between objects 2 + 3
    r23 = ((x3[i]-x2[i])**2 + (y3[i]-y2[i])**2)**.5
    u23x = (x3[i]-x2[i])/r23
    u23y = (y3[i]-y2[i])/r23
    F23 =  G*m2*m3/(r23**2)
    
    #update Object 1
    # force acting on Object 1
    Fx1[i] = F12*(u12x) + F13*(u13x)
    Fy1[i] = F12*(u12y) + F13*(u13y)
    
    ax1[i] = Fx1[i]/m1
    ay1[i] = Fy1[i]/m1
    
    u1[i+1] = u1[i] + ax1[i]*deltat
    v1[i+1] = v1[i] + ay1[i]*deltat
    
    x1[i+1] = x1[i] + u1[i]*deltat
    y1[i+1] = y1[i] + v1[i]*deltat

    #Update object 2
    # force acting on Object 2
    Fx2[i] = F12*(-u12x) + F23*(u23x)
    Fy2[i] = F12*(-u12y) + F23*(u23y)
    
    ax2[i] = Fx2[i]/m2
    ay2[i] = Fy2[i]/m2
    
    u2[i+1] = u2[i] + ax2[i]*deltat
    v2[i+1] = v2[i] + ay2[i]*deltat
    
    x2[i+1] = x2[i] + u2[i]*deltat
    y2[i+1] = y2[i] + v2[i]*deltat

    #Update object 3
    # force acting on Object 3
    Fx3[i] = F13*(-u13x) + F23*(-u23x)
    Fy3[i] = F13*(-u13y) + F23*(-u23y)
    
    ax3[i] = Fx3[i]/m3
    ay3[i] = Fy3[i]/m3
    
    u3[i+1] = u3[i] + ax3[i]*deltat
    v3[i+1] = v3[i] + ay3[i]*deltat
    
    x3[i+1] = x3[i] + u3[i]*deltat
    y3[i+1] = y3[i] + v3[i]*deltat


# Visualize Results
fig, ax = plt.subplots()
line1 = ax.plot(x1,y1)
scat1 = ax.scatter(x1,y1)
line2 = ax.plot(x2,y2)
scat2 = ax.scatter(x2,y2)
line3 = ax.plot(x3,y3)
scat3 = ax.scatter(x3,y3)
ax.axis('equal')
plt.show()




# Animate Results
fig, ax = plt.subplots()
scat1 = ax.scatter(x1,y1)
scat2 = ax.scatter(x2,y2)
scat3 = ax.scatter(x3,y3)
line1 = ax.plot(x1,y1)
line2 = ax.plot(x2,y2)
line3 = ax.plot(x3,y3)
ax.axis('equal')

def update(frame):
    x = x1[frame:frame+1]
    y = y1[frame:frame+1]
    data = np.stack([x,y]).T
    scat1.set_offsets(data)
    
    x = x2[frame:frame+1]
    y = y2[frame:frame+1]
    data = np.stack([x,y]).T
    scat2.set_offsets(data)
    
    x = x3[frame:frame+1]
    y = y3[frame:frame+1]
    data = np.stack([x,y]).T
    scat3.set_offsets(data)
    
    return (scat1, scat2, scat3)
    

ani = animation.FuncAnimation(fig=fig, func=update, frames=n, interval=.0001)
plt.show()
