
# This code solves the 2d 2-body problem in rectangular coordinates
# also, my first real try at object-oriented programming

# This code uses SI units:
#   mass: kilograms
#   distance: meters
#   velocity: meters/second
#   accelleration: meters/second^2

import numpy as np
import pygame
import csv

###############################################################################
# Solution Controls

# Real-world time per virtual year
clock_time_per_virtual_year = 15
#Frame Rate
speed = 60


#Outer loop iterations per virtual year
frames_per_year = clock_time_per_virtual_year * speed 
#Number of seconds in a year
seconds_per_year = 60*60*24*365
#Vitrtual seconds per frame
virtual_seconds_per_frame = seconds_per_year / frames_per_year
#Microtime - inner loop iterations per frame
microtime = 1000
#virtual time step per inner loop
deltat = virtual_seconds_per_frame / microtime


#speed up time - seconds per virtual year
year_px = 10


###############################################################################
# Class Definitions

class Planet:
    #Universal constants
    # Astronomical Unit
    au = 1.495978707e11
    # Graviational Constant
    G = 6.6743e-11
    
    #Build a list of all instances of this class
    all = []
    
    def __init__(self, name='Earthlike Planet', mass=5.97219e24, radius=6.378100e6, orbital_speed=29800, orbital_radius=1.5e11, location=[0,0], velocity=[0,0], colorspec=(255,255,255), radius_px=12):
        #Validate input arguments
        
        #Assign values to self object
        self.name = name
        self.mass = mass
        self.radius = radius
        self.orbital_speed = orbital_speed
        self.orbital_radius = orbital_radius
        self.location = location
        self.velocity = velocity
        self.colorspec = colorspec
        self.radius_px = radius_px
    
        #Actions to execute
        Planet.all.append(self)
        
    def print_summary(self):
        print(" ")
        print(f"Planet created: {self.name}")
        print(f"Mass [kg]: {self.mass}")
        print(f"Radius [m]: {self.radius}")
        print(f"Orbital Speed [m/s]: {self.orbital_speed}")
        print(f"Orbital Radius [m]: {self.orbital_radius}")
        print(f"Location (x,y) [m]: {self.location}")
        print(f"Velocity (v_x,v_y) [m]/s: {self.velocity}")
        print(f"Display Color [RGB-256 colors]: {self.colorspec}")
        print(f"Display Radius [pixels]: {self.radius_px}")
        print(" ")
        
    def distance_to_others(self):
        # This function calculates the distance between this Planet and all others in the simulation
        x1 = self.location[0]
        y1 = self.location[1]
        
        #distance
        r12 = [0] * (len(Planet.all) - 1)
        
        i = 0
        for item in Planet.all:
            if item.name != self.name:
                x2 = item.location[0]
                y2 = item.location[1]
                r12[i] = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                i += 1
                
        self.r12 = r12

        
    def unit_vectors_to_others(self):
        # This function determines unit vectors pointing FROM this Planet TO all others in the simulation
        
        # This MUST be run AFTER distance_to_others(), so just force it
        #self.distance_to_others()
        
        x1 = self.location[0]
        y1 = self.location[1]
        
        #unit vectors
        u12 = [(0,0)] * (len(Planet.all) - 1)
        
        i = 0
        for item in Planet.all:
            if item.name != self.name:
                x2 = item.location[0]
                y2 = item.location[1]
                
                u12x = (x2 - x1)/self.r12[i]
                u12y = (y2 - y1)/self.r12[i]
                u12[i] = (u12x,u12y)
                i += 1
                
        self.u12 = u12

        
    def other_masses(self):
        # This function builds a list of the masses of the all other planets in the simulation
        # Useful for iterating through entire system
        m_others = [0] * (len(Planet.all) - 1)
        
        i = 0
        for item in Planet.all:
            if item.name != self.name:
                m_others[i] = item.mass
                i += 1
        
        self.m_others = m_others
        
    def update_forces(self):
        #This function calculates the forces (and x,y components of force) between this planet and all others in the simulation
        #Forces point FROM this planet TO the others
        
        F12 = [0] * (len(Planet.all) - 1)
        FXY = [(0,0)] * (len(Planet.all) - 1)
        
        i = 0
        for item in Planet.all:
            if item.name != self.name:
               F12[i] = Planet.G * self.mass * self.m_others[i] / (self.r12[i]**2)
               
               u12 = self.u12[i]
               FX = F12[i] * u12[0]
               FY = F12[i] * u12[1]
               
               FXY[i] = (FX, FY)
               i +=1
               
        self.F12 = F12
        self.FXY = FXY
        
    def update_net_force(self):
        #This function calculates the net force acting on this planet, and its components
        
        self.F_net_XY = [0,0]

        
        for i in range(len(Planet.all) - 1):
            FXY = self.FXY[i]
            self.F_net_XY[0] += FXY[0]
            self.F_net_XY[1] += FXY[1]
        
        self.F_net = np.sqrt(self.F_net_XY[0]**2 + self.F_net_XY[1]**2)
        
    def update_acceleration(self):
        #This funciton calculates the acceleration components of this planet
        ax = self.F_net_XY[0] / self.mass
        ay = self.F_net_XY[1] / self.mass
        
        self.a = [ax,ay]
    
    def update_velocity(self, deltat):
        #This function calculates the new velocity components of this planet using v2 = v1 + a*deltat
        u_old = self.velocity[0]
        v_old = self.velocity[1]
        
        ax = self.a[0]
        ay = self.a[1]
        u_new = u_old + ax * deltat
        v_new = v_old + ay * deltat
        
        self.velocity = [u_new, v_new]
    
    def update_location(self, deltat):
        #This function calculates the new position (x,y) of this planet using x2 = x1 + vx*deltat
        x_old = self.location[0]
        y_old = self.location[1]
        
        u = self.velocity[0]
        v = self.velocity[1]
        
        x = x_old + u * deltat
        y = y_old + v * deltat
        
        self.location = [x, y]
      
    @classmethod
    def random_start_angle(cls, radial_position):
        #This function chooses a random starting position in polar coordinates and converts them to x,y
        theta = np.random.random()*2*np.pi
        
        return [radial_position * np.cos(theta), radial_position * np.sin(theta)], theta
        
    
    @classmethod
    def instantiate_from_csv(cls,fname):
        with open(fname,'r') as f:
        #with open('Planetary_data.csv','r') as f:
            reader = csv.DictReader(f)
            planets = list(reader)
        """
        for planet in planets:
            print(planet.get('Name'))
            print(float(planet.get('Mass (kg)')))
            print(float(planet.get('Radius (m)')))
            print(float(planet.get('Orbital Velocity (m/s)')))
            print(float(planet.get('Orbital Radius (m)')))
            print([0,float(planet.get('Orbital Radius (m)'))])
            print([float(planet.get('Orbital Velocity (m/s)')),0])
        """
        
        for planet in planets:
            
            rand_loc,theta = Planet.random_start_angle(float(planet.get('Orbital Radius (m)')))
            phi = np.pi/2 - theta
            V = [float(planet.get('Orbital Velocity (m/s)')) * np.cos(phi), float(planet.get('Orbital Velocity (m/s)')) * np.sin(phi)]
            
            if rand_loc[0] >= 0: #Q1,4
                if rand_loc[1] >= 0: #Q1
                    V = [np.abs(V[0]),-1*np.abs(V[1])]
                else: #Q4
                    V = [-1*np.abs(V[0]),-1*np.abs(V[1])]
            else: #Q2,3    
                if rand_loc[1] >= 0: #Q2
                    V = [np.abs(V[0]),np.abs(V[1])]
                else: #Q3
                    V = [-1*np.abs(V[0]),np.abs(V[1])]
            
            Planet(
                name=planet.get('Name'),
                mass=float(planet.get('Mass (kg)')),
                radius=float(planet.get('Radius (m)')),
                orbital_speed=float(planet.get('Orbital Velocity (m/s)')),
                orbital_radius=float(planet.get('Orbital Radius (m)')),
                location=rand_loc,
                #location=[0,float(planet.get('Orbital Radius (m)'))],
                velocity=V,
                #velocity=[float(planet.get('Orbital Velocity (m/s)')),0],
                colorspec=(255,255,102),
                radius_px=4
                )
Planet.instantiate_from_csv('sun_to_mars.csv')
#Planet.instantiate_from_csv('all_planets_no_moons.csv')        
#Planet.instantiate_from_csv('sun_to_earth.csv')
#Planet.instantiate_from_csv('Planetary_data.csv')




###############################################################################
# Physical Constants

# Astronomical Unit
au = Planet.au

# Graviational Constant
G = Planet.G

# Time


###############################################################################
# Obeject Data

"""
# Create the planet objects
sun = Planet("Sun", 1.9891e30, 6.96e8, 0, 0, [0.0,0.0], [0.0,0.0], (255,255,102), 24)
earth = Planet("Earth", 5.97219e24, 6.378100e6, 29800, 1.5e11, [0, 1.5e11], [29800,0], (0,89,179), 12)
moon = Planet("Moon",7.34767309e22, 1.7374e6, 1022, 3.85e8, [0, earth.orbital_radius + 3.85e8],[earth.orbital_speed + 1022, 0], (112,128,144), 6)
"""

for instance in Planet.all:
    instance.print_summary()
    instance.distance_to_others()
    instance.unit_vectors_to_others()
    instance.other_masses()


###############################################################################
# Window Controls


x_min = -1.6*au
x_max =  1.6*au
y_min = -1.6*au
y_max =  1.6*au

"""
x_min = -50*au
x_max =  50*au
y_min = -50*au
y_max =  50*au
"""

width_px, height_px = 800, 800
#define the origin as the center of the animation window
origin = (width_px // 2 , height_px // 2) 


###############################################################################
#prep the animation

pygame.init() 
screen = pygame.display.set_mode((width_px, height_px))
pygame.display.set_caption("Gravity Always Wins")


#number of inner loops to run per frame (outer loop)
#microtime = int(1//(speed*deltat))
clock = pygame.time.Clock()

t = 0 #in-simulation time, seconds



###############################################################################
#function definitions

# determine the force between two objects
def update_force(m1,x1,y1,m2,x2,y2,G):
    r12 = ((x2-x1)**2 + (y2-y1)**2)**.5
    u12x = (x2-x1)/r12
    u12y = (y2-y1)/r12
    F12 =  G*m1*m2/(r12**2)
    return (r12, (u12x, u12y), F12)

#def update_force_loop():
        

# advance the position of an object using v*deltat integration
def update_position(m,x,y,u,v,Fx,Fy,deltat):    
    ax = Fx/m
    ay = Fy/m
    
    u_new = u + ax*deltat
    v_new = v + ay*deltat
    
    x_new = x + u_new*deltat
    y_new = y + v_new*deltat
    return (x_new, y_new, u_new, v_new)

def xy__to_pixels(x, y, x_min, x_max, y_min, y_max, width_px, height_px):
    # convert an x,y location meansured in [units] to a location measured in pixels
    #     top left corner is 0,0 in pixels, and y increases down
    w_scale = width_px/(x_max - x_min) #pixels/[unit]
    h_scale = height_px/(y_max - y_min) #pixels/[inch]
    
    x_px = x * w_scale + width_px / 2
    y_px = height_px / 2 - y * h_scale
 
    return int(x_px), int(y_px)


def redrawWindow():
    # update the animation window 1/main loop
    screen.fill((0, 0, 0))
    
    for instance in Planet.all:
        #Convert x,y locations to pixels
        x_px, y_px = xy__to_pixels(instance.location[0], instance.location[1], x_min, x_max, y_min, y_max, width_px, height_px)
        #Draw circle
        pygame.draw.circle(screen, instance.colorspec, [x_px, y_px], instance.radius_px)    
    
    pygame.display.update()


###############################################################################
#Run the sim + animate
running = True
while running:
    
    #Inner loop for microtime
    for i in range(microtime):
        t += deltat
        for instance in Planet.all:
            instance.distance_to_others()
            instance.unit_vectors_to_others()
            instance.update_forces()
            instance.update_net_force()
            instance.update_acceleration()
            instance.update_velocity(deltat)
            instance.update_location(deltat)
     
        
    #Update Graphic
    clock.tick(speed)
    redrawWindow()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()