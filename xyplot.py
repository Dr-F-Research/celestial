#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:07:33 2023

@author: anthonyferrar
"""


# This code solves plots points as circles using pygame


import numpy as np
import pygame


###############################################################################
# Graph Window Controls

x_min = -10.0
x_max =  10.0
y_min = -10.0
y_max =  10.0

width_px, height_px = 800, 800
#define the origin as the center of the animation window
origin = (width_px // 2 , height_px // 2) 

###############################################################################
# points to plot

x = [0,0,0]
y = [0,5,-8]

n = len(x)

x_px = [0] * n
y_px = [0] * n

###############################################################################
#prep the animation

#Frame Rate
speed = 60

#Initialize the window
pygame.init() 
screen = pygame.display.set_mode((width_px, height_px))
pygame.display.set_caption("Graph some points!")

#number of inner loops to run per frame (outer loop)
frame = 0
clock = pygame.time.Clock()

def redrawWindow():
    screen.fill((255, 255, 255))
    for i in range(n):
        pygame.draw.circle(screen, (0, 0, 0), [x_px[i], y_px[i]],6)

    pygame.display.update()

def xy__to_pixels(x, y, x_min, x_max, y_min, y_max, width_px, height_px):
    w_scale = width_px/(x_max - x_min) #pixels/inch
    h_scale = height_px/(y_max - y_min) #pixels/inch
    
    x_px = x * w_scale + width_px / 2
    y_px = height_px / 2 - y * h_scale
 
    return int(x_px), int(y_px)

###############################################################################
#Convert points

for i in range(n):
    x_px[i], y_px[i] = xy__to_pixels(x[i], y[i], x_min, x_max, y_min, y_max, width_px, height_px)

##############################################################################
#Run the sim + animate
running = True
while running:
    
    clock.tick(speed)    
    redrawWindow() 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()













