import numpy as np
from lib import KFilter as kf


# Define a class to receive the characteristics of each line detection

class Line():
    def __init__(self, x0, t, limit):
        self.logic = False  
        
        self.stack_x, self.x, self.x_old = [], [], [] 
        self.stack_radius, self.radius = [], [] 
        self.stack_position, self.position = [], []
        
        # max distance between lines in consecutives frames
        self.dist = 0
        # max diff in pixel
        self.start=0
        self.end=0
        
        self.x0=x0
        self.t=t
        self.limit=limit
        
        self.draw = None
        
    def detected(self, x):
        if x != []:
            self.logic = True
        else:
            self.logic = False
            
    def max_distance(self, x, l=10):
        if self.x_old != [] and x !=[]:
            dist = abs( (x - self.x_old) / self.x_old )
            dist = sum(dist) / len(dist)
            
            self.dist = dist
            self.start=abs(np.mean(x[0:l]) - np.mean(self.x_old[0:l]))
            self.end=abs(np.mean(x[-l:]) - np.mean(self.x_old[-l:]))
        else:
            self.dist = 0
            self.start=0
            self.end=0
            
    def update(self, x, radius, position):
        self.detected(x)
        self.max_distance(x)
        if self.logic:
            if self.dist < self.limit:
                if len(self.stack_x) > self.t:
                    self.stack_x.pop(0)
                    self.stack_radius.pop(0)
                    self.stack_position.pop(0)
                    
                #x=self.update_filter(x)
                self.stack_x.append(x)
                self.x = np.mean(self.stack_x, axis=0)
                self.x_old = self.x[:]

                self.stack_radius.append(radius)
                self.radius = np.mean(self.stack_radius, axis=0)

                self.stack_position.append(position)
                self.position = np.mean(self.stack_position, axis=0)
                
                #self.x0=int(self.x[0])
                
    def update_filter(self, x):
        fltr = kf.Filter(x0=self.x0)
        ## linear kalman filter
        n=len(x) - 1
        for i in range(n, -1, -1):
            measure=x[i]
            if fltr.likelihood(measure) > 0.04:
                fltr.run(measure)
                x[i]=int(fltr.position())
                
        self.x0=x[0]
        return x
    
    def draw_line(self, result):
        self.draw = result