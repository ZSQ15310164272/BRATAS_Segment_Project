#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 00:56:38 2018

@author: ZSQ
"""

import numpy as np
import matplotlib.pyplot as plt

lab = label[:, :, 0]
x0 = np.where(lab == 0)[0]
y0 = np.where(lab == 0)[1]

x1 = np.where(lab == 1)[0]
y1 = np.where(lab == 1)[1]

x2 = np.where(lab == 2)[0]
y2 = np.where(lab == 2)[1]

x3 = np.where(lab == 3)[0]
y3 = np.where(lab == 3)[1]

x4 = np.where(lab == 4)[0]
y4 = np.where(lab == 4)[1]

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('Scatter Plot')  
plt.xlabel('X')  
plt.ylabel('Y')  
ax1.scatter(x0, y0, c = 'w', marker = 'o') 
ax1.scatter(x1, y1, c = 'r', marker = 'o') 
ax1.scatter(x2, y2, c = 'g', marker = 'o') 
ax1.scatter(x3, y3, c = 'k', marker = 'o') 
ax1.scatter(x4, y4, c = 'b', marker = 'o') 
plt.legend('x1')  
plt.show()  
f = plt.figure()



