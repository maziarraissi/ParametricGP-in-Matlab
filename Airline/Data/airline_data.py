#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:37:33 2017

@author: mraissi
"""

import numpy as np
import pandas as pd
import scipy.io as sio

data = pd.read_pickle('airline.pickle')
data.ArrTime = 60*np.floor(data.ArrTime/100)+np.mod(data.ArrTime, 100)
data.DepTime = 60*np.floor(data.DepTime/100)+np.mod(data.DepTime, 100)

y = data['ArrDelay'].values
names = ['Month', 'DayofMonth', 'DayOfWeek', 'plane_age', 'AirTime', 'Distance', 'ArrTime', 'DepTime']
X = data[names].values

sio.savemat('airline_data.mat', {'X':X, 'y':y, 'names':names})