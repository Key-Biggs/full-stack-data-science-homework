##Numpy Assignment##
import pandas as pd
import numpy as np

house_feat1 = ['breakfast nook', 'hall', 'living room', 'master bedroom', 'attic',
               'basement', 'garage', 'bathroom', 'guest house', 'laundry room']
house_feat2 = [45, 32, 85, 55, 250, 10000, 5000, 150, 1200, 80]
house_feat3 =[3000, 1200, 6000, 7500, 1440, 2022, 4404, 5150, 2001, 1946]

array1 = np.array(house_feat1)
array2 = np.array(house_feat2)
array3 = np.array(house_feat3)

three_d = np.stack((array1, array2, array3), axis=0)
print(three_d)
print(three_d.shape)

import pandas as pd
import numpy as np

quakes = pd.read_csv("earthquakes1970-2014.csv")
print(quakes)
array = np.array(quakes)
print(array)
print(quakes.columns)
print(quakes.iloc[:21, [3, 5, 6, 7, 11]])
quakes.loc[0] = 1



mask = quakes['mag'] > 4.5
quakes_gt_45 = quakes[mask]

print(quakes_gt_45)
array = np.array(quakes_gt_45)
print(array)
print(quakes_gt_45.iloc[:21, [3, 5, 6, 7, 11]])




quakes = pd.read_csv("earthquakes1970-2014.csv")
mask = quakes[:] > 4.5
quakes_gt_45 = quakes[mask]

means = quakes_gt_45.mean()
stds = quakes_gt_45.std()

print("Means:\n", means)
print("Standard deviations:\n", stds)

array1 = np.array(quakes_gt_45)


import numpy as np

array1 = np.array("quakes_gt_45")


mean = np.mean(array, axis=0)
std = np.std(array, axis=0)
print("Mean of each column:\n", mean)
print("Standard deviation of each column:\n", std)

array1 -= np.array([1, 25, 25, 10, 4])
print("Array after subtraction:\n", array1)