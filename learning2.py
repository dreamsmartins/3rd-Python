import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.Series([1, 2, 3, 4, 5])
y = pd.Series([5, 4, 3, 2, 1])
z = pd.Series([1, 2, 3, 4, 5])

df = pd.DataFrame({'x': x, 'y': y, 'z': z})
print(df)
#df['x'] = df['x'] + 1