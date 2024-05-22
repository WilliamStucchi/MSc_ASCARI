import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

data = np.loadtxt('inputs/trainingdata/test_set_1.csv', delimiter=',')

x = data[:, 3]
y = data[:, 4]

plt.figure(figsize=(25, 10))
ax = plt.gca()

ax.yaxis.set_major_locator(MultipleLocator(2.5))

plt.plot(x, label='Ax', color='tab:orange')
plt.plot(y, label='Ay', color='tab:blue')


plt.ylabel('test')
plt.xlabel('Time steps (10 ms)')
plt.legend()
plt.grid()
plt.show()
plt.close()
