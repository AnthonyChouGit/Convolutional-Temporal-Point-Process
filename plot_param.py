import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]) - 0.5
y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])-0.5
z = 4.5

dx = dy = 1
dz = np.flip(np.array([4.6337, 4.5960, 4.5801,
      4.6508, 4.6086, 4.6106,
      4.6536, 4.6311, 4.5883,
      4.6822, 4.6102, 4.6625,
      4.6764, 4.6490, 4.6913])-z)
color_list = ['blue', 'orange', 'green', 'grey', 'red']
colors = list()
for i in range(5):
    for j in range(3):
        colors.append(color_list[i])
ax.bar3d(x, y, z, dx, dy, dz, shade=True, edgecolor='black', color=colors)
ax.set_xlabel('horizon')
ax.set_ylabel('# of channels')
ax.set_zlabel('NLL')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3])
ax.set_xticklabels([10, 5, 3, 2, 1])
ax.set_yticklabels([3, 2, 1])
ax.set_zlim((4.5, 4.7))

plt.savefig('temp2.jpg', bbox_inches='tight')