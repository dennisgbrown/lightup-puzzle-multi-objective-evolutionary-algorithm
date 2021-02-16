# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

"""
Read a log file plot best vs. average with "error bars" to show min-max range
"""

fileroot = 'defaultSolution'

filename = 'data/' + fileroot + '.txt'

x = []
y = []
z = []
w = []

with open(filename, 'r') as reader:
   curr_line = reader.readline()
   while (curr_line):
        data_list = list(curr_line.split("\t"))
        x.append(float(data_list[0]))
        y.append(float(data_list[1]))
        z.append(float(data_list[2]))
        w.append(float(data_list[3]))
        curr_line = reader.readline()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

plt.title(fileroot)
plt.xlabel('Lit cells')
plt.ylabel('Bulb cell violations')
ax.set_zlabel('Black cell violations')
# plt.legend(loc='lower right')

plt.show()

fig.savefig('plots/' + fileroot + '.png', dpi = 600)



