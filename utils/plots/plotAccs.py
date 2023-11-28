import matplotlib.pyplot as plt 
import numpy as np 
import csv

plt.rcParams.update({
 "axes.linewidth":2,
 "xtick.major.width":2,
 "xtick.minor.width":2,
 "ytick.major.width":2,
 "ytick.minor.width":2,
 "xtick.major.size":8,
 "ytick.major.size":8,
 "xtick.minor.size":6,
 "ytick.minor.size":6
})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 20
})
fontTit = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'heavy',
        'size': 22,
        }
fontAx = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 18,
        }

x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
 
with open('./accs/modelb2a8fixV100.acc.seed3', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for r in plotting:
        x1.append(int(r[0])/1000)
        y1.append(float(r[1])*100)
with open('./accs/modelb2a8sentence2V100.acc.seed3', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for r in plotting:
        x2.append(int(r[0])/1000)
        y2.append(float(r[1])*100)
with open('./accs/modelb2a8sentence3V100.acc.seed3', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for r in plotting:
        x3.append(int(r[0])/1000)
        y3.append(float(r[1])*100)
with open('./accs/modelb2a8sentence6A80.acc.seed3', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for r in plotting:
        x4.append(int(r[0])/1000)
        y4.append(float(r[1])*100)
with open('./accs/modelb2a8sentence16A80.acc.seed2', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')
    for r in plotting:
        x5.append(int(r[0])/1000)
        y5.append(float(r[1])*100)
  
  
ig, ax = plt.subplots()
ax.set_ylim(75,96)
ax.set_xlim(0,2060)
  
ax.plot(x1, y1, label = "~1 split", linestyle="-") 
ax.plot(x2, y2, label = "~2 splits", linestyle="--") 
ax.plot(x3, y3, label = "~3 splits", linestyle="-.") 
ax.plot(x4, y4, label = "~6 splits", linestyle=":") 
ax.plot(x5, y5, label = "16 splits", linestyle="--", dashes=(5,6)) 

plt.xlabel('Training steps (x1000)', fontdict=fontAx)
plt.ylabel('Accuracy (\%)', fontdict=fontAx)
#plt.title('    ', fontdict=fontTit)
plt.tight_layout(pad=0.03)
plt.legend(loc='lower right', frameon=False, fontsize="17") 
#plt.show()
plt.savefig('trainingEvolution.png')
plt.clf()

