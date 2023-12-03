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
        'size': 28,
        }

labels = []
attribs = []
 
with open('./xai/exampleAlebrijesLong.dat', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=' ')
    for r in plotting:
        labels.append(r[0])
        attribs.append(float(r[1]))
x = list(range(0, len(labels)))  
my_xticks = labels
  
num_subplots = 6
ig, ax = plt.subplots(num_subplots,figsize=(20,30),sharey=True)

lengthX = int(len(labels)/num_subplots)

for i in range(0,num_subplots-1):
    ax[i].set_ylim([-0.052, 0.07])
    ax[i].set_xlim([i*lengthX-1, (i+1)*lengthX])
    ax[i].set_ylabel('Attribution', fontdict=fontAx)
    ax[i].tick_params(axis="y", labelsize=30)
    ax[i].plot(x[i*lengthX:(i+1)*lengthX], attribs[i*lengthX:(i+1)*lengthX], linestyle="-") 
    ax[i].set_xticks(x[i*lengthX:(i+1)*lengthX], my_xticks[i*lengthX:(i+1)*lengthX], rotation='vertical', fontsize=26)
i+=1
ax[i].set_xlim([i*lengthX-1, len(labels)])
ax[num_subplots-1].tick_params(axis="y", labelsize=30)
ax[num_subplots-1].set_ylabel('Attribution', fontdict=fontAx)
ax[num_subplots-1].plot(x[i*lengthX:len(labels)], attribs[i*lengthX:len(labels)], linestyle="-") 
ax[num_subplots-1].set_xticks(x[i*lengthX:len(labels)], my_xticks[i*lengthX:len(labels)], rotation='vertical', fontsize=26)

#plt.title('    ', fontdict=fontTit)
plt.tight_layout(pad=0.3)
#plt.show()
plt.savefig('alebrijes.png')
plt.clf()

