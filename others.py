import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import statistics
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import umap
import os
from sklearn.preprocessing import StandardScaler
import random
import umap
import time
data = pd.read_csv("data/purepbmc.csv")
trainx=data.sample(frac=1)
Y = trainx.to_numpy()
newX = Y[:,1:]
label = Y[:,0]
start = time.time()
sfid=np.random.permutation(len(newX))
X_sf=newX.T[:,sfid].T
shuffledlabel = label[sfid]
checkx = np.asarray(X_sf).astype('float32')
log10check = np.log10(checkx+1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(log10check)
path_save = 'finalmixedpbmc/'
if not os.path.isdir(path_save):
    os.makedirs(path_save)
reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_data)
new = np.zeros((92043,3))
new[:,1:] = embedding
new[:,0] = shuffledlabel
np.save(path_save+'/umap.npy', new)
num_list = random.sample(range(92043), 5000)
umapresult = embedding[num_list,:]
print(embedding.shape)
label1 = shuffledlabel[num_list]
path_saved = 'journaldrawing'
if not os.path.isdir(path_saved):
    os.makedirs(path_saved)
cdict = {1: 'red', 2: 'blue', 3: 'gray', 4: 'green', 5: 'cyan', 6: 'yellow', 7: 'pink', 8: 'black', 9: 'brown',
         10: 'darkgreen', 11: 'magenta'}
fig, ax = plt.subplots()
# ax = plt.axes(projection='3d')
for g in np.unique(label1):
  # if g==1 or g == 6 :
  #  continue
  ix = np.where(label1 == g)
  ax.scatter(umapresult[ix, 0], umapresult[ix, 1], c=cdict[g], label=g, alpha=0.5, s=15)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.set_size_inches(10, 8)
plt.show()
plt.savefig(join(path_saved, 'umapmixed.eps'), format='eps', dpi=1000)