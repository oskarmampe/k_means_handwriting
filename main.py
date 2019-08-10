import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

# print(digits.DESCR)
# print(digits.data)
# print(digits.target)

#plt.gray()

#plt.matshow(digits.images[100])

#plt.show()

# no_clusters = list(range(1, 50))
# inertias = []
# for i in no_clusters:
#   print('Calculating with {} no of clusters....'.format(i))
#   classifier = KMeans(n_clusters=i)
#   classifier.fit(digits.data)
#   inertias.append(classifier.inertia_)

# plt.plot(no_clusters, inertias, '-o')

# plt.xticks(no_clusters)

classifier = KMeans(n_clusters=10)
classifier.fit(digits.data)

new_labels = classifier.predict(digits.data)

sample = np.array(digits.data[100]).reshape(1, -1)
prediction = classifier.predict(sample)

print(prediction)
print(digits.target[100])

df = pd.DataFrame({'label': new_labels, 'cluster': digits.target})
ct = pd.crosstab(df['label'], df['cluster'])
print(ct)


fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1+i)
  ax.imshow(classifier.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()
