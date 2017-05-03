import csv as csv
import numpy as np
import iso8601
from sklearn.cluster import KMeans
from operator import itemgetter
from collections import OrderedDict
# from geopy.geocoders import Nominatim
# geolocator = Nominatim()

list_of_lists = []
qualified_entry = []
training_data = []
dev_data = []
test_data = []
places = {}
top_10_places = {}

# Read file by line and save into a list
# Break each line into fields
# Make a list of lists (a list of checkins each of which is a list of field values)
# Convert the fields into appropriate formats

# cleaned_file_object.writerow(["UserId", "Check-in Time", "Latitude", "Longitude", "PlaceID"])
with open("loc-brightkite_totalCheckins.txt") as f:
  for line in f:
    inner_list = [elt.strip() for elt in line.split('\t')]
    if int(inner_list[0]) == 7 and (float(inner_list[2]) != 0 and float(inner_list[3]) != 0):  # get the right user and make sure coordinates are not null

      if inner_list[4] in places:
        places[inner_list[4]] += 1
      else:
        places[inner_list[4]] = 1

  # Reads text until the end of the block:
  for line in f:  # This keeps reading the file
    inner_list = [elt.strip() for elt in line.split('\t')]
    if int(inner_list[0]) != 7:  # Or whatever test is needed
      break

places = OrderedDict(sorted(places.items(), key=itemgetter(1), reverse=True))
  
i = 0
while i < 10:
  top_10_places[places.keys()[i]] = places.values()[i]
  i += 1

print places
print top_10_places

with open("loc-brightkite_totalCheckins.txt") as f:
  for line in f:
    inner_list = [elt.strip() for elt in line.split('\t')]
    if int(inner_list[0]) == 7 and (float(inner_list[2]) != 0 and float(inner_list[3]) != 0): 
      d = iso8601.parse_date(inner_list[1])
      day_of_week = d.weekday()
      check_in_frequency = places[inner_list[4]]

      qualified_entry = [float(inner_list[2]), float(inner_list[3]), d.hour + d.minute / 60. + d.second / 3600., day_of_week, inner_list[4], check_in_frequency]
      list_of_lists.append(qualified_entry)

  # Reads text until the end of the block:
  for line in f:  # This keeps reading the file
    inner_list = [elt.strip() for elt in line.split('\t')]
    if int(inner_list[0]) != 7:  # Or whatever test is needed
      break

#print list_of_lists

# Plot 3D graph for data visualization
# X = np.array(list_of_lists)
# import pylab as p
# from matplotlib.pyplot import *
# import mpl_toolkits.mplot3d.axes3d as p3

# fig=p.figure()
# ax = p3.Axes3D(fig)
# # plot3D requires a 1D array for x, y, and z
# # ravel() converts the 100x100 array into a 1x10000 array
# fig.suptitle('User #7: All check-in places', fontsize=20)
# ax.scatter3D(X[:,0],X[:,1],X[:,2])
# ax.set_xlabel('lat')
# ax.set_ylabel('lon')
# ax.set_zlabel('hour of the day')
# fig.add_axes(ax)
# p.show()

training_end_index = int(len(list_of_lists) * 0.7)
dev_end_index = int(len(list_of_lists) * 0.85)

# print training_end_index
# print dev_end_index

for i in range(0, training_end_index):
  training_data.append(list_of_lists[i])

for i in range(training_end_index, dev_end_index):
  dev_data.append(list_of_lists[i])

for i in range(dev_end_index, len(list_of_lists)):
  test_data.append(list_of_lists[i])

# print training_data
# print dev_data
# print test_data

from sklearn import cluster
from sklearn import metrics

def run_k_means_and_get_metrics(data, number_of_clusters):
  X = np.array(data)
  # only take the coordinates fields
  X = X[:,[0,1]]
  k = number_of_clusters
  kmeans = cluster.KMeans(n_clusters=k)
  kmeans_model = kmeans.fit(X)

  labels = kmeans_model.labels_
  centroids = kmeans.cluster_centers_

  # print labels
  # print centroids

  score1 = metrics.silhouette_score(X, labels, metric='euclidean')
  print "The silhouette score with euclidean distance and " + str(number_of_clusters) + " clusters is " + str(score1)
  score2 = metrics.silhouette_score(X, labels, metric='cityblock')
  print "The silhouette score with cityblock distance and " + str(number_of_clusters) + " clusters is " + str(score2)

run_k_means_and_get_metrics(training_data, 2)
run_k_means_and_get_metrics(training_data, 3)

# Get dev data excluding the weekend checkins
dev_data_no_weekends = []
for checkin in dev_data:
  if checkin[3] != 5 and checkin[3] != 6:
    dev_data_no_weekends.append(checkin)
# 205 entries

# Get dev data excluding the ones not in the top 10 places the user checks in at
dev_data_only_top_10_places = []
for checkin in dev_data:
  if checkin[4] in top_10_places.keys():
    dev_data_only_top_10_places.append(checkin)
# 137 entries

# Get dev data combining the exclusion of the above two
dev_data_no_weekends_and_only_top_10_places = []
for checkin in dev_data:
  if checkin[3] != 5 and checkin[3] != 6:
    if checkin[4] in top_10_places.keys():
      dev_data_no_weekends_and_only_top_10_places.append(checkin)
# 119 entries

run_k_means_and_get_metrics(dev_data_no_weekends, 2)
run_k_means_and_get_metrics(dev_data_only_top_10_places, 2)
run_k_means_and_get_metrics(dev_data_no_weekends_and_only_top_10_places, 2)
run_k_means_and_get_metrics(dev_data_no_weekends, 3)
run_k_means_and_get_metrics(dev_data_only_top_10_places, 3)
run_k_means_and_get_metrics(dev_data_no_weekends_and_only_top_10_places, 3)

test_data_only_top_10_places = []
for checkin in test_data:
  if checkin[4] in top_10_places.keys():
    test_data_only_top_10_places.append(checkin)

print len(test_data_only_top_10_places)
run_k_means_and_get_metrics(test_data_only_top_10_places, 3)


