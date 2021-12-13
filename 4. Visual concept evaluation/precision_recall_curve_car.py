#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[3]:


counting_method = False
counting_method = True # <--

# Set to True if running on Google Colab
google_colab = True
# google_colab = True # <--

calculate_mode = False
# calculate_mode = True # <--

save_result = False
# save_result = True # <--
    
if calculate_mode == False:
    save_result = False


# In[5]:


import os
import sys
import time
import csv
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

if google_colab == True:
    # Make sure to mount drive and that you have access to IDL_group_project. All paths are from the shared folder
    from google.colab import drive
    drive._mount('/content/drive')


# In[6]:


#This function will return both the path and the file names of all files in a directory
def path_file_list_creator(path, both_path_file = False):
  path_file_list = []
  for myfile in os.listdir(path):
    full_path = os.path.join(path,myfile)
    if both_path_file == True:
      path_file_list.append((full_path, myfile))
    else:
      path_file_list.append(full_path)
  return(path_file_list,len(path_file_list))

#Concate all different classes semantic information together from seperate directories
def concate_directories(paths):
  semantic_bundles_paths_extended = []
  total_files = 0 

  for path in paths:
    class_bundle_paths , num_files = path_file_list_creator(path)
    semantic_bundles_paths_extended.extend(class_bundle_paths)
    total_files += num_files
  print("Total Files grabbed ", total_files)
  return(semantic_bundles_paths_extended)


# # Precision-Recall curve

# Load visual concepts

# In[7]:


if google_colab == True:
  visual_concepts = np.load('/content/drive/MyDrive/IDL_group_project/visual_concepts_positive.npy')
else:
  visual_concepts = np.load('/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/visual_concepts_positive_car.npy')
  # visual_concepts = np.load('/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/visual_concepts_positive.npy')

num_visual_concepts = visual_concepts.shape[0]
print(visual_concepts.shape)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(visual_concepts[:, 0], visual_concepts[:, 1])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(visual_concepts[:, 10], visual_concepts[:, 11], visual_concepts[:, 12])
# plt.show()

# # Check the norm of the visual concepts
# for visual_concept in visual_concepts:
#   print(np.linalg.norm(visual_concept))

# # Normalize the visual concepts
# for i in range(len(visual_concepts)):
#   visual_concepts[i] = visual_concepts[i]/np.linalg.norm(visual_concepts[i])


# Load feature vectors

# In[8]:


from numpy import linalg as LA

if google_colab == True:
    fp_positive_dir = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/fp_positive/'
    fp_negative_dir = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/fp_negative/'
else:
    # fp_positive_dir = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/fp_positive/'
    # fp_negative_dir = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/fp_negative/'
    fp_positive_dir = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/fp_positive_car/'
    fp_negative_dir = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/fp_negative_car/'


# In[9]:


# fp_positive_npy_paths = concate_directories(all_classes_fp_positive_dir)
fp_positive_npy_paths = concate_directories([fp_positive_dir])
fp_negative_npy_paths = concate_directories([fp_negative_dir])

x = np.load(fp_positive_npy_paths[0])
y = np.load(fp_negative_npy_paths[0])
print(LA.norm(x - y).item())

# Shuffle the feature vector sets the same way
import random
fp_all_npy_paths = list(zip(fp_positive_npy_paths, fp_negative_npy_paths))
random.shuffle(fp_all_npy_paths)
fp_positive_npy_paths, fp_negative_npy_paths = zip(*fp_all_npy_paths)

print(fp_positive_npy_paths[:3])
print(fp_negative_npy_paths[:3])

# Take a subset to reduce the computation time
dataset_cutoff = len(fp_positive_npy_paths)
fp_positive_npy_paths = fp_positive_npy_paths[:dataset_cutoff]
fp_negative_npy_paths = fp_negative_npy_paths[:dataset_cutoff]
print(len(fp_negative_npy_paths))


# In[14]:


fp_positive_npy_paths[0]


# In[13]:


print(len(fp_positive_npy_paths[2]))
print(len(fp_negative_npy_paths[2]))


# In[17]:


np.load(np.load(fp_positive_npy_paths[0]))


# Create a set of thresholds

# In[ ]:


thresholds = np.linspace(0.5, 1.5, 101, endpoint=True)
num_thresholds = len(thresholds)
print(thresholds)


# 
# Calculate the norm, compare to the thresholds

# In[ ]:


if counting_method == True:
  feature_vector_cutoff = 200
else:
  feature_vector_cutoff = 200

def check_firing(fp_npy_paths, visual_concepts, thresholds, type):
  num_visual_concepts = visual_concepts.shape[0]
  num_thresholds = len(thresholds)
    
  firing_result_list = []
  firing_result_array = np.zeros(shape=(num_thresholds, num_visual_concepts), dtype=np.int32) # num_T * 200 elements

  if counting_method == True: # Count all the fires
    for i in tqdm(range(len(fp_npy_paths))):
      feature_vectors = np.load(fp_npy_paths[i])
      np.random.shuffle(feature_vectors)
      for feature_vector in feature_vectors[:feature_vector_cutoff]:
        for j in range(num_visual_concepts):
          norm = LA.norm(feature_vector - visual_concepts[j]).item()
          for k in range(num_thresholds):
            if norm < thresholds[k]:
              firing_result_array[k, j] += 1
  else: # Assign 1 for any fire
    for i in tqdm(range(len(fp_npy_paths))):
      feature_vectors = np.load(fp_npy_paths[i])
      np.random.shuffle(feature_vectors)
      for feature_vector in feature_vectors[:feature_vector_cutoff]:
        for j in range(num_visual_concepts):
          if np.any(firing_result_array[:, j] == 0):
            norm = LA.norm(feature_vector - visual_concepts[j]).item()
            for k in range(num_thresholds):
              if norm < thresholds[k]:
                firing_result_array[k, j] = 1
                
  for i in range(num_thresholds):
    firing_result_dict = {'T': thresholds[i], type: firing_result_array[i]}
    firing_result_list.append(firing_result_dict)
    
  plt.imshow(firing_result_array, interpolation='none')
  plt.xlabel('Visual concept')
  plt.ylabel('T')
  plt.show()
    
  return firing_result_list

if calculate_mode == True:
  print('Positive')
  positive_firing_result_list = check_firing(fp_positive_npy_paths, visual_concepts, thresholds, type='positive')
  print('Negative')
  negative_firing_result_list = check_firing(fp_negative_npy_paths, visual_concepts, thresholds, type='negative')


# Save and load the results

# In[ ]:


if counting_method == True:
    if google_colab == True:
        positive_firing_result_path = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/positive_counting.pkl'
        negative_firing_result_path = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/negative_counting.pkl'
    else:
        # positive_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/positive_counting.pkl'
        # negative_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/negative_counting.pkl'
        positive_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/positive_counting_car.pkl'
        negative_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/negative_counting_car.pkl'
else:
    if google_colab == True:
        positive_firing_result_path = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/positive_binary.pkl'
        negative_firing_result_path = '/content/drive/MyDrive/IDL_group_project/Google_Colab/data/negative_binary.pkl'
    else:
        # positive_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/positive_binary.pkl'
        # negative_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/negative_binary.pkl'
        positive_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/positive_binary_car.pkl'
        negative_firing_result_path = '/home/tunx404/Cloud/Google Drive - CMU - Shared with me/IDL_group_project/Google_Colab/data/negative_binary_car.pkl'

if save_result == True:
    pkl_file = open(positive_firing_result_path, 'wb')
    pickle.dump(positive_firing_result_list, pkl_file)
    pkl_file.close()

    pkl_file = open(negative_firing_result_path, 'wb')
    pickle.dump(negative_firing_result_list, pkl_file)
    pkl_file.close()

if calculate_mode == False:
    with open(positive_firing_result_path, 'rb') as f:
        positive_firing_result_list = pickle.load(f)
    with open(negative_firing_result_path, 'rb') as f:
        negative_firing_result_list = pickle.load(f)
    dataset_cutoff = len(fp_positive_npy_paths) # If reading saved result, set the dataset_cutoff to the same value when saving


# In[ ]:


def get_result_array(firing_result_list, type):
    firing_result_array = []
    for firing_result in firing_result_list:
        firing_result_array.append(np.expand_dims(firing_result[type], axis=0))
    firing_result_array = np.concatenate(firing_result_array, axis=0)

    plt.imshow(firing_result_array, interpolation='none')
    plt.xlabel('Visual concept')
    plt.ylabel('T index')
    plt.show()
    
    return firing_result_array
    
print('Positive')
positive_firing_result_array = get_result_array(positive_firing_result_list, 'positive')
print('Negative')
negative_firing_result_array = get_result_array(negative_firing_result_list, 'negative')


# In[ ]:


NUM_SELECTED_VISUAL_CONCEPTS = 50
positive_firing_sum = np.sum(positive_firing_result_array, axis=0)
negative_firing_sum = np.sum(negative_firing_result_array, axis=0)
difference = positive_firing_sum - negative_firing_sum
best_indices = np.argsort(-difference, axis=0)[:NUM_SELECTED_VISUAL_CONCEPTS]
best_indices = list(best_indices)
best_indices.sort()

print(best_indices)
# print(positive_firing_sum)
# print(negative_firing_sum)
# print(difference)
# for index in best_indices:
#     print(difference[index])


# In[ ]:


def find_first_fire_indices(firing_result_array):
    SELECT_THRESHOLD = 1000 if counting_method == True else 0
    num_visual_concepts = firing_result_array.shape[1]
    first_fire_indices = []
    for firing_result in firing_result_array:
        for i in range(num_visual_concepts):
            if i not in first_fire_indices:
                if firing_result[i] > SELECT_THRESHOLD:
                    first_fire_indices.append(i)
                    if len(first_fire_indices) >= NUM_SELECTED_VISUAL_CONCEPTS:
                        return first_fire_indices
        if len(first_fire_indices) >= NUM_SELECTED_VISUAL_CONCEPTS:
            return first_fire_indices
        
positive_first_fire_indices = find_first_fire_indices(positive_firing_result_array)
positive_first_fire_indices.sort()
        
negative_first_fire_indices = find_first_fire_indices(negative_firing_result_array)
negative_first_fire_indices.sort()

for index in positive_first_fire_indices:
    if index in negative_first_fire_indices:
        positive_first_fire_indices.remove(index)

print(positive_first_fire_indices)
print(negative_first_fire_indices)


# In[ ]:


# def find_most_fire_indices(firing_result_array):
#     num_visual_concepts = firing_result_array.shape[1]
#     count = np.sum(firing_result_array, axis=0)
#     indices = [i for i in range(num_visual_concepts)]
#     fig = plt.figure()
#     ax = fig.add_axes([0,0,1,1])
#     ax.bar(indices, count)
#     most_fire_indices = np.argsort(-count, axis=0)[:NUM_SELECTED_VISUAL_CONCEPTS]
#     return list(most_fire_indices)
        
# positive_most_fire_indices = find_most_fire_indices(positive_firing_result_array)
# positive_most_fire_indices.sort()
        
# negative_most_fire_indices = find_most_fire_indices(negative_firing_result_array)
# negative_most_fire_indices.sort()

# for index in positive_most_fire_indices:
#     if index in negative_most_fire_indices:
#         positive_most_fire_indices.remove(index)

# print(positive_most_fire_indices)
# print(negative_most_fire_indices)


# Calculate TP FP FN TN, draw the curves

# In[ ]:


num_visual_concepts_new = num_visual_concepts
# selected_indices = [i for i in range(num_visual_concepts)]
# selected_indices = positive_most_fire_indices
# selected_indices = positive_first_fire_indices
# selected_indices = [0, 1, 6, 26, 86, 119, 126, 131, 142, 145]
selected_indices = best_indices

precisions = []
recalls = []
TPRs = []
FPRs = []
TPs = []
FPs = []
FNs = []
TNs = []

num_visual_concepts_new = len(selected_indices)
for i in range(num_thresholds):
    threshold = positive_firing_result_list[i]['T']

    true_positive  = sum(positive_firing_result_list[i]['positive'][selected_indices])
    false_positive = sum(negative_firing_result_list[i]['negative'][selected_indices])

    if counting_method == True:
        false_negative = dataset_cutoff*feature_vector_cutoff*num_visual_concepts_new - true_positive  # The number of fp+ not firing
        true_negative  = dataset_cutoff*feature_vector_cutoff*num_visual_concepts_new - false_positive # The number of fp- not firing
    else:
        false_negative = num_visual_concepts_new - true_positive
        true_negative  = num_visual_concepts_new - false_positive

    precision = 1.0*true_positive/(true_positive + false_positive + 1e-6)
    if true_positive == 0:
        precision = 1
    recall = 1.0*true_positive/(true_positive + false_negative + 1e-6)
    TPR = 1.0*true_positive/(true_positive + false_negative + 1e-6)
    FPR = 1.0*false_positive/(false_positive + true_negative + 1e-6)

    precisions.append(precision)
    recalls.append(recall)
    TPRs.append(TPR)
    FPRs.append(FPR)
    TPs.append(true_positive)
    FPs.append(false_positive)
    FNs.append(false_negative)
    TNs.append(true_negative)

plt.plot(recalls, precisions)
plt.title('Precision-Recall, AP = ' + str(round(np.average(precisions[16:90]), 2)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(FPRs, TPRs)

x = np.linspace(0, 1, 100)
plt.plot(x, x);
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print('Index \tThreshold \tPrecision \tRecall \t\tTPR \t\tFPR \t\tTP \tFP \tFN \tFN')
for i in range(num_thresholds):
    print(f'{i}\t{thresholds[i]:0.6f}\t{precisions[i]:0.6f}\t{recalls[i]:0.6f}\t{TPRs[i]:0.6f}\t{FPRs[i]:0.6f}\t{TPs[i]:d}\t{FPs[i]:d}\t{FNs[i]:d}\t{TNs[i]:d}')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Indices = ' + str(selected_indices))
ax1.imshow(positive_firing_result_array[:, selected_indices], interpolation='none')
ax2.imshow(negative_firing_result_array[:, selected_indices], interpolation='none')
ax1.plot(0, 48, len(selected_indices), 48, color='r', marker='o')
ax2.plot(0, 48, len(selected_indices), 48, color='r', marker='o')
ax1.set(xlabel='Visual concept', ylabel='T index')
ax2.set(xlabel='Visual concept', ylabel='T index')


# In[ ]:





# In[ ]:




