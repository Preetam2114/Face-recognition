from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import random
from face_features import *


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

images_pre_class=50


# loop over the training data sub-folders

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name


    k = 1
    # loop over the images in each sub-folder
    for x in range(1,images_pre_class+1):
        a=random.choice(os.listdir(dir))
        file2 = dir+'/'+a
        # read the image and resize it to a fixed-size
        image = cv2.imread(file2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, fixed_size)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_color = image[y:y+h, x:x+w]
            # cv2.imshow(roi_color)
            # Global Feature extraction
            fv_hu_moments = fd_hu_moments(roi_color)
            fv_haralick   = fd_haralick(roi_color)
            fv_histogram  = fd_histogram(roi_color)

            # Concatenate global features
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

                # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)
                #print global_feature
            i += 1
            k += 1
        # else:
        #     image = cv2.resize(image, fixed_size)
        #     fv_hu_moments = fd_hu_moments(image)
        #     fv_haralick   = fd_haralick(image)
        #     fv_histogram  = fd_histogram(image)

        #     # Concatenate global features
        #     global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        #         # update the list of labels and feature vectors
        #     labels.append(current_label)
        #     global_features.append(global_feature)
        #         #print global_feature
        #     i += 1
        #     k += 1

    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

print ("[STATUS] completed Feature Extraction...")


#time
# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

print ("[STATUS] target labels: {}".format(target))
print ("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print ("[STATUS] end of training..")

