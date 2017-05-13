# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model # logistic regression
from sklearn.metrics import accuracy_score # model evaluation
from scipy import misc # loading images
np.random.seed(1) # fixing the random value

path = '../data/arface/'

train_ids = np.arange(1, 26) # training set
test_ids = np.arange(26, 51) # test set
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21))) # select the view

# create the random projection matrix
D = 165*120 # original dimesion (165x120 pixel)
d = 500 # new dimesion that need to be reduced
ProjectionMatrix = np.random.randn(D, d) # generate the projection matrix (fat-matrix)

# listing of file names
def build_list_fn(pre, img_ids, view_ids):
  """
  Input:
    pred: 'M-' or 'W-'
    imgs_ids: indexes of miages
    view_ids: indexes of views
  Output:
    a listing of file names
  """
  filenames = []
  for i_id in img_ids:
    for v_id in view_ids:
      # filename: G-xxx-yy.bmp, ex: M-001-02.bmp
      name = path + pre + str(i_id).zfill(3) + '-' + str(v_id).zfill(2) + '.bmp'
      filenames.append(name)
  return filenames

# convert from rgb to grayscale
def rgb2gray(rgb):
  # Y' = 0.299 * R + 0.587 * G + 0.114 * B
  # https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
  return rgb[:, :, 0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# feature extraction
def vectorize_img(filename):
  # load image
  rgb = misc.imread(filename)
  # convert to gray scale
  gray = rgb2gray(rgb)
  misc.imsave(filename + '_gray.bmp', gray)
  # vetorization 
  im_vec = gray.reshape(1, D)
  misc.imsave(filename + '_gray_vec.bmp', im_vec)
  return im_vec

def build_data_matrix(img_ids, view_ids, t='train'):
  total_imgs = img_ids.shape[0]*view_ids.shape[0]*2 # total image * total view * 2 genders
  X_full = np.zeros((total_imgs, D))
  y = np.hstack((np.zeros((int(total_imgs/2), )), np.ones((int(total_imgs/2), )))) # half for male and half for female
  list_fn_m = build_list_fn('M-', img_ids, view_ids)
  list_fn_w = build_list_fn('W-', img_ids, view_ids)
  list_fn = list_fn_m + list_fn_w
  
  for i in range(len(list_fn)):
    X_full[i, :] = vectorize_img(list_fn[i])
    
  
  misc.imsave(path + 'X_' + t + '_full_.bmp', X_full)
    
  X = np.dot(X_full, ProjectionMatrix)
  
  misc.imsave(path + 'X' + t + '_full.bmp', X)
  return (X, y)
              
def feature_nomarlization(X, x_mean, x_var):
  return (X - x_mean)/x_var


(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
(X_test_full, y_test) = build_data_matrix(test_ids, view_ids, t='test')

x_mean = X_train_full.mean(axis = 0)
x_var = X_train_full.var(axis = 0)

X_train = feature_nomarlization(X_train_full, x_mean, x_var)
X_train_full = None # free this memo
X_test = feature_nomarlization(X_test_full, x_mean, x_var) # use x_mean and x_var from training set??
X_test_full = None # free this memo

misc.imsave(path + 'X_train.bmp', X_train)
misc.imsave(path + 'X_test.bmp', X_test)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#img = misc.imread(path + 'M-001-01.bmp')
#gray = rgb2gray(img)
#misc.imsave(path + 'M-001-01_gray.bmp', gray)
#x_mean = gray.mean(axis = 0)
#x_var = gray.var(axis = 0)
#gray1 = feature_nomarlization(gray, x_mean, x_var)
#misc.imsave(path + 'M-001-01_gray1.bmp', gray1)
#
#(X_train_full, y_train) = build_data_matrix(np.arange(1, 2), np.arange(1, 2))

def feature_extraction_fn(fn):
  """
  extract feature from filename
  """
  # vectorize
  im = vectorize_img(fn)
  # project
  im1 = np.dot(im, ProjectionMatrix)
  # standardization 
  return feature_nomarlization(im1, x_mean, x_var)

fn1 = path + 'M-036-18.bmp'
fn2 = path + 'W-045-01.bmp'
fn3 = path + 'M-048-01.bmp'
fn4 = path + 'W-027-02.bmp'

x1 = feature_extraction_fn(fn1)
p1 = logreg.predict_proba(x1)
print(p1)

x2 = feature_extraction_fn(fn2)
p2 = logreg.predict_proba(x2)
print(p2)

x3 = feature_extraction_fn(fn3)
p3 = logreg.predict_proba(x3)
print(p3)

x4 = feature_extraction_fn(fn4)
p4 = logreg.predict_proba(x4)
print(p4)
