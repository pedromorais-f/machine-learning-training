#!/usr/bin/env python
# coding: utf-8

# In[8]:


import math
import numpy


# In[6]:


def sigmoid_function_math(value):
    sigmoid = 1 / (1 + math.exp(-value))
    
    return sigmoid   


# In[ ]:


def sigmoid_function(value):
    sigmoid = 1 / (1 + np.exp(-value))
    
    return sigmoid


# In[ ]:


def sigmoid_gradient(value):
    sigmoid = sigmoid_function(value)
    ds = sigmoid * (1 - sigmoid)
    
    return ds


# In[ ]:


def image2vector(image):
    vector = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    
    return vector


# In[ ]:


def normalize_rows(x):
    normalization = np.linalg.norm(x, axis = 1, keepdims = True)

    return x / normalization


# In[ ]:


def softmax(x):
    matriz_e = np.exp(x)
    row_sum = np.sum(matriz_e,axis = 1,keepdims = True)
    
    return matriz_e / row_sum


# In[ ]:


def loss_l1(y, y_hat):
    return sum(abs(y_hat - y))


# In[ ]:


def loss_l2(y, y_hat):
    result = y_hat - y
    
    return np.dot(result, result)

