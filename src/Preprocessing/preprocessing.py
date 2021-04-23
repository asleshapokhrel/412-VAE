import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle


def load_data(object_name, data_location):
    """
    Return Nx784 numpy array corresponding to N drawings of size 28x28.
    object_name (str) : Name of object drawing to be loaded to the memory. 
                        E.g., "airplane", "mouse"
    """
    savename = data_location + '{}.npy'.format(object_name)
    # mmap_mode='r+' this does not load everything to memory to avoid
    # out of memory issues
    data = np.load(savename, mmap_mode='r+')

    return data


def load_mult_data(object_arr, data_location):
    """
    Return len(object_arr)xNx784 numpy array corresponding to N drawings of size 28x28.
    object_arr (str) : e.g., ["airplane", "mouse"]
    """
    data_arr = []
    for obj in object_arr:
        data = load_data(obj, data_location)
        data_arr.append(data)
    return data_arr

"""
def display_img(im_arr, img_category):
Plot im_arr.
im_arr (numpy uint8 array): Numpy array of size (784,) to be visualized.
img_category (str): name of object in im_arr
plt.imshow(im_arr.reshape(28,28), cmap='gray')
plt.title(img_category)
"""

def create_dataset(data_arr, obj_name_arr, num_samples):
    """
    """
    _, col_size = (data_arr[0]).shape
    # empty array to concatenate data
    final_arr = np.array([], dtype=np.uint8).reshape(0, col_size)
    label_arr = []

    for i in range(len(data_arr)):
        # choose num_samples randomly of each object
        index = np.random.choice(data_arr[i].shape[0], num_samples, replace=False)
        obj_data = data_arr[i][index]
        final_arr = np.concatenate((final_arr, obj_data), axis=0)   
        # create num_sample labels and concatenate to label_arr
        obj_label = [obj_name_arr[i]]*num_samples
        label_arr.extend(obj_label)  

    return final_arr, np.array(label_arr)



