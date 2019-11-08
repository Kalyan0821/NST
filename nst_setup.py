# Import required libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19  # To use the pre-trained weights from this model

mean_RGB = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # These were the RGB mean values over the ImageNet dataset that VGG19 was trained on
IM_SHAPE = (500, 500)  # All images will be reshaped to 500 X 500 X 3
num_iterations = 100  # Number of optimization steps

tf.reset_default_graph()
# Get the VGG19 pre-trained Keras model. This will be used to obtain content and style representations 
cnn = VGG19(include_top=False, weights='imagenet', input_shape=IM_SHAPE+(3,))   

def setup_image(raw_image, target_shape=IM_SHAPE):
    """Input: (a X b X 3) image 
      Main Output: RGB mean-centered np.array of size (1 X 500 X 500 X 3)"""
    resized_raw = raw_image.resize(target_shape)
    resized_raw = np.reshape(resized_raw, (1,)+target_shape+(3,))
    image = resized_raw - mean_RGB
    return image, resized_raw

def presentable(image):
    """Input: (1 X 500 X 500 X 3) np.array, assumed to be RGB mean-centered
      Output: (500 X 500 X 3) image with RGB mean values re-added"""
    image1 = image + mean_RGB
    G = np.clip(image1[0], 0, 255).astype('uint8')
    return G

def pretrained_weights(layer_name):
    """Input: String specifying a layer name of the VGG19 Keras model
      Output: tf weight tensors for that layer"""
    weights = cnn.get_layer(layer_name).get_weights()
    W = weights[0]
    b_1D = weights[1]
    b = np.reshape(b_1D, (1, 1, 1) + b_1D.shape)
    # Define the weights as constants because here we only use the network to get representations
    W = tf.constant(W, dtype=tf.float64)
    b = tf.constant(b, dtype=tf.float64)
    return W, b

def tf_conv_relu(prev_layer, layer_name):
    """Performs convolution for the layer specified by layer_name"""
    W, b = pretrained_weights(layer_name)
    Z = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME') + b
    return tf.nn.relu(Z, name=layer_name)

def tf_avg_pool(prev_layer, layer_name):
    """Performs average-pooling. This is in accordance with the original paper on Neural Style Transfer by
      Gatys et al."""
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=layer_name)

def tf_cnn(X, mode):
    """Recreation of the original VGG 19 model, with max-pooling replaced by average-pooling. 
       The output of this function is either a tensor, a list of tensors or a dictionary of tensors, depending
       on the mode with which it is called"""
    block1_conv1 = tf_conv_relu(X, 'block1_conv1')
    block1_conv2 = tf_conv_relu(block1_conv1, 'block1_conv2')
    block1_pool = tf_avg_pool(block1_conv2, 'block1_pool')
    
    block2_conv1 = tf_conv_relu(block1_pool, 'block2_conv1')
    block2_conv2 = tf_conv_relu(block2_conv1, 'block2_conv2')
    block2_pool = tf_avg_pool(block2_conv2, 'block2_pool')
    
    block3_conv1 = tf_conv_relu(block2_pool, 'block3_conv1')
    block3_conv2 = tf_conv_relu(block3_conv1, 'block3_conv2')
    block3_conv3 = tf_conv_relu(block3_conv2, 'block3_conv3')
    block3_conv4 = tf_conv_relu(block3_conv3, 'block3_conv4')
    block3_pool = tf_avg_pool(block3_conv4, 'block3_pool')
    
    block4_conv1 = tf_conv_relu(block3_pool, 'block4_conv1')
    block4_conv2 = tf_conv_relu(block4_conv1, 'block4_conv2')
    block4_conv3 = tf_conv_relu(block4_conv2, 'block4_conv3')
    block4_conv4 = tf_conv_relu(block4_conv3, 'block4_conv4')
    block4_pool = tf_avg_pool(block4_conv4, 'block4_pool')
    
    block5_conv1 = tf_conv_relu(block4_pool, 'block5_conv1')
    block5_conv2 = tf_conv_relu(block5_conv1, 'block5_conv2')
    block5_conv3 = tf_conv_relu(block5_conv2, 'block5_conv3')
    block5_conv4 = tf_conv_relu(block5_conv3, 'block5_conv4')
    block5_pool = tf_avg_pool(block5_conv4, 'block5_pool')
    
    if mode == 'content':  # Return content representation
        return block4_conv2
    elif mode == 'style':  # Return outputs of 5 layers, to represent style
        return [block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1]
    elif mode == 'generate':  # Return both
        outputs = {"content_layer": block4_conv2,
                  "style_layers": [block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1]}
    return outputs

