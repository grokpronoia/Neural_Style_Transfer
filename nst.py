import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained VGG-19 model
# model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# print(model)

'''
Build the content cost function  Jcontent(C,G)Jcontent(C,G)
Build the style cost function  Jstyle(S,G)Jstyle(S,G)
Put it together to get  J(G)=αJcontent(C,G)+βJstyle(S,G)J(G)=αJcontent(C,G)+βJstyle(S,G) .
'''

#Sample content image
content_image = scipy.misc.imread("/Users/marrowgari/Documents/HappyClouds/mars.jpg")
imshow(content_image)

# Steps for computing content cost
'''
1.Retrieve dimensions from a_G:
    To retrieve dimensions from a tensor X, use: X.get_shape().as_list()
2.Unroll a_C and a_G
3.Compute the content cost:
'''
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))

    # compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)

    return J_content

tf.reset_default_graph()

with tf.Session() as test:
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))

#Sample style image
style_image = scipy.misc.imread("/Users/marrowgari/Documents/HappyClouds/kadinsky.jpg")
imshow(style_image)

# Compute Style matrix / Gram matrix
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, tf.transpose(A))

    return GA

tf.reset_default_graph()

with tf.Session() as test:
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)

    print("GA = " + str(GA.eval()))

#Steps for computing style cost
'''
1.Retrieve dimensions from the hidden layer activations a_G:
    To retrieve dimensions from a tensor X, use: X.get_shape().as_list()
2.Unroll the hidden layer activations a_S and a_G into 2D matrices
3.Compute the Style matrix of the images S and G
4.Compute the Style cost
'''
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * n_C**2 * (n_W * n_H)**2)

    return J_style_layer

tf.reset_default_graph()

with tf.Session() as test:
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer.eval()))

# Merging style results from several different layers
# values for  lambd[l] are given in STYLE_LAYERS
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

# Create a cost function that minimizes both the style and the content cost
# The total cost is a linear combination of the content cost J_content(C,G) and the style cost J_style(S,G)
# alpha and beta are hyperparameters that control the relative weighting between content and style
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style

    return J

tf.reset_default_graph()

with tf.Session() as test:
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))

# Steps
'''
Create an Interactive Session
Load the content image
Load the style image
Randomly initialize the image to be generated
Load the VGG16 model
Build the TensorFlow graph:
    Run the content image through the VGG16 model and compute the content cost
    Run the style image through the VGG16 model and compute the style cost
    Compute the total cost
    Define the optimizer and the learning rate
Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
'''

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

# load, reshape, and normalize "content" image
content_image = scipy.misc.imread("/Users/marrowgari/Documents/HappyClouds/mars.jpg")
content_image = reshape_and_normalize_image(content_image)

# load, reshape, and normalize "style" image
style_image = scipy.misc.imread("/Users/marrowgari/Documents/HappyClouds/kadinsky.jpg")
style_image = reshape_and_normalize_image(style_image)

# Initialize generated image as noisy image created from content image
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

# load the VGG16 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

#to compute the content cost using layer conv4_2, assign a_C and a_G to be the appropriate hidden layer activations.
'''
Assign the content image to be the input to the VGG model.
Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
Set a_G to be the tensor giving the hidden layer activation for the same layer.
Compute the content cost using a_C and a_G.
'''

# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. a_G references model['conv4_2']
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Run the Tensorflow graph
# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# Compute the total cost
J = total_cost(J_content, J_style,  alpha = 10, beta = 40)

# set up the Adam optimizer in TensorFlow using a learning_rate of 2.0
# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

# Implement the model_nn() function which initializes the variables of the tensorflow graph, assigns the input image (initial generated image) as the input of the VGG16 model and runs the train_step for a large number of steps.
def model_nn(sess, input_image, num_iterations = 200):

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

# Generate image
model_nn(sess, generated_image)
