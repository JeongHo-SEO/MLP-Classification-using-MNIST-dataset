### Project_2_code ###
print(f"StudentID: {202011101}\tName: {'서정호'}")
print("angle = 0: before training dataset augmentation for comparison.")

### import libraries ###
import numpy as np
from scipy.special import expit as sigmoid #
import matplotlib.pyplot as plt # library for plotting arrays, i.e, data visualization
import cv2 # OpenCV for image processing, b/c scipy.misc is deprecated #
from scipy.ndimage import rotate # scipy.ndimage for image rotation, interpolation.rotate was changed to rotate.

"""
### numpy library: how to generate column vectors
## np.array.reshape(row, column): array to matrix
## np.array.ndim: dimension of array
# np.array(list, tuple, ...): 1D row vector
# np.array(list, tuple, ...).reshape(-1, 1): 2D column vector for matrix multiplication. Not same to np.array.T which is a 1D column vector.
# np.dot(matrix, matrix): matrix multiplication, including vector dot product
# np.zeros(row, column): matrix of zeros
# np.random.normal(mean, std, (row, column)): matrix of random numbers
# np.argmax(array): return index(int type)
# np.array(sequence, dtype=np.float64): array with float data type
"""

# As a result, every array is 2D column vector for matrix multiplication Furthermore, every array data type should be float.
# so I used np.array(sequence, dtype=np.float64).reshape(-1, 1) instead of np.array.T

"""
# training dataset: augmentation full-MNIST training dataset with rotation(5, 10, 15, 20, 25), 60,000 * 3 = 180,000
# testing dataset: full-MNIST testing dataset, 10,000
# performance: check to 98%.
"""

## MNIST dataset.csv ##
# (label, pixel_1, pixel_2, ..., pixel_784), 28*28 pixels = 784 pixels
# label: 0~9, pixel: 0~255 (0:black, 255:white) for brightness
# word: white, background: black: inversion needed (invert smaller hand-writing dataset which is word: black, background: white)
# each line: target + comma + pixel + comma + pixel +... + comma + pixel + '\n' => file.readlines(): list of str element(each line)

## hyper-parameters values ##
input_size = 784 # 28*28
hidden_size = 200 # experimentally determined
output_size = 10 # 0~9
learning_rate = 0.1 # experimentally determined
epochs = [5, 10]

## rotation angles for augmentation ##
angles = [0, 5, 10, 15, 20, 25] # angle = 0: Before training dataset augmentation for comparison.


## loading training dataset ##
with open("../MNIST_dataset_csv/60,000_full_mnist_train.csv", 'r') as f:
    MNIST_training_data_list = f.readlines() # list of str element(each element)

## loading testing dataset ##
with open("../MNIST_dataset_csv/10,000_full_mnist_test.csv", 'r') as f:
    MNIST_testing_data_list = f.readlines() # list of str element(each element)


### Preprocessing for MNIST data ###

## re-scale(normalization, 0.0 to 1.0 & shift, 0.01 to 1.00)
def re_scale(img_array):
    return (img_array / 255.0 * 0.99) + 0.01

## one-hot encoding for label ##
def encoding(label):
    label_array = np.zeros(output_size).reshape(-1,1) + 0.01
    label_array[int(label)] = 0.99
    return label_array

def data_preprocessing(data_list):
    data = []
    for element in data_list: # str
        # split each element by the ',' commas
        all_values = element.split(',') # str to list, all_values: [ "label", "pixel_1", "pixel_2", ..., "pixel_784" ]

        # Preprocessing
        img_array = re_scale( np.array(all_values[1:], dtype=np.float64).reshape(-1,1) )
        label_array = encoding(all_values[0])

        data += [ (img_array, label_array) ] # (img_array, label_array) of float

    return data

training_data = data_preprocessing(MNIST_training_data_list) # (img_array, label_array) of float
testing_data = data_preprocessing(MNIST_testing_data_list) # (img_array, label_array) of float


## data_augmentation: rotation ##
def rotate_image(img_array, label_array, angle): # img_array: 2D column vector, 784x1
    if angle == 0:
        return img_array, label_array
    else:
        ## parameters of rotate function ##
        # input array: should be at least 2D, thus reshape to (28,28) matrix is better than 2D column vector(sometimes cause error)
        # (optional)cval: padding value, default 0.0
        # (optional)reshape: output shape, default True

        # reshape to 28x28 image
        img_matrix= img_array.reshape(28, 28)
        anticlockwise_matrix = rotate(img_matrix, angle, cval=0.01, reshape=False)  # angle: positive
        clockwise_matrix = rotate(img_matrix, -angle, cval=0.01, reshape=False)

        # reshape to 2D column vector
        original_array = img_array
        anticlockwise_array = anticlockwise_matrix.reshape(-1,1)
        clockwise_array = clockwise_matrix.reshape(-1,1)
        return (img_array, label_array), (anticlockwise_array, label_array), (clockwise_array, label_array) # tuple of 3 (img_array, label_array)


## training_data update: Augmentation ##
# (dict) key: angle, value: list of (img_array, label_array)
augmented_training_data = dict() # initialization dictionary
augmented_training_data.update( { 0 : training_data } )
augmented_training_data.update(
    { angle : [ rotate_image(img_array, label_array, angle)[0] for img_array, label_array in training_data ]
    + [ rotate_image(img_array, label_array, angle)[1] for img_array, label_array in training_data ]
    + [ rotate_image(img_array, label_array, angle)[2] for img_array, label_array in training_data ]
    for angle in angles if angle != 0 }
    )


## Neural Network Class Definition ##
class Neural_Network: # MLP
    # Initialize the neural network: Constructor #
    def __init__(self, input_size = input_size, hidden_size = hidden_size, output_size = output_size, learning_rate = learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # link Weights matrices, W_ih and W_ho
        # Initialize Weights matrix with gaussian distribution
        self.W_ih = np.random.normal(0.0, pow(self.hidden_size, -0.5), (self.hidden_size, self.input_size))
        self.W_ho = np.random.normal(0.0, pow(self.output_size, -0.5), (self.output_size, self.hidden_size))

        # learning rate
        self.lr = learning_rate
        
        # activation function: sigmoid function
        self.activation_function = lambda x: sigmoid(x) # def, return omission is okay in lambda function, generally used for simple functions

    # train the neural network #
    def train(self, input_array, target_array):
        # convert list to matrix: [1, 2, 3, 4] to [[1], [2], [3], [4]].T
        inputs = input_array
        targets = target_array
        
        ## Forward Propagation ##

        # input to hidden layer
        hidden_inputs = np.dot(self.W_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # hidden to output layer
        final_inputs = np.dot(self.W_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        ## Backward Propagation ##
        # error
        output_errors = targets - final_outputs

        # hidden_errors are distributed according to the proportion of weights from output_errors
        # ignore the denominator for simplification
        hidden_errors = np.dot(self.W_ho.T, output_errors) 
        
        # update the weights
        self.W_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.W_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # query the neural network #
    def query(self, img_array):
        inputs = img_array
        
        ## Forward Propagation ##

        # input to hidden layer
        hidden_inputs = np.dot(self.W_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # hidden to output layer
        final_inputs = np.dot(self.W_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Initialize for the performance graph #
pts = { epoch : [ ] for epoch in epochs } # value: (angle, accuracy)

## train & test the neural network ##
for epoch in epochs: # 5, 10
    for angle in angles:
        # initialization new instance for each angle
        instance = Neural_Network() # input_size = 784, hidden_size = 200, output_size = 10, learning_rate = 0.1
        ### train the neural network ###
        for _ in range(epoch):
            for img_array, label_array in augmented_training_data[angle]: # (img_array, label_array) of float
                instance.train(img_array, label_array) # present instance's weights update

        ### test the neural network ###
        # scorecard for how well the network performs, initially empty
        scorecard = []
        for img_array, label_array in testing_data: # (img_array, label_array) of float
            output_array = instance.query(img_array) # using updated weights

            # the index of the highest value corresponds to the label
            output = np.argmax(output_array)
            target = np.argmax(label_array)
            # append correct or incorrect to list
            if (output == target):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                scorecard.append(0)

        # calculate the performance score, the fraction of correct answers
        scorecard_array = np.array(scorecard, dtype=np.float64)
        # performance score
        accuracy = scorecard_array.sum() / scorecard_array.size
        # print accuracy
        print(f"Performance: {accuracy} | angle: {angle}, epoch: {epoch}")
        # store coordinate data for result graph #
        pts[epoch].append( (angle, accuracy) )


## visualizing the results with graph: including marker and label ##
x = { epoch : [ angle for angle, accuracy in pts[epoch] ] for epoch in epochs }
y = { epoch : [ accuracy for angle, accuracy in pts[epoch] ] for epoch in epochs }

## plot the graph together ##
plt.plot( x[epochs[0]], y[epochs[0]], 'bD-', label = f'{epochs[0]} epochs' ) # epoch = 5
plt.plot( x[epochs[1]], y[epochs[1]], 'rD-', label = f'{epochs[1]} epochs' ) # epoch = 10

# title of the graph
plt.title("Performance and Epoch\nMNIST dataset with 3-layer neural network")
# title of each axis
plt.xlabel("additional images at a+/- angle (degrees)")
plt.ylabel("performance")
# label of each graph
plt.legend()
# grid of y-axis
plt.grid(axis = 'y')

plt.show()