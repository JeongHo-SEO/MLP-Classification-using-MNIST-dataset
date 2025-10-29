### Project_1_code ###
print(f"StudentID: {202011101}\tName: {'서정호'}")

### import libraries ###
import numpy as np
from scipy.special import expit as sigmoid #
import matplotlib.pyplot as plt # library for plotting arrays, i.e, data visualization
import cv2 # OpenCV for image processing, b/c scipy.misc is deprecated #

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
# training dataset: full-MNIST training dataset, 60,000
# testing dataset: my own hand-writing dataset(png, 28x28 pixels)
# performance: under 97% is okay.
"""

## MNIST dataset.csv ##
# (label, pixel_1, pixel_2, ..., pixel_784), 28*28 pixels = 784 pixels
# label: 0~9, pixel: 0~255 (0:black, 255:white) for brightness
# word: white, background: black: inversion needed (invert smaller hand-writing dataset which is word: black, background: white)
# each line: target + comma + pixel + comma + pixel +... + comma + pixel + '\n' => file.readlines(): list of str element(each line)

## hand-writing dataset: 10*3 = 30 ##
# 1_Tablet_capture_image
# 2_hand_writing_image_phone_scan
# 3_MS_word_capture_image ##

## hyper-parameters values ##
input_size = 784 # 28*28
hidden_size = 200 # experimentally determined
output_size = 10 # 0~9
learning_rate = [0.1, 0.2] # experimentally determined
epochs = [1, 2, 4, 5, 6, 7, 10, 20] # experimentally determined


## loading training dataset ##
with open("../MNIST_dataset_csv/60,000_full_mnist_train.csv", 'r') as f:
    MNIST_training_data_list = f.readlines() # list of str element(each element)

## loading testing dataset: 30 hand-writing data ##
# (label, path)
hand1_path_list = [ (label, f"./handwriting_images/1_Tablet_capture_image/1_{label}.png") for label in range(0,10) ] # label: 0~9
hand2_path_list = [ (label, f"./handwriting_images/2_hand_writing_image_phone_scan/2_{label}.png") for label in range(0,10) ] # label: 0~9
hand3_path_list = [ (label, f"./handwriting_images/3_MS_word_capture_image/3_{label}.png") for label in range(0,10) ] # label: 0~9


### Preprocessing for MNIST data ###

## re-scale(normalization, 0.0 to 1.0 & shift, 0.01 to 1.00)
def re_scale(img_array):
    return (img_array / 255.0 * 0.99) + 0.01

## one-hot encoding for label ##
def encoding(label):
    label_array = np.zeros(output_size).reshape(-1,1) + 0.01
    label_array[int(label)] = 0.99
    return label_array

## training_data Preprocessing ##
training_data = []
for element in MNIST_training_data_list: # str
    # split the element by the ',' commas
    all_values = element.split(',') # str to list, all_values: [ "label", "pixel_1", "pixel_2", ..., "pixel_784" ]

    # Preprocessing
    img_array = re_scale( np.array(all_values[1:], dtype=np.float64).reshape(-1,1) )
    label_array = encoding(all_values[0])

    training_data += [ (img_array, label_array) ] # (img_array, label_array) of float


### Preprocessing for Hand-writing data ###

## image preprocessing: image to array ##
def path_to_array(path, target_size = (28,28)):
    # read the image as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # resize the image
    img_resized = cv2.resize(img, target_size) # 28*28 = 784 pixels

    # color binarization: words are black(0), background is white(255)
    _, img_bin = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY) # _ = threshold 128

    # # convert to 1D array
    # img_array = img_bin.flatten()
    img_array = img_bin.reshape(-1,1)
    return img_array

## inversion for handwriting_data ##
def invert(img_array):
    return 255.0 - img_array


testing_data = [ ( re_scale(invert(path_to_array(path))), encoding(label) ) for label, path in hand1_path_list + hand2_path_list + hand3_path_list ] # (img_array, label_array) of float


## Neural Network Class Definition ##
class Neural_Network: # MLP
    # Initialize the neural network: Constructor #
    def __init__(self, learning_rate, input_size = input_size, hidden_size = hidden_size, output_size = output_size):
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
pts = { lr : [ ] for lr in learning_rate } # value: (epoch, accuracy)

## train & test the neural network ##
for lr in learning_rate: # 0.1, 0.2
    for epoch in epochs: # 1, 2, 4, 5, 6, 7, 10, 20
        # initialization new instance for each epoch
        instance = Neural_Network(lr) # learning_rate, input_size = 784, hidden_size = 200, output_size = 10
        ### train the neural network ###
        for _ in range(epoch):
            for img_array, label_array in training_data:
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
        print(f"Performance: {accuracy} | epoch: {epoch}, learning rate: {lr}")
        # store coordinate data for result graph #
        pts[lr].append( (epoch, accuracy) )


## visualizing the results with graph: including marker and label ##
x = { lr : [ epoch for epoch, accuracy in pts[lr] ] for lr in learning_rate }
y = { lr : [ accuracy for epoch, accuracy in pts[lr] ] for lr in learning_rate }

## plot the graph together ##
plt.plot( x[learning_rate[0]], y[learning_rate[0]], 'rD-', label = f'Ir = {learning_rate[0]}' ) # lr = 0.1
plt.plot( x[learning_rate[1]], y[learning_rate[1]], 'bs-', label = f'Ir = {learning_rate[1]}' ) # lr = 0.1

# title of the graph
plt.title("Performance and Epoch\nMNIST dataset with 3-layer neural network")
# title of each axis
plt.xlabel("number of epochs")
plt.ylabel("performance")
# label of each graph
plt.legend()
# grid of y-axis
plt.grid(axis = 'y')

plt.show()