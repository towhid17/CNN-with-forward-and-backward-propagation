import numpy as np
import os
import numpy as np
import pandas as pd
import cv2
import pickle
import copy
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score, confusion_matrix
# import f1 score
from sklearn.metrics import f1_score
import datetime
import seaborn as sns


import abc


# Model Component Definition
class ModelComponent(abc.ABC):
    @abc.abstractmethod
    def forward(self, u):
        pass
    
    @abc.abstractmethod
    def backward(self, del_v, lr):
        pass

class Convolution(ModelComponent):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, learning_rate=0.01):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = np.zeros((out_channels, 1))
        self.learning_rate = learning_rate

    def forward(self, input):
        batch_size, height, width, in_channels = input.shape

        # initialize weights with xavier initialization
        if self.weights is None:
            self.weights = np.random.randn(self.out_channels, self.kernel_size, self.kernel_size, in_channels) / math.sqrt(self.kernel_size * self.kernel_size)

        # if self.weights is None:
        #     self.weights = np.random.randn(self.out_channels, self.kernel_size, self.kernel_size, in_channels) / math.sqrt(self.kernel_size * self.kernel_size)
        
        self.input = input

        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        input_padded = np.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(out_height):
                    for l in range(out_width):
                        out[i, k, l, j] = np.sum(input_padded[i, k*self.stride:k*self.stride+self.kernel_size, l*self.stride:l*self.stride+self.kernel_size, :] * self.weights[j, :, :, :]) + self.biases[j, 0]
        

        # for h in range(out_height):
        #     for w in range(out_width):
        #         x_slice = input_padded[:, h*self.stride:h*self.stride+self.kernel_size, w*self.stride:w*self.stride+self.kernel_size, :]
        #         # use np.dot
        #         out[:, h, w, :] = np.dot(x_slice, self.weights.reshape(self.out_channels, -1).T) + self.biases.T

        return out

    def backward(self, grad_output):
        batch_size, height, width, in_channels = self.input.shape
        batch_size, out_height, out_width, out_channels = grad_output.shape

        grad_input = np.zeros((batch_size, height, width, in_channels))
        grad_input_padded = np.pad(grad_input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        grad_weights = np.zeros((self.out_channels, self.kernel_size, self.kernel_size, in_channels))
        grad_biases = np.zeros((self.out_channels, 1))

        for i in range(batch_size):
            image = grad_input_padded[i]
            for j in range(self.out_channels):
                for k in range(out_height):
                    for l in range(out_width):
                        grad_input_padded[i, k*self.stride:k*self.stride+self.kernel_size, l*self.stride:l*self.stride+self.kernel_size, :] += self.weights[j, :, :, :] * grad_output[i, k, l, j]                    
                        grad_biases[j, 0] += grad_output[i, k, l, j]

                        row_start = k * self.stride
                        row_end = row_start + self.kernel_size
                        col_start = l * self.stride
                        col_end = col_start + self.kernel_size
                        region = image[row_start:row_end, col_start:col_end, :]
                        grad_weights[j] += region * grad_output[i, k, l, j]


        grad_input = grad_input_padded[:, self.padding:self.padding+height, self.padding:self.padding+width, :]


        # # normalize gradients
        # grad_weights /= batch_size
        # grad_biases /= batch_size

        # clip gradients
        grad_weights = np.clip(grad_weights, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases


        return grad_input



class ReLU(ModelComponent):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_outputs):
        grad_inputs = np.where(self.inputs > 0, grad_outputs, 0)
        return grad_inputs


class MaxPooling(ModelComponent):
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride

    def forward(self, inputs):
        n, h, w, c = inputs.shape
        h_out = (h - self.filter_dim) / self.stride + 1
        w_out = (w - self.filter_dim) / self.stride + 1
        h_out, w_out = int(h_out), int(w_out)
        outputs = np.zeros((n, h_out, w_out, c))
        for i in range(n):
            for j in range(c):
                for k in range(h_out):
                    for l in range(w_out):
                        outputs[i, k, l, j] = np.max(
                            inputs[i, k * self.stride: k * self.stride + self.filter_dim, l * self.stride: l * self.stride + self.filter_dim, j])
        self.inputs = inputs
        return outputs

    def backward(self, grad_outputs):
        n, h, w, c = self.inputs.shape
        h_out = (h - self.filter_dim) / self.stride + 1
        w_out = (w - self.filter_dim) / self.stride + 1
        h_out, w_out = int(h_out), int(w_out)
        grad_inputs = np.zeros_like(self.inputs)
        for i in range(n):
            for j in range(c):
                for k in range(h_out):
                    for l in range(w_out):
                        window = self.inputs[i, k * self.stride: k * self.stride +
                                        self.filter_dim, l * self.stride: l * self.stride + self.filter_dim, j]
                        m = np.max(window)
                        grad_inputs[i, k * self.stride: k * self.stride + self.filter_dim, l * self.stride: l *
                                    self.stride + self.filter_dim, j] = np.where(window == m, grad_outputs[i, k, l, j], 0)
        return grad_inputs


class Flatten(ModelComponent):
    def forward(self, inputs):
        n, h, w, c = inputs.shape
        outputs = inputs.reshape(n, c * h * w)
        self.inputs = inputs
        return outputs

    def backward(self, grad_outputs):
        n, h, w, c = self.inputs.shape
        grad_inputs = grad_outputs.reshape(n, h, w, c)
        return grad_inputs


class Dense(ModelComponent):
    def __init__(self, output_dim, learning_rate=0.01):
        self.weights = None
        self.biases = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.input_dim = inputs.shape[1]


        # if self.weights is None:
        #     self.weights = np.random.normal(0, 0.1, (inputs.shape[1], self.biases.shape[1]))

        # initialize weights with Xavier initialization
        if self.weights is None:
            self.weights = np.random.normal(0, np.sqrt(2/(inputs.shape[1] + self.biases.shape[1])), (inputs.shape[1], self.biases.shape[1]))
        
        outputs = np.dot(inputs, self.weights) + self.biases
        return outputs

    def backward(self, grad_outputs):
        grad_weights = np.dot(self.inputs.T, grad_outputs)
        grad_biases = np.sum(grad_outputs, axis=0, keepdims=True)

        # normalize gradients
        # grad_weights /= self.input_dim
        # grad_biases /= self.input_dim

        # gradient clipping
        grad_weights = np.clip(grad_weights, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)

        self.weights -= grad_weights*self.learning_rate
        self.biases -= grad_biases*self.learning_rate
        grad_inputs = np.dot(grad_outputs, self.weights.T)

        # print(self.weights)
        return grad_inputs


class SoftmaxCrossEntropy(ModelComponent):
    
    def forward(self, input_data):
        # print(input_data)
        input_data = input_data - np.max(input_data, axis=1, keepdims=True)
        inp_exp = np.exp(input_data)
        output_data = inp_exp / np.sum(inp_exp, axis=1, keepdims=True)
        self.output = output_data
        return output_data

    def backward(self, grad_outputs):
        grad_inputs = np.copy(grad_outputs)
        return grad_inputs


folder_list = ["./NumtaDB/training-a"]
folder_list_test = ["./NumtaDB/training-b"]


def load_training_data(folder_list, size):
    X_train = []
    Y_train = []

    for i in range(len(folder_list)):
        # Load images
        folder = folder_list[i]
        image_files = [os.path.join(folder, f)
                       for f in os.listdir(folder) if f.endswith(".png")]
        
        # read images amount of size
        images = [cv2.imread(f, 0) for f in image_files[:size]]
        
        imgs = []
        for img in images:
            g_img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
            imgs.append(g_img)

        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs = (imgs - mean) / std

        imgs = [np.expand_dims(img, axis=2) for img in imgs]

        images = np.array(imgs)
        X_train.extend(images)

        # Load CSV file
        csv_file = os.path.join("", folder + ".csv")
        df = pd.read_csv(csv_file)
        labels = df["digit"].to_numpy()
        Y_train.extend(labels)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # take only 1000 images
    X_train = X_train[:size]
    Y_train = Y_train[:size]

    return X_train, Y_train



def load_data(size):
    X, Y = load_training_data(folder_list, size)
    X_test, Y_test = load_training_data(folder_list_test, size)

    # take 80% of training data for training and 20% for validation
    X_train = X[:int(size*0.8)]
    Y_train = Y[:int(size*0.8)]   

    X_val = X[int(size*0.8):]
    Y_val = Y[int(size*0.8):]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



def predict_from_pickle(model, X_test, Y_test):
    # load the model parameters
    model.load_parameters("model.pkl")
    # Forward pass
    output = model.forward(X_test)
    predictions = np.argmax(output, axis=1)
    accuracy = accuracy_score(Y_test, predictions)
    return accuracy



class Model_CNN:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def create_model(self):
        lr = self.learning_rate
        conv1 = Convolution(6, 5, 1, 1, learning_rate=lr)
        relu1 = ReLU()
        max = MaxPooling(2, 2)
        # conv2 = Convolution(16, 5, 1, 1, learning_rate=lr)
        # relu2 = ReLU()
        # max2 = MaxPooling(2, 2)
        flat = Flatten()
        dense1 = Dense(100, learning_rate=lr)
        relu3 = ReLU()
        dense2 = Dense(10, learning_rate=lr)
        # relu4 = ReLU()
        # dense3 = Dense(10, learning_rate=lr)
        soft = SoftmaxCrossEntropy()

        self.conv1 = conv1
        self.relu1 = relu1
        self.max = max
        # self.conv2 = conv2
        # self.relu2 = relu2
        # self.max2 = max2
        self.flat = flat
        self.dense1 = dense1
        self.relu3 = relu3
        self.dense2 = dense2
        # self.relu4 = relu4
        # self.dense3 = dense3
        self.soft = soft
    
    def forward(self, input):
        out = self.conv1.forward(input)
        out = self.relu1.forward(out)
        out = self.max.forward(out)
        # out = self.conv2.forward(out)
        # out = self.relu2.forward(out)
        # out = self.max2.forward(out)
        out = self.flat.forward(out)
        out = self.dense1.forward(out)
        out = self.relu3.forward(out)
        out = self.dense2.forward(out)
        # out = self.relu4.forward(out)
        # out = self.dense3.forward(out)
        out = self.soft.forward(out)

        return out
    
    def backward(self, y_true, y_pred):
        out = self.soft.backward(y_pred-y_true)
        # out = self.dense3.backward(out)
        # out = self.relu4.backward(out)
        out = self.dense2.backward(out)
        out = self.relu3.backward(out)
        out = self.dense1.backward(out)
        out = self.flat.backward(out)
        # out = self.max2.backward(out)
        # out = self.relu2.backward(out)
        # out = self.conv2.backward(out)
        out = self.max.backward(out)
        out = self.relu1.backward(out)
        out = self.conv1.backward(out)

        return out

    def get_parameters(self):
        # Get all the parameters
        parameters = []
        parameters.append(self.conv1.weights)
        parameters.append(self.conv1.biases)
        # parameters.append(self.conv2.weights)
        # parameters.append(self.conv2.biases)
        parameters.append(self.dense1.weights)
        parameters.append(self.dense1.biases)
        parameters.append(self.dense2.weights)
        parameters.append(self.dense2.biases)
        # parameters.append(self.dense3.weights)
        # parameters.append(self.dense3.biases)

        return parameters

    def set_parameters(self, parameters):
        # Set all the parameters
        self.conv1.weights = parameters[0]
        self.conv1.biases = parameters[1]
        # self.conv2.weights = parameters[2]
        # self.conv2.biases = parameters[3]
        self.dense1.weights = parameters[2]
        self.dense1.biases = parameters[3]
        self.dense2.weights = parameters[4]
        self.dense2.biases = parameters[5]
        # self.dense3.weights = parameters[8]
        # self.dense3.biases = parameters[9]


    def save_parameters(self, path):
        # Save all the parameters
        parameters = self.get_parameters()
        with open(path, 'wb') as f:
            pickle.dump(parameters, f)
    
    def load_parameters(self, path):
        # Load all the parameters
        with open(path, 'rb') as f:
            parameters = pickle.load(f)
        self.set_parameters(parameters)
    
    
    def predict(self, X_test, y_test):
        # Forward pass
        y_pred = self.forward(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        # Return the predictions
        acc_s = accuracy_score(y_test, y_pred)

        return acc_s


        

def train(model, X_train, y_train, X_val, Y_val, epochs, batch_size):
    # save time stamp in results.txt
    with open('results.txt', 'a') as f:
        f.write('----------------------------------------\n')
        f.write('time: ' + str(datetime.datetime.now()) + '\n')
        f.write('----------------------------------------\n')
        f.close()

    prev_train_acc = 0
    for epoch in range(epochs):
        train_predictions = []
        for i in range(0, X_train.shape[0], batch_size):
            batch_train_images = X_train[i:i+batch_size]
            batch_train_labels = y_train[i:i+batch_size]
            one_hot_labels = np.zeros((batch_train_labels.shape[0], 10))
            # one hot
            one_hot_labels[np.arange(batch_train_labels.shape[0]), batch_train_labels] = 1
            # forward
            probs = model.forward(batch_train_images)
            # save train predictions
            train_predictions.extend(probs)
            # backward
            model.backward(one_hot_labels, probs)

    
        # epoch: calculate f1_score, train_acc, train_loss, validation_loss, validation_acc
        # softmax loss of train
        log_probs_train = np.log(train_predictions)
        train_loss = -np.mean(log_probs_train[np.arange(y_train.shape[0]), y_train])
        # train accuracy
        train_predictions = np.argmax(train_predictions, axis=1)
        train_acc = accuracy_score(y_train, train_predictions)
        
        # fi score of train
        f1_train = f1_score(y_train, train_predictions, average='macro')

        # validation
        probs_val = model.forward(X_val)
        # softmax loss of validation
        log_probs_val = np.log(probs_val)
        val_loss = -np.mean(log_probs_val[np.arange(Y_val.shape[0]), Y_val])
        # validation accuracy
        y_pred = np.argmax(probs_val, axis=1)
        val_acc = accuracy_score(Y_val, y_pred)
        # f1 score
        f1_validation = f1_score(Y_val, y_pred, average='macro')

        print('------------------------------------')
        print('New Epoch: ', epoch+1)
        print('------------------------------------')
        print(f'Epoch {epoch+1}/{epochs}: Train Accuracy: {train_acc}')
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss}')
        print(f'Epoch {epoch+1}/{epochs}: Validation Loss: {val_loss}')
        print(f'Epoch {epoch+1}/{epochs}: Validation Accuracy: {val_acc}')
        print(f'Epoch {epoch+1}/{epochs}: F1 Score Train: {f1_train}')
        print(f'Epoch {epoch+1}/{epochs}: F1 Score Validation: {f1_validation}')
        print('------------------------------------')

        # save acc, loss and f1 score in a txt file
        with open('results.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{epochs}: Train Accuracy: {train_acc}\n')
            f.write(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss}\n')
            f.write(f'Epoch {epoch+1}/{epochs}: Validation Loss: {val_loss}\n')
            f.write(f'Epoch {epoch+1}/{epochs}: Validation Accuracy: {val_acc}\n')
            f.write(f'Epoch {epoch+1}/{epochs}: F1 Score Train: {f1_train}\n')
            f.write(f'Epoch {epoch+1}/{epochs}: F1 Score Validation: {f1_validation}\n')
            f.write('------------------------------------\n')
            f.close()

        # save model weights and biases in a pickle file 
        if(prev_train_acc<=val_acc):
            model.save_parameters("model.pkl")
            print("Model saved::", val_acc)
            prev_train_acc = val_acc
        else:
            print("Model not saved::", val_acc)
    

    return model


def independentTest(model):    
    size = 12000
    X_train, Y_train, _, _, X_test, Y_test = load_data(size)
    model.load_parameters("model.pkl")
    # model = Model_CNN(learning_rate=0.0001)
    # model.create_model()
    # model = train(model, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32)
    X_test = X_test[:2000]
    Y_test = Y_test[:2000]
    # print("Image and label data loaded for independent test")
    # print(X_test.shape)
    test_scores = model.forward(X_test)
    y_pred = np.argmax(test_scores, axis=1)
    # print(y_pred)
    acc = accuracy_score(Y_test, y_pred)    
    f1 = f1_score(Y_test, y_pred, average='macro')
    print(f'Independent Test Accuracy: {acc}')
    print(f'Independent Test F1: {f1}')

    print('\nConfusion Matrix: ')
    print(confusion_matrix(Y_test, y_pred))
    #Plot the confusion matrix using seaborn
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load the dataset
    lr = 0.0001
    size = 12000
    batch_size = 32
    ep = 10
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(size)
    # Create the model
    model = Model_CNN(learning_rate=lr)
    model.create_model()
    independentTest(model)
    # model = train(model, X_train, Y_train, X_val, Y_val, epochs=ep, batch_size=batch_size)
    
    # model.save_parameters("model.pkl")

    # Predict
    # accuracy = model.predict(X_test, Y_test)
    # print("Test accuracy: %f" % (accuracy*100))

    ############################
    # Load the parameters from the pickle file and predict
    # test_acc = predict_from_pickle(model, X_test, Y_test)
    # print("Test accuracy: %f" % (test_acc*100))




if __name__ == "__main__":
    main()