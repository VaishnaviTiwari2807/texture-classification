# texture-classification
INTRODUCTION
For image classification applications, convolutional neural networks (CNNs) are a typical type of neural
network architecture. In this project, we'll use PyTorch to train a CNN using the Describable Textures
Dataset (DTD). To comprehend how the network learns to represent textures, we will display intermediate
values. We will also investigate the implications of underfitting and overfitting. A specific type of neural
network called a convolutional neural network (CNN) is made to handle data having a known grid-like
architecture, such as photographs. Traditional machine learning models are outperformed by CNNs in
image classification and other computer vision tasks.
DATASET
A collection of textures, each falling under one of 47 categories, can be found in the Describable Textures
Dataset (DTD). 5640 photos having a dimension of 640 by 480 pixels are included in the dataset. We will
resize the photos to 224 by 224 pixels for this project.
Training a CNN
The following steps are involved in training a CNN:
● Preprocessing the data entails shrinking the photos to a set size and normalizing the pixel values
in order to prepare the input data.
● The CNN architecture is as follows: Convolutional layers, activation algorithms (like ReLU), and
pooling layers are what make up a CNN. Before the output is formed, one or more completely
connected layers are frequently passed through to create the final output.
● After the CNN architecture has been established, the model is assembled using a loss function, an
optimizer, and metrics.
● Model training: Using backpropagation and batches of data, the model is trained. Every epoch,
the loss is calculated and utilized to modify the model's parameters.
● Model evaluation: Following training, the model is assessed on a test set to determine its
accuracy.
Base Model(task1)
The provided model is a deep convolutional neural network (CNN) for image classification, with output
classes of 47 and input picture dimensions of 224 × 224.
The convolutional layers and the fully connected layers are the two key parts of the model.
Three 2D convolutional layers with a kernel size of 3 x 3 and a stride of 1 make up the convolutional
layers. When the padding value is set to 1, each convolutional layer's output will have the same
dimensions as the input. The first convolutional layer comprises 32 output channels and 3 input channels
(for the RGB image). There are 64 output channels and 32 input channels in the second convolutional
layer. There are 64 input channels and 128 output channels in the third convolutional layer.
A 2D max-pooling layer with a kernel size of 2 x 2 and stride of 2 is applied after each convolution layer,
followed by the Rectified Linear Unit (ReLU) activation function. The convolution layer's output is
downsampled by the max-pooling layer, which also helps the output's spatial dimensions.
Two linear layers are present in the fully connected layers. The output of the last convolution layer is
flattened to a 1D tensor by the first linear layer. The sum of the output channels (128) and the output's
spatial dimensions (128 x 28 x 28) determines the layer's input size. (28 x 28).This layer has 256 output
features, and the ReLU activation function is applied after the linear transformation. The Softmax
function is used to determine the final probability for each class, and the second linear layer has an output
size equal to the number of output classes (47).
There are a total of 526,823 parameters for the model to learn.
Result: This model was run for 50 epoch and results in 10.6915% training accuracy and 7.55% testing
accuracy.
Underfitting(task2) :I use the same model for 20 iterations, but if the dataset is vast and complex, it might not
have enough time to understand the pattern of the data. In this scenario, the model won't perform well on
either the training or validation sets, and the metrics for accuracy and loss won't considerably improve. It
is therefore an instance of underfitting.
Result: For 20 epoch the training accuracy and testing accuracy are 5.0000% and 4.68% respectively
Overfitting(task3): I used the model with 100 epochs, when the model trains on the training data for an
excessively long time and begins to memorize it, it overfits due to additional epochs and loses its ability
to generalize to new data. In turn, this causes the model to perform well on training data but badly on
validation or test data.
The model becomes more complicated and begins to fit the noise in the training data as the number of
epochs rises, which causes overfitting. An increase in training accuracy and a drop in validation or test
accuracy are two signs of this.
