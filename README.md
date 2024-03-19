# texture-classification
Introduction
For image classification applications, Convolutional Neural Networks (CNNs) are a common type of neural network architecture. In this project, we utilize PyTorch to train a CNN on the Describable Textures Dataset (DTD). Our goal is to understand how the network learns to represent textures by displaying intermediate values. We will also explore the implications of underfitting and overfitting in our model. CNNs are specialized for handling data with a known grid-like structure, such as images, and are known to outperform traditional machine learning models in tasks like image classification and other computer vision challenges.

Dataset
The Describable Textures Dataset (DTD) is comprised of a collection of textured images categorized into 47 distinct classes. The dataset includes 5640 images, each with dimensions of 640x480 pixels. For the purposes of this project, we will resize the images to 224x224 pixels.

Training a CNN
Training a CNN involves several key steps:

Data Preprocessing: This step includes resizing the images to a fixed size and normalizing pixel values to prepare the input data.

CNN Architecture: A CNN consists of convolutional layers, activation functions (such as ReLU), and pooling layers. The network often passes through one or more fully connected layers before producing the final output.

Model Compilation: After defining the CNN architecture, the model is compiled using a loss function, an optimizer, and various metrics.

Model Training: The model is trained using backpropagation and batches of data. The loss is calculated with each epoch and used to adjust the model's parameters.

Model Evaluation: After training, the model's performance is evaluated on a test set to measure its accuracy.

Base Model (Task 1)
The base model is a deep CNN designed for image classification, featuring an output class size of 47 and input dimensions of 224x224 pixels. The model consists of two main components: convolutional layers and fully connected layers. There are three 2D convolutional layers, each with a kernel size of 3x3 and a stride of 1. Each convolutional layer's output dimensions are preserved through the use of padding. The convolutional layers are followed by 2D max-pooling layers and the ReLU activation function. The fully connected layers include two linear layers, with the first transforming the flattened output of the last convolutional layer into a 1D tensor. The second linear layer's output size corresponds to the number of classes (47), with the softmax function applied to determine the final class probabilities. The model has a total of 526,823 trainable parameters.

Result: After 50 epochs, the model achieved a training accuracy of 10.6915% and a testing accuracy of 7.55%.

Underfitting (Task 2)
Underfitting occurs when the model is unable to capture the underlying pattern of the data. This can happen if the model is too simple or if the training duration is too short. In this task, we trained the same model for 20 epochs, which might not have been sufficient for a complex dataset like the DTD. This led to poor performance on both the training and validation sets, with minimal improvement in accuracy and loss metrics.

Result: The training and testing accuracies were 5.0000% and 4.68% respectively, after 20 epochs, indicating underfitting.

Overfitting (Task 3)
Overfitting occurs when the model trains on the training data for too long and starts memorizing it instead of learning to generalize from it. This typically leads to high performance on training data but poor performance on unseen data. In this task, we trained the model for 100 epochs, which led to the model memorizing the training data and losing its ability to generalize, indicated by an increase in training accuracy and a decrease in validation or test accuracy.

