---
layout: post
title: Introduction to Deep Learning
---

## Introduction

![_config.yml]({{ site.baseurl }}/images/DNN_img.png)

If you have reached here, then i assume that you have atleast heard once about Deep Learning. But actually what is it? This is the point where we lose it. Time and again we hear about the groundbreaking researchs being done in this field, products being built by tech giants like Google, Microsoft, Apple, Amazon and many more. What does actually go under the hood of these awesome projects? <br>
The basic underline of Deep Learning is Neural Network, not just any Neural Network but **Deep Neural Network (DNN)**. It looks like this.

![_config.yml]({{ site.baseurl }}/deep_neural_network.png)

Looks frightening isn't it. But actually is not that complex to understand. Little amount of **linear algebra** and **calculus** coupled with knowledge of a programming language and we are good to go forward with it. However, prior knowledge of the mathematical stuff is not a necessary condition.

## Motivation

Tech giants are competing and pushing the boundaries of technology everyday and Deeplearning being the hottest field for advancement has certainly caught eye of these blue chip companies.

![_config.yml]({{ site.baseurl }}/images/deepmind.jpg)
![_config.yml]({{ site.baseurl }}/images/facebook.png)
![_config.yml]({{ site.baseurl }}/images/openAI.jpg)

Day after day revolutionary engineering has resulted in many groundbreaking technologies ranging from medical, entertainment to Gaming. These huge successes and many more to come, interests us to delve into this field. With such motivation we must continue on our journey to learn about the very foundation of this hugely popular terminology "Deep Learning".

## What are Deep Neural Networks 
Deep Neural Networks are nothing but stacking of layers of **neurons** one over the other. A Neurons is a unit of Network which performs a set of computation and passes on the output to next layer where the same process is repeated again.
![_config.yml]({{ site.baseurl }}/images/network.png)
As you can see in above image we have a neuron computing two components and forwarding the output to the next layer where similar computation is repeated.<br>
A **DNN** primarily consists of three components:<br>
+ Input Layer<br>
+ Hidden Layers (>=1, say L)<br>
+ Output Layer (=1)<br>
The network collectively is called (L+1) Layer Neural Network<br>
![_config.yml]({{ site.baseurl }}/images/layer_by_layer.jpg)<br>
Each successive hidden layer is capable of computing complex features from the given input and its computation is comparatively more complex than its predecessor layers.<br>
We also have a 1 Layer Neural Network which is also called **Logistic Regression**, but is not as good as compared to a DNN.<br> 

## Under The Hood
The above confusing Neural Network image can be demystified in following steps:<br>
As usual you will follow the Deep Learning methodology to build the model:<br>
+ Aquiring Dataset<br>
+ Data Preprocessing and Building Utilities.<br>
+ Build The DNN<br>
+ Initialize parameters / Define hyperparameters<br>
+ Loop for num_iterations:<br>
&nbsp;&nbsp;&nbsp;a. Forward propagation<br>
&nbsp;&nbsp;&nbsp;b. Compute cost function<br>
&nbsp;&nbsp;&nbsp;c. Backward propagation<br>
&nbsp;&nbsp;&nbsp;d. Update parameters (using parameters, and grads from backprop) <br>
+ Use trained parameters to predict labels<br>
Let's take each step and expand upon the complete process.<br>
We will be usnig **Python** as our programming language combined with **numpy** for our mathematical computations.<br>

Reference can be taken from following links for any issues : <br>
[Python](https://www.python.org/)
[Numpy](http://www.numpy.org/)

### Dataset
To Demonstrate the power of Deep Neural Network we will try to Build an Image Classifier(Cat vs Non Cat). The dataset will consist of:<br>
* Training (13 Images)<br>
* Testing (24 Images)<br>
On the same dataset we will compare it to one using **Logistic Regression** model.
 
### Building Utilties

![_config.yml]({{ site.baseurl }}/images/img_vec.png)<br>
Any image cannot be directly fed into the Neural Network ans computer understands numbers. So the image must be converted into numerical form so that it can be feeded into the DNN. This conversion of Image to numerical data is called **Image Vectorization**.So the image has to be converted into a shape: **(ht, wd, 3)**.<br>
We will have **m**(13 in our case) training examples so final matrix shape that will be feeded as the **Input Layer** becomes: **(m, ht, wd, 3)**.<br>
A color image basically has three components.<br>
* Height(ht)<br>
* Width(wd)<br>
* Color Channels(**RGB** = 3)<br>

The following code snippet demonstrate the above explained process.<br>
![_config.yml]({{ site.baseurl }}/images/utils.png)<br>
Components of code are: <br>
* **image_to_arr**<br>
Takes list of all images and converts them into a (m, 64, 64, 3) matrix, special care must be taken of dimensions and datatype stored in the numpy arrays.<br>
* **gen_labels**<br>
Takes the labels of the images converted to numpy arrays and genrates output labels for them (0=Non_Cat, 1= Cat). Shape of array must be (1,m).<br>
* **load_image**<br>
Takes both functions and computes the result for training and testing images and returns the requisite numpy arrays as output.<br>

Work not done yet!<br>
Need to create one more utility file which contains:<br>
* **sigmoid_activation**: The forward activation function for the Output layer.<br>
* **relu_activation**: The forward activation function for hidden layers.<br>
* **sigmoid_derivative**: The derivative of sigmoid function for backpropagation.<br>
* **relu_derivative** : The derivative of relu function for backpropagation.<br>

Python code for the utility can be seen below:<br>
![_config.yml]({{ site.baseurl }}/images/L_DNN_utils.png)<br>

### Data Preprocessing
Load pre-requisite dependencies<br>
![_config.yml]({{ site.baseurl }}/images/1.png)<br>

Load the data set<br>
![_config.yml]({{ site.baseurl }}/images/2.png)<br>

The data that we work with is loaded using the Utility we built in previous step. But, furthermore preprocessing is needed before we include it in our computation. To achieve fast computation results we will use a process called **Vectorization** using numpy. Data preprocessing is a necessary step for that.
<br>
Steps involved in preprocessing are:<br>
* **Array Flattening**: Converting input data shape from **(m, ht, wd, 3 )** to **(ht * wd * 3, m)**<br>
* **Data Standardization**: Dividing every value in matrix from 255 (255 being the max value in the input matrix)<br>

![_config.yml]({{ site.baseurl }}/images/5.png)<br>


### Building the Deep Neural Network

A Deep Neural Network has following components.<br>
**Inputs (X)**: The input matrix provided to the DNN as training and testing datasets.<br>

**Hidden Layers**: Each hidden layer is given a task to compute **Forward Propagation** variables<br>
* **Z (W * X + b)**: In this step we compute linear outputs corresponding to **X** but combine<br> &nbsp;&nbsp;&nbsp;it with **W**(weights assignmed to each Hidden Layer) with an added bias value **b**.<br>
*  **A ( g(Z) )**: In this step we compute **activations** for our computed linear outputs so as<br> &nbsp;&nbsp;&nbsp;to obtain some **non-linearity** in our learning (This is an important aspect of Neural Networks).<br>

**Output Layer**: The output layer is responsible to compute the final output values ( 0/1 ).

**Dimensions**: A lot of care must go into keeping a check on the dimensional integrity of the variables and matrices we are computing. Below is a quick guide to for what the dimensions of these computations must be.<br>
![_config.yml]({{ site.baseurl }}/images/dimensions.png)<br>

### Activations
Activations are functions that must be applied to computed linear variables (Z) so as to obtain non-linearity.<br>
**Why non-linearity ?** It helps the neural network compute interesting features. **Linear** hidden layers are useless.<br>
Types of activations that we use in our DNN are:<br><br>
**Sigmoid**<br>
![_config.yml]({{ site.baseurl }}/images/sigmoid.png)<br>
One of the famous activation functions. **MUST BE USED IN OUTPUT LAYER**<br><br>
**ReLU**<br>
![_config.yml]({{ site.baseurl }}/images/relu.png)<br>
Also called rectified Linear Unit. To calculate interesting features **MUST BE USED IN HIDDEN LAYERS**.

### Componenents of DNN model

**Initialize Parameters**<br>
The main components of the Network are the parameters which will govern the performance and learning from data. These parameters are the inputs to a **neuron** which facilitate the formation output. However there initialisation is a bit different from each other. These components include:<br>
**Weights(W)**<br>
Initialized as a random array of dimensions (n[L], n[L-1]) --> see dimensions image above for consultance. This random initialisation is due to the fact that if weights are initialized to 0 or some fixed value then all the neurons in same layer will be computing same values for features with no improvements which will result in symmetric Network(undesirable).<br>
**Bias(b)**<br>
Initialized as a numpy array of zeros (n[L],1).<br><br>
The following can be implemented in python as follows<br>
![_config.yml]({{ site.baseurl }}/images/Model2.png)<br>

### Structure of our Neural Network
![_config.yml]({{ site.baseurl }}/images/DNN_structure.png)<br>
The image above demonstrates the exact structure that we are going to implement.<br>
+ We will have in total **L** layers in our DNN and out of these **L-1** will be hidden layers.<br>
+ An outermost loop will run for **num_iteration** times which is as per user requirements. <br>
+ First L-1 layers will compute **forward propagation** variables and store them in cache for future use. We will use **ReLU** activation in this layer.<br>
+ Last layer will compute the same variables but activation used in this layer will be **sigmoid**.<br>
+ The output of last layer(L) will be used to compute **loss** of our model. This loss will help us initiate the **backpropagation** mechanism.<br>
+ The backpropagation will take help of utilities built earlier and caches maintained during forward propagation to compute the gradients necessary for tuning our model parameters.<br>
+ The gradients computed via backpropagation will be used to update our parameters. 
+ Running above propagation steps will enable proper tuning of parameters which can then be used in **prediction** of output on test data.


### Forward Propagation
The first step step is to propagate inside our neural network skeleton built by initialising the parameeters mentioned above. In this step we compute the forward linear function(Z) for each neuron and their respective activations(A). 















 



