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
&nbsp;&nbsp;&nbsp;1. Input Layer<br>
&nbsp;&nbsp;&nbsp;2. Hidden Layers (>=1, say L)<br>
&nbsp;&nbsp;&nbsp;3. Output Layer (=1)<br>
The network collectively is called (L+1) Layer Neural Network<br>
![_config.yml]({{ site.baseurl }}/images/layer_by_layer.jpg)<br>
Each successive hidden layer is capable of computing complex features from the given input and its computation is comparatively more complex than its predecessor layers.<br>
We also have a 1 Layer Neural Network which is also called **Logistic Regression**, but is not as good as compared to a DNN.<br> 

## Under The Hood

The above confusing Neural Network image can be demystified in following steps:<br>
As usual you will follow the Deep Learning methodology to build the model:<br>
    1. Aquiring Dataset
    2. Data Preprocessing and Building Utilities.
    3. Initialize parameters / Define hyperparameters<br>
    4. Loop for num_iterations:<br>
    &nbsp;&nbsp;&nbsp;a. Forward propagation<br>
    &nbsp;&nbsp;&nbsp;b. Compute cost function<br>
    &nbsp;&nbsp;&nbsp;c. Backward propagation<br>
    &nbsp;&nbsp;&nbsp;d. Update parameters (using parameters, and grads from backprop) <br>
    5. Use trained parameters to predict labels<br>
Let's take each step and expand upon the complete process.<br>
We will be usnig **Python** as our programming language combined with **numpy** for our mathematical computations.<br>

Reference can be taken from following links for any issues : <br>
[Python](https://www.python.org/)
[Numpy](http://www.numpy.org/)

### Dataset
To Demonstrate the power of Deep Neural Network we will try to Build an Image Classifier(Cat vs Non Cat). The dataset will consist of:<br>
 &nbsp;&nbsp;&nbsp;1. Training (13 Images)<br>
 &nbsp;&nbsp;&nbsp;2. Testing (24 Images)<br>
On the same dataset we will compare it to one using **Logistic Regression** model.
 
### Building Utilties

![_config.yml]({{ site.baseurl }}/images/img_vec.png)<br>
Any image cannot be directly fed into the Neural Network ans computer understands numbers. So the image must be converted into numerical form so that it can be feeded into the DNN. This conversion of Image to numerical data is called **Image Vectorization**.<br>
A color image basically has three components.<br>
&nbsp;&nbsp;&nbsp;1. Height(ht)<br>
&nbsp;&nbsp;&nbsp;2. Width(wd)<br>
&nbsp;&nbsp;&nbsp;3. Color Channels(**RGB** = 3)<br>
So the image has to be converted into a shape: **(ht, wd, 3)**.<br>
We will have **m**(13 in our case) training examples so final matrix shape that will be feeded as the **Input Layer** becomes: **(m, ht, wd, 3)**.<br>
The following code snippet demonstrate the above explained process.<br>
![_config.yml]({{ site.baseurl }}/images/utils.png)<br>
Components of code are: <br>
**image_to_arr**<br>
&nbsp;Takes list of all images and converts them into a (m, 64, 64, 3) matrix, special care must be taken of dimensions and datatype stored in the numpy arrays.<br>
**gen_labels**<br>
&nbsp; Takes the labels of the images converted to numpy arrays and genrates output labels for them (0=Non_Cat, 1= Cat). Shape of array must be (1,m).<br>
**load_image**<br>
&nbsp; Takes both functions and computes the result for training and testing images and returns the requisite numpy arrays as output.


 



