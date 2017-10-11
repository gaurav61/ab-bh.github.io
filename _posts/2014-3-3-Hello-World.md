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

## Under The Hood

The above confusing Neural Network image can be demystified in following steps:<br>
As usual you will follow the Deep Learning methodology to build the model:<br>
    1. Data Preprocessing and Building Utilities.
    2. Initialize parameters / Define hyperparameters<br>
    3. Loop for num_iterations:<br>
    &nbsp;&nbsp;&nbsp;a. Forward propagation<br>
    &nbsp;&nbsp;&nbsp;b. Compute cost function<br>
    &nbsp;&nbsp;&nbsp;c. Backward propagation<br>
    &nbsp;&nbsp;&nbsp;d. Update parameters (using parameters, and grads from backprop) <br>
    4. Use trained parameters to predict labels<br>
Let's take each step and expand upon the complete process.<br>
We will be usnig **Python** as our programming language combined with **numpy** for our mathematical computations.<br>

Reference can be taken from following links for any issues : <br>
[Python](https://www.python.org/)
[Numpy](http://www.numpy.org/)



