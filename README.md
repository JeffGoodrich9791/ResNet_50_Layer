## Residual Network 50 Layer 
### Building A Residual Network for image recognition of SIGNS Dataset

### Summary

Very deep neural networks do not work because they suffer from the problem of "vanising gradients" or "exploding gradients". The Residual Network (ResNet) proposed by He et al. (2015) utilizes "shortcut" connections of identity blocks and convolutional blocks skipping every two or three layers in the main path of the network. In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers. This approach not only addresses the Vanishing Gradient problem, but also makes it easier to train while lowering training and test set errors. This project utilizes a 50 layer ResNet to capture image recognition of the SIGNS dataset.  

### Code

Template code is provided in the `ResNet_50_Layer.ipynb` notebook file. The layers of the network were constructed and compiled using Keras in an iPyton Notebook. The identity block was first constructed with 3 components of the main path then adding a shortcut value to pass through a ReLU activation. The first two components consists of Conv2D, BatchNorm, and ReLU activation while the third component has only Conv2D and BatchNorm, with the RelU activation included in the shortcut path component. 

<img src= "https://github.com/JeffGoodrich9791/ResNet_50_Layer/blob/master/Identity Block.png" />

The first three components of the convolutional block is constructed exactly as the identity block structure. The shortcut component consist of Conv2D as well as BatchNorm, then it is added to the main path and passed through a ReLU activation function. 

<img src= "https://github.com/JeffGoodrich9791/ResNet_50_Layer/blob/master/Convolutional Block.png" />

### Run

In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:

```ipython notebook Logistic Support_Vector_Machine.ipynb```  
```jupyter notebook Logistic Support_Vector_Machine.ipynb```

This will open the iPython Notebook software and project file in your browser.

### Results

Support Vector Machine with a non-linear "rbf" kernel produced a prediction accuracy of 93% on 100 training examples
