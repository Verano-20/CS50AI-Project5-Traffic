# CS50AI-Project5-Traffic
An AI to identify traffic signs using a neural network with TensorFlow.

For this problem I investigated several variations on the structure of the neural network. This was by no means an exhaustive investigation, and is just a quick look into how some paramters affect the output. To investigate this fully it would make sense to implement a more in depth algorithm as it is an iterative process which could be framed as an optimisation problem.

## Methodology:

First, to enable easy analysis and record keeping, I included two extra parts to main():  
    - A loop to fit and evaluate the model several times and take an average (to better evaluate a given structure).  
    - A method to output the accuracy, loss and NN structure to a csv, for easy analysis and better record keeping.

### model.compile
To keep things relatively simple I used "Adam" optimiser and "categorical_crossentropy" loss when evaluating structures. I have not researched alternatives to any significant extent.

### Structure Testing
The results of all of the tests can be seen in optimise.csv.  
The input layer remained constant; model.add(tf.keras.Input(shape = (30, 30, 3)))  
The output layer remained constant; model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")) where NUM_CATEGORIES = 43.  
A flattening layer before any dense layers remained constant; model.add(tf.keras.layers.Flatten(name="flattened"))

Other layers are altered throughout testing.

#### Dense Layers Only

# Test 1
Optimise the number of units in 1, 2, 10 and 100 identical dense layers from 1 to 1000.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test1-structure.png" alt="Test 1 Structure" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/Test1-Accuracy.png" alt="Test 1 Accuracy Graph" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/Test1-Loss.png" alt="Test 1 Loss Graph" />

We can see clearly from test 1 that when only using dense layers, changing the number of layers and number of units in those layers (if the layers are identical) has no significant effect on the accuracy or the loss. When using only 1 dense layer, there is a very slight trend towards decreased accuracy and increased loss as the number of units increases.

#### Introduce a Conv2D layer

# Test 2
Start with no Dense layers and a single Conv2D layer with 16 filters and 1x1, 3x3 and 5x5 kernel sizes.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test2-structure.png" alt="Test 2 Structure" />

1x1: Accuracy = 5.01%, Loss = 14.2  
3x3: Accuracy = 4.66%, Loss = 13.4  
5x5: Accuracy = 5.46%, Loss = 12.5  

Conclusion: 5x5 marginally better than others, but not conclusive. Will continue testing with a 3x3 kernel.

#### Introduce a Max Pooling layer

# Test 3
Keeping a single Conv2D layer with 16 filters and a 3x3 kernel size, introduce a Max Pooling layer with a 2x2, 4x4, and 6x6 pool size.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test3-structure.png" alt="Test 3 Structure" />

2x2: Accuracy = 5.52%, Loss = 12.2  
4x4: Accuracy = 0.58%, Loss = 1.19e-7  
6x6: Accuracy = 5.77%, Loss = 7.85  

A strange drop in accuracy with the 4x4 pool size, but no significant difference in accuracy between the 2x2 and 6x6. Between these, the 6x6 had the lower loss.

#### Reintroduce Dense layers

From previous we can see that individually, these layers are not enough to give a good result. If combined with a Conv2D and a Max Pooling layer, we may see better results.

# Test 4
With a single Conv2D layer with 16 filters and a 3x3 kernel size, and a Max Pooling layer with a 6x6 pool size, optimise the number of units in 1, 10 and 100 identical dense layers from 1 to 1000.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test4-structure.png" alt="Test 4 Structure" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/mediaTest4-Accuracy.png" alt="Test 4 Accuracy Graph" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/Test4-Loss.png" alt="Test 4 Loss Graph" />

Adding a single dense layer enables accuracy to reach ~93%. While the number of units is 10 or below accuracy is ~5%, but when the number of units reaches 30 accuracy jumps to ~82%, and by 250 units accuracy has peaked at ~93%.

Interestingly, with 10 dense layers accuracy begins to increase and loss begins to decrease as the number of units in each layer approaches 70, and then accuracy drops and loss increases to the values they held with only a single unit in each layer, remaining at this level as the number of units increases to 1000. It is unclear why this is the case, or if it is an error in the result.

With 100 dense layers, the accuracy never increased significantly and the loss never decreased significantly regardless of the number of units in each layer.

It is clear then that more is not necessarily better, and that there must exist a sweet spot.

#### Optimal size for single hidden layer

# Test 5
Repeat test 4 for 1 dense layer with a finer iteration on the number of units in the dense layer. Find more precisely where the peak accuracy is.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test5-structure.png" alt="Test 5 Structure" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/Test5.png" alt="Test 5 Graph" />

The accuracy and loss are much more volatile at lower numbers of units, and the improvement tapers off as the number of units reaches 200. The maximum accuracy in this test was 92.56% which occurred with 190 units in the single dense layer.

#### Introduce a second Conv2D layer and Max Pooling layer.

# Test 6
Using the single dense layer with 190 units as described in Test 5, add another Conv2D layer with 32 filters and a 3x3 kernel size, and another Max Pooling layer with a 2x2 pool size.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/test6-structure.png" alt="Test 6 Structure" />

For the test, the model was trained and evaluated 10 times before an average was taken, as opposed to 5 times for every previous test. This is to improve accuracy as it is a single test. The benchmark of a single Conv2D and Max pooling layer with a single dense layer with 190 units as described in Test 5 is repeated here, again with 10 reps for equivalent accuracy.

One Conv2D and One Max Pool layer:  
Accuracy = 92.77%  
Loss = 0.24

Two Conv2D and Two Max Pool layers:  
Accuracy = 90.68%  
Loss = 0.33

Adding these two layers does not lead to an improvement in performance. There may be variations on the parameters of these layers that would give an improvement, but that is beyond the scope of this investigation.

#### Conclusion

To conclude, the best model I found from this investigation was as follows:

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/media/final-structure.png" alt="Final Structure" />

-A Conv2D layer with 16 filters and a 3x3 kernel size  
-A Max Pooling layer with a 6x6 pool size  
-A flattening layer to 1D  
-A Dense layer with 190 units  
-A Dense layer with 43 units for the output