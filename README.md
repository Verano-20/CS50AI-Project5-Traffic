# CS50AI-Project5-Traffic
An AI to identify traffic signs using a neural network with TensorFlow.

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

Test 1 - Optimise the number of units in 1, 2, 10 and 100 identical dense layers from 1 to 1000.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/Test1-Accuracy.png" alt="Test 1 Accuracy Graph" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/Test1-Loss.png" alt="Test 1 Loss Graph" />

We can see clearly from test 1 that when only using dense layers, changing the number of layers and number of units in those layers (if the layers are identical) has no significant effect on the accuracy or the loss. When using only 1 dense layer, there is a very slight trend towards decreased accuracy and increased loss as the number of units increases.

#### Introduce a Conv2D layer

Test 2 - Start with no Dense layers and a single Conv2D layer with 32 filters and 1x1, 3x3 and 5x5 kernel sizes.

1x1: Accuracy = 5.01%, Loss = 14.2
3x3: Accuracy = 4.66%, Loss = 13.4
5x5: Accuracy = 5.46%, Loss = 12.5

Conclusion: 5x5 marginally better than others, but not conclusive. Will continue testing with a 3x3 kernel.

#### Introduce a Max Pooling layer

Test 3 - Keeping a single Conv2D layer with 32 filters and a 3x3 kernel size, introduce a Max Pooling layer with a 2x2, 4x4, and 6x6 pool size.

2x2: Accuracy = 5.52%, Loss = 12.2
4x4: Accuracy = 0.58%, Loss = 1.19e-7
6x6: Accuracy = 5.77%, Loss = 7.85

A strange drop in accuracy with the 4x4 pool size, but no significant difference in accuracy between the 2x2 and 6x6. Between these, the 6x6 had the lower loss.

#### Reintroduce Dense layers

From previous we can see that individually, these layers are not enough to give a good result. If combined with a Conv2D and a Max Pooling layer, we may see better results.

Test 4 - Including with a single Conv2D layer with 32 filters and a 3x3 kernel size, and a Max Pooling layer with a 6x6 pool size, optimise the number of units in 1, 10 and 100 identical dense layers from 1 to 1000.

<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/Test4-Accuracy.png" alt="Test 4 Accuracy Graph" />
<img src="https://github.com/Verano-20/CS50AI-Project5-Traffic/blob/master/Test4-Loss.png" alt="Test 4 Loss Graph" />

Adding a single dense layer enables accuracy to reach ~93%. While the number of units is 10 or below accuracy is ~5%, but when the number of units reaches 30 accuracy jumps to ~82%, and by 250 units accuracy has peaked at ~93%.

Interestingly, with 10 dense layers accuracy begins to increase and loss begins to decrease as the number of units in each layer approaches 70, and then accuracy drops and loss increases to the values they held with only a single unit in each layer, remaining at this level as the number of units increases to 1000. It is unclear why this is the case, or if it is an error in the result.

With 100 dense layers, the accuracy never increased significantly and the loss never decreased significantly regardless of the number of units in each layer.

It is clear then that more is not necessarily better, and that there must exist a sweet spot.