package divisio.djl.example;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;

import java.util.List;
import java.util.function.Function;

/**
 * Simple fully connected network. (a.k.a. Multilayer Perceptron, Feedforward Neural Network etc.)
 */
public class FullyConnectedNetwork extends SequentialBlock {

    /**
     * Reshapes the given array, so it only consists of two dimensions, the batch size and the data.
     * E.g. An array shaped like [B, W, H, C] gets turned into [B, W*H*C]
     * Reshaping is an important tool for building complex networks, so it is worth to have a closer look at how this
     * method works.
     * @param ndArray an array with at least two dimensions
     * @return the reshaped array, now with exactly two dimensions
     */
    private NDArray flattenForFullyConnected(final NDArray ndArray) {
        //Get the current shape of the ndArray
        final Shape shape = ndArray.getShape();
        //Get the batch size (the first dimension) of the shape
        final long batchSize = shape.get(0);
        //Get the remaining size
        final long dataSize = shape.size() / batchSize;
        //Reshape the array, so we have one batch dimension and only one for the data
        return ndArray.reshape(batchSize, dataSize);
    }

    /**
     * @param hiddenSizes sizes for the hidden layer outputs
     * @param outputSize output size of the final linear layer
     * @param activationFunction activation function to be used after each hidden layer (not used on the output layer)
     * @param dropoutProbability probabiltiy to use dropout after each hidden layer, if <= 0 no dropout is used
     */
    public FullyConnectedNetwork(final List<Integer> hiddenSizes, final int outputSize, final Function<NDList, NDList> activationFunction, final float dropoutProbability) {
        //An initial reshape makes sure the input is "flat", i.e. it has shape [batchSize, inputSize].
        //We do not need the batch size at the moment, the reshape operation can be given an "unknown" size of -1 and
        //calculate the correct reshaping on the fly when it gets actual input
        add(new LambdaBlock((final NDList ndList) -> new NDList(flattenForFullyConnected(ndList.singletonOrThrow()))));
        //Build hidden layers for each hidden size.
        for (final int hiddenSize : hiddenSizes) {
            add(Linear.builder().setUnits(hiddenSize).optBias(true).build());
            //We apply the activation function after each hidden layer...
            add(new LambdaBlock(activationFunction));
            //...and optionally a dropout layer
            if (dropoutProbability > 0) {
                add(Dropout.builder().optRate(dropoutProbability).build());
            }
        }
        //Attach final output layer, a single linear projection
        add(Linear.builder().setUnits(outputSize).build());
    }
}
