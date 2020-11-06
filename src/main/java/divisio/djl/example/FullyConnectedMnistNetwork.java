package divisio.djl.example;

import ai.djl.nn.Activation;

import java.util.Arrays;

/**
 * A Fully Connected network with hidden sizes 128, 64, one output for each digit,
 * ReLU Activation and a dropout probability of 0.2 during training.
 */
public class FullyConnectedMnistNetwork extends FullyConnectedNetwork {
    public FullyConnectedMnistNetwork() {
        super(Arrays.asList(256, 128, 64), 10, Activation::relu, 0.2f);
    }
}
