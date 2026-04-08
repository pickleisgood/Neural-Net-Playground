package model;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Random;

// A linear layer in a Neural Network
public class Linear {

    private static final long SEED = 42;
    private static final Random rand = new Random();

    Value[][] weights;
    Value[] biases;
    String activation;

    /* REQUIRES: in > 0, out > 0, activation one of "TanH" or ""
     * EFFECTS: initializes weights and biases to random values
     * the weights is a matrix of shape (in, out)
     * the biases is a vector of shape (out)
    */
    public Linear(int in, int out, String activation) {
        this.activation = activation;
        biases = new Value[out];
        weights = new Value[in][out];

        rand.setSeed(SEED);
        double k = Math.sqrt(1.0 / in);

        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                weights[i][j] = new Value((double)(rand.nextDouble() * 2.0 - 1.0) * k);
            }
        }
        for (int j = 0; j < out; j++) {
            biases[j] = new Value((double)(rand.nextDouble() * 2.0 - 1.0) * k);
        }
    }

    /*
     * EFFECTS: Performs the forward pass on the input and returns
     * the output from that pass
    */
    public Value[][] forward(Value[][] input, boolean isLast) {
        assertEquals(input[0].length, weights.length);

        Value[][] out = new Value[input.length][weights[0].length];

        for (int j = 0; j < input.length; j++) {
            for (int i = 0; i < weights[0].length; i++) {
                out[j][i] = biases[i];
                for (int k = 0; k < weights.length; k++) {
                    out[j][i] = out[j][i].add(input[j][k].mul(weights[k][i]));
                }
                // if last layer, use TanH, otherwise use specified activation
                if (!isLast) {
                    out[j][i] = handleActivation(out[j][i], this.activation);
                } else {
                    out[j][i] = handleActivation(out[j][i], "TanH");
                }
            }
        }
        return out;
    }

    /*
     * MODIFIES: v
     * EFFECTS: If there is an activation function present, perform it onto
     * the value
    */
    public Value handleActivation(Value v, String activation) {
        if (activation.equals("None")) {
            return v;
        } else if (activation.equals("TanH")) {
            return v.tanh().mul(new Value(5.0 / 3.0)); // apply a gain of 5/3x
        } else {
            return v.relu().mul(new Value(Math.sqrt(2.0)));
        }
    }

    public Value[][] getWeights() {
        return weights;
    }

    public Value[] getBiases() {
        return biases;
    }

    public String getActivation() {
        return activation;
    }

    /*
    * EFFECTS: returns a list of all Value objects (weights and biases) 
    * in this layer that need to be updated during training
    */
    public ArrayList<Value> getParameters() {
        ArrayList<Value> allParams = new ArrayList<>();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                allParams.add(weights[i][j]);
            }
        }

        for (int i = 0; i < biases.length; i++) {
            allParams.add(biases[i]);
        }

        return allParams;
    }

}
