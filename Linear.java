package model;

import static org.junit.jupiter.api.Assertions.assertEquals;


// A linear layer in a Neural Network
public class Linear {
    Value[][] weights;
    Value[] biases;

    /*
     * EFFECTS: initializes weights and biases to random values
     * the weights is a matrix of shape (in, out)
     * the biases is a vector of shape (out)
    */
    public Linear(int in, int out){
        biases = new Value[out];
        weights = new Value[in][out];
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                weights[i][j] = new Value((double)(Math.random() * 2.0 - 1.0));
            }
        }
        for (int j = 0; j < out; j++) {
            biases[j] = new Value((double)(Math.random() * 2.0 - 1.0));
        }
    }

    public Value[][] forward(Value[][] input){
        assertEquals(input[0].length, weights.length);

        Value[][] out = new Value[input.length][weights[0].length];

        for (int j = 0; j < input.length; j++) {
            for (int i = 0; i < weights[0].length; i++) {
                out[j][i] = biases[i];
                for (int k = 0; k < weights.length; k++) {
                    out[j][i] = out[j][i].add(input[j][k].mul(weights[k][i]));
                }
            }
        }
        return out;
    }

}
