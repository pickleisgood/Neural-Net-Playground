package model;

import java.util.ArrayList;

public class MLP {
    ArrayList<Linear> layers;
    String activation;

    /* REQUIRES: in > 0, out > 0, activation one of "TanH" or "ReLU" or ""
     * EFFECTS: initializes MLP with in, hidden, and out Linear layer dimensions
     * with learning rate (lr), decay rate, and activation function.
    */
    public MLP(int in, ArrayList<Integer> hidden, int out, String activation) {

        this.activation = activation;
        this.layers = new ArrayList<>();

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(in);
        sizes.addAll(hidden);
        sizes.add(out);

        for (int i = 0; i < sizes.size() - 1; i++) {
            layers.add(new Linear(sizes.get(i), sizes.get(i + 1), activation));
        }
    }

    /*
     * EFFECTS: Feeds in input through the layers
    */
    public Value[][] forward(Value[][] in) {
        for (int i = 0; i < layers.size(); i++) {
            // pass in whether the index we are on is the last layer
            // if it is the last layer, use TanH to squash output to (-1, 1)
            in = layers.get(i).forward(in, (i == layers.size() - 1));
        }
        return in;
    }

    public ArrayList<Linear> getLayers() {
        return layers;
    }

    public String getActivation() {
        return activation;
    }

    public ArrayList<Value> getParameters() {
        ArrayList<Value> allParams = new ArrayList<>();

        for (Linear l:layers) {
            allParams.addAll(l.getParameters());
        }

        return allParams;
    }
}

