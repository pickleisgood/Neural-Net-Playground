package model;

import persistence.Writable;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.ArrayList;

// An MLP representing a list of linear layers
public class MLP implements Writable {
    ArrayList<Linear> layers;
    String activation;
    double lr;
    String dataset;

    private int inDim;
    private ArrayList<Integer> hiddenDims;
    private int outDim;

    /* REQUIRES: in > 0, out > 0, activation one of "TanH" or "ReLU" or "None"
     * EFFECTS: initializes MLP with in, hidden, and out Linear layer dimensions
     * with learning rate (lr), decay rate, and activation function.
    */
    public MLP(int in, ArrayList<Integer> hidden, int out, String activation) {
        EventLog.getInstance().logEvent(
                new Event("Deleting and Reinitializing Model"));
        this.layers = new ArrayList<>();
        this.activation = activation;

        this.inDim = in;
        this.hiddenDims = new ArrayList<>(hidden);
        this.outDim = out;

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(in);
        sizes.addAll(hidden);
        sizes.add(out);

        for (int i = 0; i < sizes.size() - 1; i++) {
            layers.add(new Linear(sizes.get(i), sizes.get(i + 1), activation));
            EventLog.getInstance().logEvent(
                new Event("Added linear layer of size " + sizes.get(i + 1)));
        }
    }

    /*
     * EFFECTS: Feeds in input through the layers
    */
    public Value[][] forward(Value[][] in) {
        // EventLog.getInstance().logEvent(
        //         new Event("Initiating forward pass on layers"));
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

    public ArrayList<Integer> getHiddenDims() {
        return hiddenDims;
    }

    //REQUIRES: lr > 0
    // MODIFIES: this
    // EFFECTS: saves learning rate and dataset used for JSON saving
    public void setSavingState(double lr, String dataset) {
        this.lr = lr;
        this.dataset = dataset;
        EventLog.getInstance().logEvent(
                new Event("Saved MLP State"));
    }

    public double getLr() {
        return lr;
    }

    public String getDataset() {
        return dataset;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        json.put("in", inDim);
        json.put("out", outDim);
        json.put("activation", activation);
        json.put("lr", lr);
        json.put("dataset", dataset);

        JSONArray hiddenJson = new JSONArray();
        for (int h : hiddenDims) {
            hiddenJson.put(h);
        }
        json.put("hidden", hiddenJson);

        JSONArray paramsJson = new JSONArray();
        for (Value p : this.getParameters()) {
            paramsJson.put(p.getData());
        }
        json.put("parameters", paramsJson);

        return json;
    }
}

