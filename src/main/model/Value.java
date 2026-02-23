package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

// A single scalar value with automatic gradient calculation
public class Value {
    double data; // the scalar value
    double grad; // the gradient
    ArrayList<Value> children;


    private Runnable backward = () -> { }; // action to perform during backpropagation
    
    /*
     * EFFECTS: sets the data to the given value and initializes gradient to 0
     * adds children to this.children if not null
    */
    public Value(double data, Value... children) {
        this.data = data;
        this.grad = 0.0;
        this.children = new ArrayList<>(Arrays.asList(children));
    }
    
    public double getData() {
        return this.data;
    }

    public void setData(double data) {
        this.data = data;
    }
    
    public double getGrad() {
        return this.grad;
    }
    
    public void setGrad(double grad) {
        this.grad = grad;
    }

    public ArrayList<Value> getChildren() {
        return this.children;
    }

    /*
     * EFFECTS: creates an output Value who's data is data exponentiated
     * to a given power and who's children are only this
     */
    public Value pow(int power) {
        double newData = Math.pow(this.data, power);
        Value out = new Value(newData, this);

        /*
        * MODIFIES: this
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this
        */
        out.backward = () -> {
            this.grad += power * Math.pow(this.data, power - 1) * out.grad;
        };
        return out;
    }

    /*
     * EFFECTS: creates an output Value who's data is TanH(this.data)
     * and who's children are only this
     */
    public Value tanh() {
        double newData = Math.tanh(data);
        Value out = new Value(newData, this);

        /*
        * MODIFIES: this
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this
        */
        out.backward = () -> {
            this.grad += (1 - Math.pow(out.data, 2)) * out.grad; 
        };
        return out;
    }

    /*
     * EFFECTS: creates an output Value who's data is max(0, this.data)
     * and who's children are only this
     */
    public Value relu() {
        double newData = Math.max(0.0, data);
        Value out = new Value(newData, this);

        /*
        * MODIFIES: this
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this
        */
        out.backward = () -> {
            this.grad += ((out.data > 0) ? 1.0 : 0.0) * out.grad; 
        };
        return out;
    }
    /*
     * EFFECTS: creates an output Value who's data is the product 
     * the data of this and other and whos children are this and other
     * 
     */

    public Value mul(Value other) {
        double newData = this.data * other.data;
        Value out = new Value(newData, this, other);

        /*
        * MODIFIES: this, other
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this and other
        */
        out.backward = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }
    
    /*
     * EFFECTS: creates an output Value who's data is the sum 
     * the data of this and other and whos children are this and other
     */
    public Value add(Value other) {
        double newData = this.data + other.data;
        
        Value out = new Value(newData, this, other);
        
        /*
        * MODIFIES: this, other
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this and other
        */
        out.backward = () -> {
            this.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    /*
     * EFFECTS: returns a list of Values in order for backpropagation
     * List returned is in reversed order
     */
    public ArrayList<Value> buildTopo() {
        ArrayList<Value> topo = new ArrayList<>();
        HashSet<Value> visited = new HashSet<>();
        
        // Internal recursive helper
        buildTopoRecursive(this, visited, topo);
        
        return topo;
    }

    /*
     * EFFECTS: recursively iterates through the graph of calculations
     * and adds children that are not visited
     */
    private void buildTopoRecursive(Value v, HashSet<Value> visited, ArrayList<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.children) {
                buildTopoRecursive(child, visited, topo);
            }
            topo.add(v);
        }
    }

    /*
     * EFFECTS: backpropagate gradients through calculation graph
     */
    public void backward() {
        ArrayList<Value> topo = buildTopo();

        this.setGrad(1.0);
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i).backwardAction();
        }
    }

    /*
     * EFFECTS: backpropagate gradients to children
     */
    public void backwardAction() {
        this.backward.run();
    }
}