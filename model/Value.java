package model;

// A single scalar value with automatic differentiation
public class Value {
    double data; // the scalar value
    double grad; // the gradient

    private Runnable backwardAction = () -> {}; // action to perform during backpropagation

    
    /*
     * EFFECTS: sets the data to the given value and initializes gradient to 0
    */
    public Value(double data) {
        this.data = data;
        this.grad = 0.0;
    }
    
    public double getData() {
        return this.data;
    }
    
    public double getGrad() {
        return this.grad;
    }
    
    public void setGrad(double grad) {
        this.grad = grad;
    }

    /*
     * EFFECTS: creates an output Value who's data is the product 
     * the data of this and other
     */
    public Value mul(Value other) {
        double new_data = this.data * other.data;
        Value out = new Value(new_data);

        /*
        * MODIFIES: this, other
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this and other
        */
        out.backwardAction = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }
    
    /*
     * EFFECTS: creates an output Value who's data is the sum 
     * the data of this and other
     */
    public Value add(Value other) {
        double new_data = this.data + other.data;
        Value out = new Value(new_data);
        
        /*
        * MODIFIES: this, other
        * EFFECTS: sets the backwardAction of out to update 
        * the gradients of this and other
        */
        out.backwardAction = () -> {
            this.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }
}