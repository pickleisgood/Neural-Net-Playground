package model;

import java.util.ArrayList;

public class GeneratePoints {
    
    public GeneratePoints() {

    }

    /*
     * EFFECTS: returns a list containing inputs and targets for a
     * regression problem of two deterministically generated 
     * blobs of points in the x,y plane labelled either -1 or 1.
     * 
     * In this case, there is one circular blob on the top right on the
     * plane and another circular blob on the bottom left of the plane.
     * The x, y plane is of size (side * side)  
    */
    @SuppressWarnings("methodlength")
    public ArrayList<Value[][]> twoBlobs(int side) {
        int numPoints = side * side;
        Value[][] inputs = new Value[numPoints][2];
        Value[][] targets = new Value[numPoints][1];
        
        double centerAx = -2.5;
        double centerAy = -2.5; 
        double centerBx = 2.5;
        double centerBy = 2.5;   
        double radiusSq = 4.0; 

        int index = 0;
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {

                // map x and y to [-6, 6] for model input
                double x = ((double) j / (side - 1) * 12.0) - 6.0;
                double y = 6.0 - ((double) i / (side - 1) * 12.0);

                inputs[index][0] = new Value(x);
                inputs[index][1] = new Value(y);

                double distSqA = Math.pow(x - centerAx, 2) + Math.pow(y - centerAy, 2);
                double distSqB = Math.pow(x - centerBx, 2) + Math.pow(y - centerBy, 2);

                if (distSqA < radiusSq) {
                    targets[index][0] = new Value(-1.0); // Inside Circle A
                } else if (distSqB < radiusSq) {
                    targets[index][0] = new Value(1.0);  // Inside Circle B
                }

                index++;
            }
        }

        ArrayList<Value[][]> result = new ArrayList<>();
        result.add(inputs);
        result.add(targets);
        return result;
    }

    /*
     * EFFECTS: returns a list containing inputs and targets for a
     * regression problem of two deterministically generated 
     * blobs of points in the x,y plane labelled either -1 or 1.
     * 
     * In this case, there are four 'blobs' of points one of the top left, 
     * top right, bottom left and bottom right. The blobs diagonaly 
     * opposite to eachother are from the same class.
     * The x, y plane is of size (side * side)  
    */
    @SuppressWarnings("methodlength")
    public ArrayList<Value[][]> fourBlobs(int side) {
        int numPoints = side * side;
        Value[][] inputs = new Value[numPoints][2];
        Value[][] targets = new Value[numPoints][1];
        
        double centerAx = -2.5;
        double centerAy = -2.5; 
        double centerBx = 2.5;
        double centerBy = 2.5;

        double centerCx = -2.5;
        double centerCy = 2.5; 
        double centerDx = 2.5;
        double centerDy = -2.5;
        double radiusSq = 4.0; 

        int index = 0;
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {

                // map x and y to [-6, 6] for model input
                double x = ((double) j / (side - 1) * 12.0) - 6.0;
                double y = 6.0 - ((double) i / (side - 1) * 12.0);

                inputs[index][0] = new Value(x);
                inputs[index][1] = new Value(y);

                double distSqA = Math.pow(x - centerAx, 2) + Math.pow(y - centerAy, 2);
                double distSqB = Math.pow(x - centerBx, 2) + Math.pow(y - centerBy, 2);
                double distSqC = Math.pow(x - centerCx, 2) + Math.pow(y - centerCy, 2);
                double distSqD = Math.pow(x - centerDx, 2) + Math.pow(y - centerDy, 2);

                if (distSqA < radiusSq || distSqB < radiusSq) {
                    targets[index][0] = new Value(-1.0); // Inside Circle A
                } else if (distSqC < radiusSq || distSqD < radiusSq) {
                    targets[index][0] = new Value(1.0);  // Inside Circle B
                }

                index++;
            }
        }

        ArrayList<Value[][]> result = new ArrayList<>();
        result.add(inputs);
        result.add(targets);
        return result;
    }

    /*
     * EFFECTS: returns a list containing inputs and targets for a
     * regression problem of two deterministically generated 
     * blobs of points in the x,y plane labelled either -1 or 1.
     * 
     * In this case, there is one circular blob in the middle of the plane
     * with an outer circle surrounding it.
     * The x, y plane is of size (side * side)  
    */
    @SuppressWarnings("methodlength")
    public ArrayList<Value[][]> concentricCircles(int side) {
        int numPoints = side * side;
        Value[][] inputs = new Value[numPoints][2];
        Value[][] targets = new Value[numPoints][1];
        
        double centerX = 0.0;
        double centerY = 0.0;
        
        double innerRadius = 4.0; // inner circle radius
        double innerOuterRadius = 16.0; // inner radius of outer circle
        double outerOuterRadius = 25.0; // outer radius of outer circle

        int index = 0;
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {

                // map x and y to [-6, 6] for model input
                double x = ((double) j / (side - 1) * 12.0) - 6.0;
                double y = 6.0 - ((double) i / (side - 1) * 12.0);

                inputs[index][0] = new Value(x);
                inputs[index][1] = new Value(y);

                double distSq = Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2);

                if (distSq < innerRadius) {
                    targets[index][0] = new Value(1.0);  // inner circle
                } else if (distSq > innerOuterRadius && distSq < outerOuterRadius) {
                    targets[index][0] = new Value(-1.0); // outer circle
                } 
                
                index++;
            }
        }

        ArrayList<Value[][]> result = new ArrayList<>();
        result.add(inputs);
        result.add(targets);
        return result;
    }

    /**
     * Regression problem of two deterministically generated 
     * interleaving spirals.
     * * Simplified Logic:
     * 1. Twist the space using Sine to create the spiral arms.
     * 2. To keep the arms from getting "fat" at the edges (like a wedge),
     * we shrink the selection threshold as the radius (distSq) increases.
    */
    @SuppressWarnings("methodlength")
    public ArrayList<Value[][]> doubleSpiral(int side) {
        int numPoints = side * side;
        Value[][] inputs = new Value[numPoints][2];
        Value[][] targets = new Value[numPoints][1];
        
        double density = 1.25;   // How much the spiral winds
        double thickness = 0.8; // How thick the lines are

        int index = 0;
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {

                // Map x and y to [-6, 6] for model input
                double x = ((double) j / (side - 1) * 12.0) - 6.0;
                double y = 6.0 - ((double) i / (side - 1) * 12.0);

                inputs[index][0] = new Value(x);
                inputs[index][1] = new Value(y);

                double distSq = x * x + y * y;
                double r = Math.sqrt(distSq);
                double theta = Math.atan2(y, x);
                
                double angle = theta + (r * density);
                double val = Math.sin(angle);

                double threshold = thickness / (distSq/4 + 2.0);

                if (r < 6.0) { // this is max radius
                    if (val > (1.0 - threshold)) {
                        targets[index][0] = new Value(1.0);  // Arm 1 (Sine Peaks)
                    } else if (val < (-1.0 + threshold)) {
                        targets[index][0] = new Value(-1.0); // Arm 2 (Sine Troughs)
                    }
                }
                
                index++;
            }
        }
        ArrayList<Value[][]> result = new ArrayList<>();
        result.add(inputs);
        result.add(targets);
        return result;
    }        
}
