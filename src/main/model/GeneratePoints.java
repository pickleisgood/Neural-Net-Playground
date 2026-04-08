package model;

import java.util.ArrayList;
import java.util.Arrays;

/*
* Generates various geometric point patterns based on a provided type string.
*/
public class GeneratePoints {

    public GeneratePoints() {
    }

    /*
     * REQUIRES: type is one of "twoBlobs", "fourBlobs", 
     * "concentricCircles", "double spiral"
     * EFFECTS: returns a list containing inputs and targets for the specified 
     * regression problem (twoBlobs, fourBlobs, concentricCircles, doubleSpiral).
     * The x, y plane is of size (side * side).
     * Returns empty dataset if the type passed in is unsupported
     */
    public ArrayList<Value[][]> getDataset(int side, String type) {
        Value[][] inputs = new Value[side * side][2];
        Value[][] targets = new Value[side * side][1];

        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {
                int idx = i * side + j;
                
                // map x and y to [-6, 6] for model input
                double x = ((double) j / (side - 1) * 12.0) - 6.0;
                double y = 6.0 - ((double) i / (side - 1) * 12.0);

                inputs[idx][0] = new Value(x);
                inputs[idx][1] = new Value(y);

                Double label = getLable(type, x, y);

                if (label != null) {
                    targets[idx][0] = new Value(label);
                }
            }
        }
        EventLog.getInstance().logEvent(
                new Event("Generate dataset of type: " + type));
        return new ArrayList<>(Arrays.asList(inputs, targets));
    }

    /*
     * REQUIRES: type is one of "twoBlobs", "fourBlobs", 
     * "concentricCircles", "double spiral"
     * EFFECTS: returns the label of an x, y point given type.
     */
    public Double getLable(String type, double x, double y) {
        Double label = null;
        switch (type) {
            case "twoBlobs":
                label = getTwoBlobsLabel(x, y);
                break;
            case "fourBlobs":
                label = getFourBlobsLabel(x, y);
                break;
            case "concentricCircles":
                label = getConcentricCirclesLabel(x, y);
                break;
            case "doubleSpiral":
                label = getDoubleSpiralLabel(x, y);
                break;
            default:
                label = null;
                break;
        }
        return label;
    }

    /*
     * Logic for twoBlobs: one circle top-right, one bottom-left.
     */
    private Double getTwoBlobsLabel(double x, double y) {
        if (distSq(x, y, -2.5, -2.5) < 4.0) {
            return -1.0;
        }
        if (distSq(x, y, 2.5, 2.5) < 4.0) {
            return 1.0;
        }
        return null;
    }

    /*
     * Logic for fourBlobs: diagonally opposite blobs are same class.
     */
    private Double getFourBlobsLabel(double x, double y) {
        if (distSq(x, y, -2.5, -2.5) < 4.0 || distSq(x, y, 2.5, 2.5) < 4.0) {
            return -1.0;
        }
        if (distSq(x, y, -2.5, 2.5) < 4.0 || distSq(x, y, 2.5, -2.5) < 4.0) {
            return 1.0;
        }
        return null;
    }

    /*
     * Logic for concentricCircles: inner circle and an outer ring.
     */
    private Double getConcentricCirclesLabel(double x, double y) {
        double d2 = distSq(x, y, 0, 0);
        if (d2 < 4.0) {
            return 1.0;
        }
        if (d2 > 16.0 && d2 < 25.0) {
            return -1.0;
        }
        return null;
    }

    /*
     * Logic for doubleSpiral: double sided spiral.
     */
    private Double getDoubleSpiralLabel(double x, double y) {
        double r = Math.sqrt(x * x + y * y);
        if (r >= 6.0) {
            return null;
        }
        double val = Math.sin(Math.atan2(y, x) + (r * 1.25));
        double threshold = 0.8 / (r * r / 4 + 2.0);
        if (val > (1.0 - threshold)) {
            return 1.0;
        }
        if (val < (-1.0 + threshold)) {
            return -1.0;
        }
        return null;
    }

    private double distSq(double x1, double y1, double x2, double y2) {
        return Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
    }
}