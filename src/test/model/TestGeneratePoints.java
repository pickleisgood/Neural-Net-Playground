package model;

import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestGeneratePoints {
    GeneratePoints gp;
    int side;

    @BeforeEach 
    void setup() {
        gp = new GeneratePoints();
        side = 75;
    }

    @Test
    void testTwoBlobsCenterPoints() {
        ArrayList<Value[][]> data = gp.twoBlobs(side);

        Value[][] targets = data.get(1);

        int centerIdx = (side / 2) * side + (side / 2);   
        int topRightIdx = (side / 4) * side + (3 * side / 4);
        int botLeftIdx = (3 * side / 4) * side + (side / 4);
        
        assertNull(targets[centerIdx][0]);
        assertEquals(1.0, targets[topRightIdx][0].getData());
        assertEquals(-1.0, targets[botLeftIdx][0].getData());
    }

    @Test
    void testFourBlobsCenterPoints() {
        ArrayList<Value[][]> data = gp.fourBlobs(side);

        Value[][] targets = data.get(1);

        int centerIdx = (side / 2) * side + (side / 2);   
        int topLeftIdx = (side / 4) * side + (side / 4);
        int topRightIdx = (side / 4) * side + (3 * side / 4);
        int botLeftIdx = (3 * side / 4) * side + (side / 4);
        int botRightIdx = (3 * side / 4) * side + (3 * side / 4);
        
        assertNull(targets[centerIdx][0]);
        assertEquals(-1.0, targets[topRightIdx][0].getData());
        assertEquals(-1.0, targets[botLeftIdx][0].getData());
        assertEquals(1.0, targets[topLeftIdx][0].getData());
        assertEquals(1.0, targets[botRightIdx][0].getData());
    }

    @Test
    void testConcentricCirclesBoundaries() {
        ArrayList<Value[][]> data = gp.concentricCircles(side);

        Value[][] targets = data.get(1);

        int centerIdx = (side / 2) * side + (side / 2);
        int outerIdx = centerIdx + 25;

        assertEquals(1.0, targets[centerIdx][0].getData());
        assertEquals(-1.0, targets[outerIdx][0].getData());
        assertNull(targets[0][0]);
    }

    @Test
    void testDoubleSpiralSimple() {
        ArrayList<Value[][]> data = gp.doubleSpiral(side);
        Value[][] targets = data.get(1);

        int centerIdx = (side / 2) * side + (side / 2);
        assertNull(targets[centerIdx][0]);

        int spiralAIdx = centerIdx - 5; 
        assertEquals(-1.0, targets[spiralAIdx][0].getData());

        int spiralBIdx = centerIdx + 5;
        assertEquals(1.0, targets[spiralBIdx][0].getData());
    }

}
