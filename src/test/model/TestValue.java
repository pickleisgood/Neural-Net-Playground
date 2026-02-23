package model;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
 
public class TestValue {

    Value valA;
    Value valB;
    
    @BeforeEach
    void runBefore() {
        valA = new Value(5);
        valB = new Value(4);
    }

    @Test
    void testConstructor() {
        assertEquals(5, valA.getData());
        assertEquals(0, valA.getGrad());
        assertEquals(0, valA.getChildren().size());
        assertEquals(4, valB.getData());
        assertEquals(0, valB.getGrad());
        assertEquals(0, valB.getChildren().size());
    }

    @Test
    void testMulBackpropagation() {
        Value c = valA.mul(valB);
        
        // Check Forward Pass
        assertEquals(20.0, c.getData());
        assertEquals(0.0, valA.getGrad());
        assertEquals(0.0, valB.getGrad());
        
        c.backward();
        
        assertEquals(1.0, c.getGrad());
        assertEquals(4.0, valA.getGrad());
        assertEquals(5.0, valB.getGrad());
    }
    
    @Test
    void testReluNegativeValue() {
        Value x = new Value(-5.0);
        Value y = x.relu();
        
        y.backward();

        assertEquals(0.0, y.getData());
        assertEquals(1.0, y.getGrad());
        assertEquals(0.0, x.getGrad());
    }
    @SuppressWarnings("methodlength")
    @Test
    void testComplexBackpropagation() { 
        Value valA = new Value(0.5);
        Value valB = new Value(2.0);
        Value c = new Value(0.75);

        Value v = valA.mul(valB);      //1.0
        Value w = v.mul(valA);         // 0.5
        Value x = w.add(c);            //1.25
        Value r = x.relu();            //1.25
        Value y = r.tanh();            //0.84828363995
        Value z = y.pow(2);     //0.71958513382

        z.backward();
        
        double expectedZ = Math.pow(Math.tanh(Math.max(0, 0.5 * 2 * 0.5 + 0.75)), 2);
        assertEquals(expectedZ, z.getData(), 1e-9);
        assertEquals(1.0, z.getGrad());

        double dzdy = 2 * y.getData();
        assertEquals(dzdy, y.getGrad(), 1e-9);

        double dydr = 1 - Math.pow(y.getData(), 2);
        double dzdr = dzdy * dydr;
        assertEquals(dzdr, r.getGrad(), 1e-9);

        double drdx = 1.0;
        double dzdx = dzdr * drdx;
        assertEquals(dzdx, x.getGrad(), 1e-9);

        double dzdw = dzdx * 1.0;
        double dzdc = dzdx * 1.0;
        assertEquals(dzdw, w.getGrad(), 1e-9);
        assertEquals(dzdc, c.getGrad(), 1e-9);

        double dzdv = dzdw * valA.getData();
        double dzda_from_w = dzdw * v.getData();
        assertEquals(dzdv, v.getGrad(), 1e-9);

        double dzda = dzda_from_w + (dzdv * valB.getData());
        double dzdb = dzdv * valA.getData();

        assertEquals(dzda, valA.getGrad(), 1e-9);
        assertEquals(dzdb, valB.getGrad(), 1e-9);
    }
}
