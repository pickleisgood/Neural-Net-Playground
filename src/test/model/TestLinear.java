package model;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestLinear {

    Linear linA;
    Linear linB;
    Linear linC;
    Linear linD;
    
    @BeforeEach
    void runBefore() {
        linA = new Linear(2, 5, "TanH");
        linB = new Linear(1, 2, "");
        linC = new Linear(1, 2, "TanH");
        linD = new Linear(1, 2, "ReLU");
    }

    @Test
    void testConstructor() {
        assertEquals(2, linA.getWeights().length);
        assertEquals(5, linA.getWeights()[0].length);
        assertEquals(5, linA.getBiases().length);
        assertEquals("TanH", linA.getActivation());

        assertEquals(1, linB.getWeights().length);
        assertEquals(2, linB.getWeights()[0].length);
        assertEquals(2, linB.getBiases().length);
        assertEquals("", linB.getActivation());
    }

    @Test
    void testForward() {
        Value a = new Value(1);
        Value b = new Value(2);
        Value c = new Value(3);
        Value d = new Value(4);

        Value[][] input = {{a, b},
                           {c, d}};

        Value[][] output = linA.forward(input, false);

        assertEquals(2, output.length);
        assertEquals(5, output[0].length);

        output = linA.forward(input, true);

        assertEquals(2, output.length);
        assertEquals(5, output[0].length);
    }

    @Test
    void handleActivation() { 
        Value a = new Value(0.5);
        Value[][] input = {{a}};
        
        Value[][] linCa = linC.forward(input, false);
        Value[][] linBa = linB.forward(input, false);
        Value[][] linDa = linD.forward(input, false);

        assertEquals(1, linCa.length);
        assertEquals(2, linCa[0].length);
        assertEquals(1, linBa.length);
        assertEquals(2, linBa[0].length);
        assertEquals(1, linDa.length);
        assertEquals(2, linDa[0].length);
    }

    @Test
    void testMathCorrectness() {
        Linear smallLin = new Linear(1, 1, "");
    
        smallLin.getWeights()[0][0].setData(2.0);
        smallLin.getBiases()[0].setData(5.0);

        Value[][] input = {{ new Value(3.0) }};
        Value[][] output = smallLin.forward(input, false);

        assertEquals(11.0, output[0][0].getData(), 0.0001);
    }

    @Test
    void testGetParameters() {
        assertEquals(15, linA.getParameters().size());
        assertEquals(4, linB.getParameters().size());
        assertEquals(4, linC.getParameters().size());
    }
}
