package model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestMLP {

    MLP mlpA;
    MLP mlpB;
    MLP mlpC;
    
    @BeforeEach
    void runBefore() {
        ArrayList<Integer> hiddenA = new ArrayList<>(List.of(2));
        ArrayList<Integer> hiddenB = new ArrayList<>(List.of(2, 2));
        ArrayList<Integer> hiddenC = new ArrayList<>(List.of());

        mlpA = new MLP(1, hiddenA, 2, "TanH");
        mlpB = new MLP(2, hiddenB, 1, "TanH");
        mlpC = new MLP(2, hiddenC, 4, "");
    }

    @Test
    void testConstructor() {
        assertEquals(1, mlpA.getLayers().get(0).getWeights().length);
        assertEquals(2, mlpA.getLayers().get(1).getWeights().length);
        assertEquals("TanH", mlpA.getActivation());

        assertEquals(2, mlpB.getLayers().get(0).getWeights().length);
        assertEquals(2, mlpB.getLayers().get(1).getWeights().length);
        assertEquals(2, mlpB.getLayers().get(2).getWeights().length);
        assertEquals("TanH", mlpB.getActivation());
    }

    @Test
    void testForward() {
        Value a = new Value(1);
        Value b = new Value(2);
        Value c = new Value(3);
        Value d = new Value(4);

        Value[][] input = {{a, b},
                           {c, d}};

        Value[][] output = mlpB.forward(input);

        assertEquals(2, output.length);
        assertEquals(1, output[0].length);
    }

    @Test
    void testForwardNoHidden() {
        Value a = new Value(1);
        Value b = new Value(2);
        Value c = new Value(3);
        Value d = new Value(4);

        Value[][] input = {{a, b},
                           {c, d}};

        Value[][] output = mlpC.forward(input);

        assertEquals(2, output.length);
        assertEquals(4, output[0].length);
    }

    @Test
    void testGetParameters() {
        assertEquals(10, mlpA.getParameters().size());
        assertEquals(15, mlpB.getParameters().size());
        assertEquals(12, mlpC.getParameters().size());
    }
}
