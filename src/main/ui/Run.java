package ui;

import model.*;
import persistence.*;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;
import java.io.FileNotFoundException; 
import java.io.IOException;

// Neural Net PlayGround Runner
public class Run {
    private static final int TRAIN_EPOCHS = 100;
    private static final String JSON_STORE = "./data/savedModel.json";
    private static final int SIDE = 75;

    private Scanner scanner = new Scanner(System.in);
    private GeneratePoints generator = new GeneratePoints();
    private MLP mlp;

    private String dataset;
    private Value[][] inputs;
    private Value[][] targets;
    private double lr = 0.01;

    private ArrayList<Integer> layerSizes = new ArrayList<>();
    private String activation = "None";
    
    // EFFECTS: creates instance of the application
    public Run() {
        int input = 0;
        while (input != 111) {
            displayOptions();
            input = scanner.nextInt();
            handelInput(input);
        }
    }

    // EFFECTS: displays options to user
    public void displayOptions() {
        System.out.println("Options:");
        System.out.println("1. Add Layer");
        System.out.println("2. View Layers");
        System.out.println("3. Pick data");
        System.out.println("4. Visualize data");
        System.out.println("5. Pick Activation Function");
        System.out.println("6. Set Learning Rate");
        System.out.println("7. Reset MLP");
        System.out.println("8. Reset Layer Sizes");
        System.out.println("9. Train Model");
        System.out.println("10. Save Model");
        System.out.println("11. Load Model");
        System.out.println("111. Quit");
    }

    // EFFECTS: performs action based on user input
    @SuppressWarnings("methodlength")
    public void handelInput(int input) {
        switch (input) {
            case 1:
                addLayer();
                break;
            case 2:
                viewSetup();
                break;
            case 3:
                pickData();
                break;
            case 4:
                visualize();
                break;
            case 5:
                pickActivationFunction();
                break;
            case 6:
                setLearningRate();
                break;
            case 7:
                resetMLP();
                break;
            case 8:
                resetLayerSizes();
                break;
            case 9:
                train();
                break;
            case 10:
                saveModel();
                break;
            case 11:
                loadModel();
                break;
            case 111:
                System.out.println("Exiting...");
                System.exit(0);
                break;
            default:
                System.out.println("Invalid option inputted. Please try again.");
        }
    }

    // MODIFIES: this.layerSizes
    // EFFECTS: adds a int to layerSizes which will be used to instantiate MLP
    public void addLayer() {
        System.out.println("Current sizes: " + layerSizes);
        System.out.print("Enter number of neurons for this layer: ");
        int size = scanner.nextInt();
        layerSizes.add(size);
        viewSetup();
    }

    // EFFECTS: displays the current MLP setup to the user
    public void viewSetup() {
        if (layerSizes.isEmpty()) {
            System.out.println("No layers defined yet.");
        } else {
            System.out.println("Architecture: Input(2) -> " + layerSizes + " -> Output(1)");
            System.out.println("Learning Rate: " + this.lr);
            System.out.println("Activation: " + ((activation.length() > 0) ? activation : "None"));
            System.out.println("Dataset: " + this.dataset);
        }
    }

    /*  EFFECTS: allows the user to pick their data and sets 
     * the inputs and targets to be of that data.
    */
    public void pickData() {
        System.out.println("Select Data:");
        System.out.println("0. Two Blobs");
        System.out.println("1. Four Blobs");
        System.out.println("2. Concentric Circles");
        System.out.println("3. Double Spiral");
        
        int choice = scanner.nextInt();

        if (choice == 0) {
            this.dataset = "twoBlobs";
        } else if (choice == 1) {
            this.dataset = "fourBlobs";
        } else if (choice == 2) {
            this.dataset = "concentricCircles";
        } else if (choice == 3) {
            this.dataset = "doubleSpiral";
        } else {
            System.out.println("Invalid choice, keeping current data");
            return;
        }
        setData(this.dataset);
    }

    // REQUIRES: dataset one of "twoBlobs", "fourBlobs", "concentricCircle", "doubleSpiral"
    // MODIFIES: inputs, targets, dataset
    /* EFFECTS: sets inputs and targets to the dataset given.
    */
    public void setData(String dataset) {
        ArrayList<Value[][]> data = generator.getDataset(SIDE, dataset);
        inputs = data.get(0);
        targets = data.get(1);
        System.out.println("Dataset set to: " + this.dataset);
    }

    // MODIFIES: activation
    // EFFECTS: allows the user to pick their activation function
    public void pickActivationFunction() {
        System.out.println("Select Activation Function:");
        System.out.println("0. None");
        System.out.println("1. TanH");
        System.out.println("2. ReLU");
        
        int choice = scanner.nextInt();
        
        if (choice == 0) {
            this.activation = "None"; 
            System.out.println("Activation set to: None");
        } else if (choice == 1) {
            this.activation = "TanH";
            System.out.println("Activation set to: TanH");
        } else if (choice == 2) {
            this.activation = "ReLU";
            System.out.println("Activation set to: ReLU");
        } else {
            System.out.println("Invalid choice. Keeping current: " + activation);
        }
    }

    // EFFECTS: displays current dataset chosen 
    public void visualize() {
        if (targets == null) {
            System.out.println("No data picked");
        } else {
            for (int i = 0; i < SIDE; i++) {
                for (int j = 0; j < SIDE; j++) {
                    int index = i * SIDE + j;
                    Value target = targets[index][0];

                    if (target == null) {
                        System.out.print(". "); 
                    } else {
                        double val = target.getData();
                        if (val == -1.0) {
                            System.out.print("O "); 
                        } else if (val == 1.0) {
                            System.out.print("X "); 
                        }
                    }
                }
                System.out.println(); 
            }
        }
    }

    // EFFECTS: let user set learning rate
    public void setLearningRate() {
        System.out.println("Pick a learning rate");
        double newLR = scanner.nextDouble();
        while (newLR < 0) {
            System.out.println("Learning Rate must be greater than 0");
            newLR = scanner.nextDouble();
        }
        lr = newLR;
        System.out.println("New Learning Rate: " + lr);
    }

    // EFFECTS: reinitializes MLP
    public void resetMLP() {
        System.out.println("Reinitialising MLP");
        mlp = new MLP(2, layerSizes, 1, this.activation);
    }

    // EFFECTS: makes layerSizes empty
    public void resetLayerSizes() {
        System.out.println("Layer Sizes Reset");
        layerSizes = new ArrayList<>();
    }

    @SuppressWarnings("methodlength")
    // EFFECTS: trains the MLP
    public void train() {
        if (inputs == null || targets == null) {
            System.out.println("Pick data before training");
            return;
        }

        if (mlp == null) {
            mlp = new MLP(2, layerSizes, 1, this.activation);
        }

        // create a list of indices to 'shuffle' the data as it trains
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            indices.add(i);
        }

        System.out.println("Starting training for " + TRAIN_EPOCHS + " epochs...");
        for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
            zeroGradients(); 

            Collections.shuffle(indices);
            Value[][] predictions = mlp.forward(inputs);
            Value totalLoss = new Value(0.0);
            int activePoints = 0;

            for (int idx: indices) {
                if (targets[idx][0] == null) {
                    continue;
                }

                activePoints++;

                Value targetVal = targets[idx][0];
                Value diff = predictions[idx][0].add(targetVal.mul(new Value(-1.0))); // pred - target
                Value mseLoss = diff.pow(2);
                totalLoss = totalLoss.add(mseLoss);
            }

            Value averageLoss = totalLoss.mul(new Value(1.0 / activePoints));

            averageLoss.backward();
            updateParameters();

            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d | Loss: %.6f\n", epoch, averageLoss.getData());
            }
        }
    }

    // EFFECTS: updates MLP parameters
    private void updateParameters() {
        for (Value p : mlp.getParameters()) {
            p.setData(p.getData() - lr * p.getGrad());
        }
    }

    // EFFECTS: sets all parameter gradients of MLP to 0
    private void zeroGradients() {
        for (Value p : mlp.getParameters()) {
            p.setGrad(0.0);
        }
    }

    // MODIFIES: JSON_STORE file
    // EFFECTS: saves the current mlp state to file
    private void saveModel() {
        if (mlp == null) {
            System.out.println("Initialize and train a model before saving!");
            return;
        }

        JsonWriter writer = new JsonWriter(JSON_STORE);
        try {
            mlp.setSavingState(lr, dataset);
            writer.open();
            writer.write(mlp);
            writer.close();
            System.out.println("Saved model to " + JSON_STORE);
        } catch (FileNotFoundException e) {
            System.out.println("Unable to write to file: " + JSON_STORE);
        }
    }

    // MODIFIES: this.mlp
    // EFFECTS: loads mlp state from file and prints out loaded setup
    private void loadModel() {
        JsonReader reader = new JsonReader(JSON_STORE);
        try {
            mlp = reader.read();
        
            this.layerSizes = mlp.getHiddenDims();
            this.activation = mlp.getActivation();
            this.lr = mlp.getLr();
            this.dataset = mlp.getDataset();

            if (dataset != null) {
                setData(dataset);
            }
            
            System.out.println("Model successfully loaded from " + JSON_STORE);
            viewSetup();
        } catch (IOException e) {
            System.out.println("Unable to read from file: " + JSON_STORE);
        }
    }    
}
