package ui;

import model.*;
import javax.swing.*;

import java.awt.*;
import java.util.ArrayList;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

// A graphical user interface for the neural network
public class PlaygroundUI extends JFrame {

    private MLP mlp;
    private GeneratePoints generator = new GeneratePoints();
    private Value[][] inputs;
    private Value[][] targets;
    private ArrayList<Integer> indices = new ArrayList<>();
    private ArrayList<Integer> layerSizes = new ArrayList<>();
    
    private double lr = 0.01;
    private String activation = "TanH";
    private int side = 50; 
    private boolean isTraining = false;
    private Thread trainingThread;

    private VisPanel visPanel;
    private LayerPanel archPanel;
    private LossPanel lossPanel; 
    private JLabel epochLabel;
    private int epochCount = 0;

    private static final String JSON_STORE = "./data/gui_saved_model.json";
    private ArrayList<Double> lossHistory = new ArrayList<>(); 
    private String currentDataset = "Two Blobs";

    //EFFECTS: Instantiates the main neural network application
    public PlaygroundUI() {
        layerSizes.add(4);
        layerSizes.add(4);

        indices = new ArrayList<>();
        for (int i = 0; i < Math.pow(side, 2); i++) {
            indices.add(i);
        }

        resetModelAndData("Two Blobs", true); 

        setTitle("Neural Network Playground");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        getContentPane().setBackground(new Color(245, 245, 245));

        add(createTopPanel(), BorderLayout.NORTH);
        
        visPanel = new VisPanel();
        add(visPanel, BorderLayout.WEST);
        
        archPanel = new LayerPanel();
        add(archPanel, BorderLayout.CENTER);
        
        add(createDataPanel(), BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                printLog();
            }
        });
    } 

    /**
     * EFFECTS: Prints all events in the EventLog to the console
     */
    private void printLog() {
        for (model.Event next : model.EventLog.getInstance()) {
            System.out.println(next.toString());
        }
    }

    //REQUIRES: datasetChoice one of "twoBlobs", "fourBlobs", "concentricCircles", "doubleSpiral"
    //MODIFIES: mlp, epochCount, lossHistory, epochLabel, inputs, targets, visPanel, archPanel
    /*EFFECTS: sets training epoch to 0, clears loss history, resets mlp if resetWeights is true, 
     *sets inputs and targets to desired datasetChoice, repaints visPanel and archPanel
    */
    private void resetModelAndData(String datasetChoice, boolean resetWeights) {
        epochCount = 0;
        lossHistory.clear();

        if (epochLabel != null) {
            epochLabel.setText("Epoch: 0");
        }
        if (resetWeights) {
            mlp = new MLP(2, layerSizes, 1, activation);
        }
        String choice = (datasetChoice == null || datasetChoice.isEmpty()) ? currentDataset : datasetChoice;
        ArrayList<Value[][]> data = getData(choice); 

        if (data != null && !data.isEmpty()) {
            inputs = data.get(0);
            targets = data.get(1);
        }

        if (visPanel != null) {
            visPanel.repaint();
        }

        if (lossPanel != null) {
            lossPanel.repaint();
        }
    }

    //REQUIRES: datasetChoice one of "twoBlobs", "fourBlobs", "concentricCircles", "doubleSpiral" 
    //EFFECTS: returns desired data
    private ArrayList<Value[][]> getData(String datasetChoice) {
        switch (datasetChoice) {
            case "Two Blobs":
                currentDataset = "Two Blobs";
                return generator.getDataset(side, "twoBlobs");
            case "Four Blobs":
                currentDataset = "Four Blobs";
                return generator.getDataset(side, "fourBlobs"); 
            case "Concentric Circles":
                currentDataset = "Concentric Circles";
                return generator.getDataset(side, "concentricCircles");
            case "Double Spiral":
                currentDataset = "Double Spiral";
                return generator.getDataset(side, "doubleSpiral");
            default:
                return new ArrayList<>(); 
        }
    }

    //MODIFIES: isTraining
    //EFFECTS: starts training if not training and stops if is training
    private void changeTrainState() {
        isTraining = !isTraining;
        if (isTraining) {
            trainingThread = new Thread(() -> {
                while (isTraining) {
                    trainStep();
                    SwingUtilities.invokeLater(() -> {
                        epochLabel.setText("Epoch: " + epochCount);
                        visPanel.repaint(); 
                        lossPanel.repaint();
                    });
                    try {
                        Thread.sleep(20); 
                    } catch (InterruptedException e) { 
                        break; 
                    }
                }
            });
            trainingThread.start();
        } else {
            if (trainingThread != null) {
                trainingThread.interrupt();
            }
        }
    }

    //EFFECTS: trains the model by 1 train step
    private void trainStep() {
        java.util.Collections.shuffle(indices);

        for (Value p : mlp.getParameters()) {
            p.setGrad(0.0);
        }

        Value[][] predictions = mlp.forward(inputs);
        Value totalLoss = new Value(0.0);
        int active = 0;

        for (int idx:indices) {
            if (targets[idx][0] == null) {
                continue;
            }
            active++;
            Value diff = predictions[idx][0].add(targets[idx][0].mul(new Value(-1.0)));
            totalLoss = totalLoss.add(diff.pow(2));
        }

        Value avgLoss = totalLoss.mul(new Value(1.0 / active));
        avgLoss.backward();
        lossHistory.add(avgLoss.getData());

        for (Value p : mlp.getParameters()) {
            p.setData(p.getData() - lr * p.getGrad());
        }
        epochCount++;
    }

    //EFFECTS: saves the mlp model configuration
    private void saveModel() {
        if (mlp == null) {
            return;
        }

        persistence.JsonWriter writer = new persistence.JsonWriter(JSON_STORE);
        try {
            mlp.setSavingState(this.lr, this.currentDataset);
            writer.open();
            writer.write(mlp);
            writer.close();
            JOptionPane.showMessageDialog(this, "Model saved to " + JSON_STORE);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error saving: " + e.getMessage());
        }
    }

    //EFFECTS: loads the mlp model configuration
    private void loadModel() {
        persistence.JsonReader reader = new persistence.JsonReader(JSON_STORE);
        try {
            mlp = reader.read();
            
            this.layerSizes = new ArrayList<>(mlp.getHiddenDims());
            this.activation = mlp.getActivation();
            this.lr = mlp.getLr();
            this.currentDataset = mlp.getDataset();
            
            lossHistory.clear();
            archPanel.refresh();
            resetModelAndData(currentDataset, false);
            
            JOptionPane.showMessageDialog(this, "Model loaded from checkpoint!");
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error loading: " + e.getMessage());
        }
    }

    @SuppressWarnings("methodlength")
    /*EFFECTS: creates a set of buttons at the top of the application including
     * "Play / Pause", "Restart", "Save Model", "Load Model", "Epoch", "activation" and "lr"
    */
    private JPanel createTopPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 20, 10));
        
        JButton playBtn = new JButton("Play / Pause");
        playBtn.addActionListener(e -> changeTrainState());
        
        JButton resetBtn = new JButton("Restart");
        resetBtn.addActionListener(e -> {
            isTraining = false;
            resetModelAndData("", true);
        });

        JButton saveBtn = new JButton("Save Model");
        saveBtn.addActionListener(e -> saveModel());

        JButton loadBtn = new JButton("Load Model");
        loadBtn.addActionListener(e -> loadModel());

        panel.add(saveBtn);
        panel.add(loadBtn);

        epochLabel = new JLabel("Epoch: 0", SwingConstants.LEFT); 
        epochLabel.setFont(new Font("Arial", Font.BOLD, 14));

        epochLabel.setPreferredSize(new Dimension(150, 25));

        String[] activations = {"TanH", "ReLU", "None"};
        JComboBox<String> actCombo = new JComboBox<>(activations);
        actCombo.addActionListener(e -> {
            activation = (String) actCombo.getSelectedItem();
            resetModelAndData(currentDataset, true);
        });

        String[] lrs = {"0.1", "0.03", "0.01", "0.003", "0.001"};
        JComboBox<String> lrCombo = new JComboBox<>(lrs);
        lrCombo.setSelectedItem("0.01");
        lrCombo.addActionListener(e -> lr = Double.parseDouble((String) lrCombo.getSelectedItem()));

        panel.add(playBtn);
        panel.add(resetBtn);
        panel.add(epochLabel);
        panel.add(new JLabel("  Activation:"));
        panel.add(actCombo);
        panel.add(new JLabel("Learning Rate:"));
        panel.add(lrCombo);
        return panel;
    }

    //EFFECTS: Creates a panel on the right allowing the user to select the desired data
    private JPanel createDataPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder("DATA"));
        panel.setPreferredSize(new Dimension(150, 0));

        String[] names = {"Two Blobs", "Four Blobs", "Concentric Circles", "Double Spiral"};
        ButtonGroup group = new ButtonGroup();

        for (int i = 0; i < names.length; i++) {
            JRadioButton btn = new JRadioButton(names[i]);
            int index = i;
            btn.addActionListener(e -> resetModelAndData(names[index], true));
            if (i == 0) {
                btn.setSelected(true);
            }
            group.add(btn);
            panel.add(btn);
            panel.add(Box.createRigidArea(new Dimension(0, 10)));
        }

        panel.add(Box.createVerticalGlue());
        lossPanel = new LossPanel();
        panel.add(lossPanel);
    
        return panel;
    }

    //A panel on the left displaying model decision boundary
    private class VisPanel extends JPanel {
        private final int SIZE = 400;
        private final int BLOCKSIZE = 8; 

        //EFFECTS: instantiate the panel
        public VisPanel() {
            setPreferredSize(new Dimension(SIZE, SIZE));
            setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        }

        @SuppressWarnings("methodlength")
        //EFFECTS: Draws model decision boundary
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;

            // Draw Decision Boundary Background
            for (int x = 0; x < SIZE; x += BLOCKSIZE) {
                for (int y = 0; y < SIZE; y += BLOCKSIZE) {
                    double mappedX = ((double) x / SIZE * 12.0) - 6.0;
                    double mappedY = 6.0 - ((double) y / SIZE * 12.0);
                    
                    Value[][] input = {{new Value(mappedX), new Value(mappedY)}};
                    Value[][] pred = mlp.forward(input);
                    double val = pred[0][0].getData();

                    g2d.setColor(getGradient(val));
                    g2d.fillRect(x, y, BLOCKSIZE, BLOCKSIZE);
                }
            }

            // Draw the actual data points on top
            if (inputs == null || targets == null) {
                return;
            }
            
            for (int i = 0; i < inputs.length; i++) {
                if (targets[i][0] == null) {
                    continue;
                }

                double mapX = inputs[i][0].getData();
                double mapY = inputs[i][1].getData();
                
                int px = (int) (((mapX + 6.0) / 12.0) * SIZE);
                int py = (int) (((6.0 - mapY) / 12.0) * SIZE);

                double targetVal = targets[i][0].getData();
                if (targetVal == 1.0) {
                    g2d.setColor(new Color(255, 140, 0)); // Solid Orange
                } else {
                    g2d.setColor(new Color(0, 100, 255)); // Solid Blue
                }

                g2d.fillOval(px - 4, py - 4, 8, 8);
                g2d.setColor(Color.WHITE);
                g2d.drawOval(px - 4, py - 4, 8, 8);
            }
        }

        //EFFECTS: returns the colour of the decision boundary
        private Color getGradient(double val) {
            val = Math.max(-1.0, Math.min(1.0, val));
            float mix = (float) (val + 1.0) / 2.0f; 

            int r = (int)(255 * mix + 0 * (1 - mix));
            int g = (int)(165 * mix + 100 * (1 - mix));
            int b = (int)(0 * mix + 255 * (1 - mix));
            
            return new Color(r, g, b, 180); 
        }
    }

    //A panel in the middle allowing user to modify mlp
    private class LayerPanel extends JPanel {

        //EFFECTS: instantiate the panel
        public LayerPanel() {
            refresh();
        }

        @SuppressWarnings("methodlength")
        //EFFECTS: resets the panel
        public void refresh() {
            removeAll();
            setLayout(new FlowLayout(FlowLayout.CENTER, 30, 20));

            for (int i = 0; i < layerSizes.size(); i++) {
                add(createLayerColumn(i));
            }

            JButton addLayerBtn = new JButton("+ Layer");
            addLayerBtn.addActionListener(e -> {
                if (layerSizes.size() < 6) {
                    layerSizes.add(4);
                    resetModelAndData("", true);
                    refresh();
                }
            });
            
            JButton subLayerBtn = new JButton("- Layer");
            subLayerBtn.addActionListener(e -> {
                if (layerSizes.size() > 1) {
                    layerSizes.remove(layerSizes.size() - 1);
                    resetModelAndData("", true);
                    refresh();
                }
            });

            JPanel controls = new JPanel(new GridLayout(2, 1, 0, 5));
            controls.add(addLayerBtn);
            controls.add(subLayerBtn);
            add(controls);

            revalidate();
            repaint();
        }

        //EFFECTS: creates a subpanel representing a layer in the mlp
        private JPanel createLayerColumn(int index) {
            JPanel col = new JPanel(new GridLayout(4, 1, 0, 5));
            col.add(new JLabel("Hidden " + (index + 1), SwingConstants.CENTER));
            
            JButton plus = new JButton("+");
            plus.addActionListener(e -> {
                if (layerSizes.get(index) < 8) {
                    layerSizes.set(index, layerSizes.get(index) + 1);
                    resetModelAndData("", true);
                    refresh();
                }
            });

            col.add(new JLabel(layerSizes.get(index) + " neurons", SwingConstants.CENTER));
            col.add(plus);

            JButton minus = new JButton("-");
            minus.addActionListener(e -> {
                if (layerSizes.get(index) > 1) {
                    layerSizes.set(index, layerSizes.get(index) - 1);
                    resetModelAndData("", true);
                    refresh();
                }
            });
            col.add(minus);

            return col;
        }
    }

    //A panel to visualize model loss
    private class LossPanel extends JPanel {

        //EFFECTS: instantiates the panel
        public LossPanel() {
            setPreferredSize(new Dimension(150, 150));
            setBorder(BorderFactory.createTitledBorder("LOSS"));
            setBackground(Color.WHITE);
        }

        @SuppressWarnings("methodlength")
        //EFFECTS: draws in a new frame of the loss
        protected void paintComponent(Graphics g) {
            if (lossHistory.isEmpty()) {
                return;
            }
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(Color.RED);

            double currentLoss = lossHistory.get(lossHistory.size() - 1);
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Monospaced", Font.BOLD, 12));
            g2.drawString(String.format("%.4f", currentLoss), 10, 20);

            if (lossHistory.size() < 2) {
                return;
            }

            double maxLoss = lossHistory.stream().max(Double::compare).orElse(1.0);
            int w = getWidth();
            int h = getHeight();

            for (int i = 0; i < lossHistory.size() - 1; i++) {
                int x1 = (i * w) / lossHistory.size();
                int y1 = h - (int) ((lossHistory.get(i) / maxLoss) * h);
                int x2 = ((i + 1) * w) / lossHistory.size();
                int y2 = h - (int) ((lossHistory.get(i + 1) / maxLoss) * h);
                g2.drawLine(x1, y1, x2, y2);
            }
        }
    }

    //EFFECTS: runs the program
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new PlaygroundUI());
    }

}