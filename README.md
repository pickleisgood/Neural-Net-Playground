# Neural Net Playground

A **console-based** neural network playground in Java. You build a small MLP (multi-layer perceptron), choose a 2D dataset, and train the model from a text menu in the terminal—inspired by [TensorFlow Playground](https://playground.tensorflow.org/), but without a graphical UI.

The network is implemented on top of a **custom autograd engine**: scalar values form a computation graph and gradients are computed automatically via backpropagation, so no external ML library is used.

---

## What the program does

The application is **entirely terminal-based**. When you run it, a numbered menu appears. You interact by typing numbers and values at the prompts.

- **Build the network**  
  Add hidden layers one at a time and specify how many neurons each layer has. The model always has 2 inputs (for 2D points) and 1 output (binary classification).

- **Choose a dataset**  
  Pick one of: Two Blobs, Four Blobs, Concentric Circles, or Double Spiral. The data is used as (input, target) pairs for training.

- **Configure training**  
  Set the activation function (None, TanH, or ReLU) and the learning rate. You can reset the MLP or clear layer sizes and start over.

- **Train**  
  Run training for 100 epochs. Loss is printed every 10 epochs. Training uses MSE loss and your chosen learning rate.

- **Visualize data**  
  Print the current dataset in the console as a grid of `O` and `X` (the two classes).

There is no GUI—all interaction is through the console menu and keyboard input.

---

### From an IDE

1. Open the project and set the **source root** to `src/main` (so `model` and `ui` are packages).
2. Run the **`Run`** class: open `src/main/ui/Run.java` and run the program.


