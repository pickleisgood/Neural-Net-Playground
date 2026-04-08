package persistence;

import model.MLP;
import java.io.*;

// Represents a writer that writes JSON representation of an MLP to file
// Adapted from https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemoV
public class JsonWriter {
    private static final int TAB = 4;
    private PrintWriter writer;
    private String destination;

    // EFFECTS: constructs writer to write to destination file
    public JsonWriter(String destination) {
        this.destination = destination;
    }

    // MODIFIES: this
    // EFFECTS: opens writer; throws FileNotFoundException if destination file cannot be opened
    public void open() throws FileNotFoundException {
        writer = new PrintWriter(new File(destination));
    }

    // MODIFIES: this
    // EFFECTS: writes JSON representation of MLP to file
    public void write(MLP mlp) {
        writer.print(mlp.toJson().toString(TAB));
    }

    // MODIFIES: this
    // EFFECTS: closes writer
    public void close() {
        writer.close();
    }
}