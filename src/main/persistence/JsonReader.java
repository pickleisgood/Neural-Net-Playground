package persistence;

import model.MLP;
import model.Value;
import org.json.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Stream;

// Represents a reader that reads MLP data from JSON data stored in file
// Adapted from https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemoV
public class JsonReader {
    private String source;

    public JsonReader(String source) {
        this.source = source;
    }

    // EFFECTS: reads MLP from file and returns it; throws IOException if error occurs
    public MLP read() throws IOException {
        String jsonData = readFile(source);
        JSONObject jsonObject = new JSONObject(jsonData);
        return parseMLP(jsonObject);
    }

    private String readFile(String source) throws IOException {
        StringBuilder contentBuilder = new StringBuilder();
        try (Stream<String> stream = Files.lines(Paths.get(source), StandardCharsets.UTF_8)) {
            stream.forEach(contentBuilder::append);
        }
        return contentBuilder.toString();
    }

    // EFFECTS: parses MLP from JSON object and returns it
    private MLP parseMLP(JSONObject jsonObject) {
        // Reconstruct Architecture & Activation
        int in = jsonObject.getInt("in");
        int out = jsonObject.getInt("out");
        String activation = jsonObject.getString("activation");
        double lr = jsonObject.getDouble("lr");
        String dataset = jsonObject.getString("dataset");
        
        JSONArray hiddenArray = jsonObject.getJSONArray("hidden");
        ArrayList<Integer> hidden = new ArrayList<>();
        for (int i = 0; i < hiddenArray.length(); i++) {
            hidden.add(hiddenArray.getInt(i));
        }

        MLP mlp = new MLP(in, hidden, out, activation);
        mlp.setSavingState(lr, dataset);

        // 2. Put in Saved Weights and Biases
        JSONArray paramArray = jsonObject.getJSONArray("parameters");
        ArrayList<Value> params = mlp.getParameters();
        for (int i = 0; i < params.size(); i++) {
            params.get(i).setData(paramArray.getDouble(i));
        }
        
        return mlp;
    }
}
