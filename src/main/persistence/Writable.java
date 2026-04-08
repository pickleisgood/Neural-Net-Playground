package persistence;

import org.json.JSONObject;

// Adapted from https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemoV
public interface Writable {
    // EFFECTS: returns this as a JSON object
    JSONObject toJson();
}