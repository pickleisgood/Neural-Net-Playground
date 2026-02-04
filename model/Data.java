package model;
import java.util.ArrayList;

public class Data {
    private ArrayList<DataPoint> DataPoints;

    public Data(){
        DataPoints = new ArrayList<>();
    }

    public void addPoint(DataPoint d){
        DataPoints.add(d);
    }

    public ArrayList<DataPoint> getPoints(){
        return DataPoints;
    }
}
