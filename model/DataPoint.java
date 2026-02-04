package model;

public class DataPoint {
    private int xCoord;
    private int yCoord;

    public DataPoint(int x, int y){
        this.xCoord = x;
        this.yCoord = y;
    }

    public int getX(){
        return xCoord;
    }

    public int getY(){
        return yCoord;
    }
}
