package SystemDesignJava.LLDElevatorSystem;

public class Display {
    public int floor;
    public Direction direction;

    public void setDisplay(int floor, Direction direction) {
        this.floor = floor;
        this.direction = direction;
    }

    public void showDisplay() {
        System.out.println(this.floor);
        System.out.println(this.direction);
    }
}
