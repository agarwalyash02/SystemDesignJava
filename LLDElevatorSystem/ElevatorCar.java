package SystemDesignJava.LLDElevatorSystem;

public class ElevatorCar {
    public int id;
    public Display display;
    public int currentFloor;
    public ElevatorState elevatorState;
    public InternalButton internalButton;
    public ElevatorDoor elevatorDoor;
    public Direction elevatorDirection;

    public void ElevatorCar() {
        this.display = new Display();
        internalButton = new InternalButton();
        elevatorState = ElevatorState.IDLE;
        elevatorDirection = elevatorDirection.UP;
        currentFloor = 0;
    }

    public void showDisplay() {
        display.showDisplay();
    }

    public void pressButton(int destination) {
        internalButton.pressButton(destination, this);
    }

    public void setDisplay() {
        this.display.setDisplay(currentFloor, elevatorDirection);
    }

    public boolean moveElevator(Direction dir, int destinationFloor) {
        int startFloor = currentFloor;
        if (dir == Direction.UP) {
            for (int i = startFloor; i <= destinationFloor; i++) {
                this.currentFloor = startFloor;
                setDisplay();
                showDisplay();
                if (i == destinationFloor) {
                    return true;
                }
            }
        }
        if (dir == Direction.DOWN) {
            for (int i = startFloor; i >= destinationFloor; i--) {
                this.currentFloor = startFloor;
                setDisplay();
                showDisplay();
                if (i == destinationFloor) {
                    return true;
                }
            }
        }
        return false;
    }
}
