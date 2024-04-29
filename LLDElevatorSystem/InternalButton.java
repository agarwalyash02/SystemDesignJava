package SystemDesignJava.LLDElevatorSystem;

public class InternalButton {
    public int[] availableButtons = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    public int buttonSelected;

    public void pressButton(int buttonPressed, ElevatorCar elevatorCar) {
        // 1.check if destination is in the list of available floors

        // 2.submit the request to the jobDispatcher
        System.out.println(buttonPressed + " " + elevatorCar);
    }
}
