package SystemDesignJava.LLDElevatorSystem;

import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public class ElevatorController {
    public PriorityQueue<Integer> upMinPQ;
    public PriorityQueue<Integer> downMaxPQ;
    ElevatorCar elevatorCar;

    ElevatorController(ElevatorCar elevatorCar) {
        this.elevatorCar = elevatorCar;
        upMinPQ = new PriorityQueue<>();
        downMaxPQ = new PriorityQueue<>((a, b) -> (b - a));
    }

    public void submitExternalRequest(int floor, Direction direction) {
        if (direction == Direction.DOWN) {
            downMaxPQ.offer(floor);
        } else {
            upMinPQ.offer(floor);
        }
    }

    public void sumitInternalRequest(int floor, Direction direction, ElevatorCar elevatorCar) {

    }

    public void controlElevator() {
        while (true) {
            if (elevatorCar.elevatorDirection == Direction.UP) {

            }
        }
    }
}
