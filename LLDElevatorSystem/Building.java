package SystemDesignJava.LLDElevatorSystem;

import java.lang.reflect.Array;
import java.util.List;

public class Building {
    List<Floor> floorList;

    Building(List<Floor> floorList) {
        this.floorList = floorList;
    }

    public void addNewFloow(Floor floor) {
        this.floorList.add(floor);
    }

    public void removeFloors(Floor removeFloor) {
        floorList.remove(removeFloor);
    }

    List<Floor> getAllFloorList() {
        return floorList;
    }

}
