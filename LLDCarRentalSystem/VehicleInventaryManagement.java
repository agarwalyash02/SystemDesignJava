package SystemDesignJava.LLDCarRentalSystem;

import java.util.List;

import SystemDesignJava.LLDCarRentalSystem.Product.Vehicle;

public class VehicleInventaryManagement {
    List<Vehicle> vehicles;

    public VehicleInventaryManagement(List<Vehicle> vehicles) {
        this.vehicles = vehicles;
    }

    public List<Vehicle> getVehicles() {
        // filtering
        return vehicles;
    }

    public void setVehicles(List<Vehicle> vehicles) {
        this.vehicles = vehicles;
    }

}
