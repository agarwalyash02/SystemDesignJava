import Strategy.DriverStrategy;

public class Vehicle {
    DriverStrategy driveStrategy;

    public Vehicle(DriverStrategy driverStrategy) {
        this.driveStrategy = driverStrategy;
    }

    public void drive() {
        driveStrategy.drive();
    }
}
