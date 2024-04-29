package SystemDesignJava.LLDCarRentalSystem;

import java.util.ArrayList;
import java.util.List;

import SystemDesignJava.LLDCarRentalSystem.Product.Car;
import SystemDesignJava.LLDCarRentalSystem.Product.Store;
import SystemDesignJava.LLDCarRentalSystem.Product.Vehicle;
import SystemDesignJava.LLDCarRentalSystem.Product.VehicleType;

public class Main {
    public static void main(String[] args) {

        List<User> users = addUser();
        List<Vehicle> vehicles = addVehicles();
        List<Store> stores = addStores(vehicles);
        VehicleRentalSystem vehicleRentalSystem = new VehicleRentalSystem(stores, users);

        // user 1 comes
        User user1 = users.get(0);

        // 1. user search store based on location
        Location location = new Location(403012, "Bangalore", "Karnataka", "India");
        Store store1 = vehicleRentalSystem.getStore(location);

        // 2. get All vehicles you are interested in (based upon different filters)
        List<Vehicle> storeVehicles = store1.getVehicles(VehicleType.CAR);

        // 3.reserving the particular vehicle
        Reservation reservation = store1.createReservation(storeVehicles.get(0), user1);

        // 4. generate the bill
        Bill bill = new Bill(reservation);

        // 5. make payment
        Payment payment = new Payment();
        payment.payBill(bill);

        // 6. trip completed, submit the vehicle and close the reservation
        store1.completeReservation(reservation.reservationId);
    }

    private static List<User> addUser() {
        List<User> userList = new ArrayList<>();
        User user1 = new User();
        user1.setUserId(1);
        userList.add(user1);
        return userList;
    }

    private static List<Vehicle> addVehicles() {
        List<Vehicle> vehicleList = new ArrayList<>();
        Vehicle vehicle1 = new Car();
        vehicle1.setVehicleID(1);
        vehicle1.setVehicleType(VehicleType.CAR);

        Vehicle vehicle2 = new Car();
        vehicle1.setVehicleID(2);
        vehicle1.setVehicleType(VehicleType.CAR);

        vehicleList.add(vehicle1);
        vehicleList.add(vehicle2);
        return vehicleList;
    }

    private static List<Store> addStores(List<Vehicle> vehicles) {
        List<Store> stores = new ArrayList<>();
        Store store1 = new Store();
        store1.storeId = 1;
        store1.setVehicles(vehicles);
        stores.add(store1);
        return stores;
    }
}
