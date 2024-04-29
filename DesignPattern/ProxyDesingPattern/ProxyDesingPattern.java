import EmployeeTable.EmployeeDao;
import EmployeeTable.EmployeeDaoProxy;
import EmployeeTable.EmployeeDo;

public class ProxyDesingPattern {
    public static void main(String[] args) {
        try {
            EmployeeDao employeeTableObj = new EmployeeDaoProxy();
            employeeTableObj.create("USER", new EmployeeDo());
            System.out.println("Operation Successful");
        } catch (Exception err) {
            System.out.println(err.getMessage());
        }
    }
}
