package EmployeeTable;

public class EmployeeDaoImpl implements EmployeeDao {

    @Override
    public void create(String client, EmployeeDo obj) throws Exception {
        System.out.println("created new row in employee table");
    }

    @Override
    public void delete(String client, int employeeId) throws Exception {
        System.out.println("deleted row in employee table with employee ID: " + employeeId);
    }

    @Override
    public EmployeeDo get(String client, int employeeId) throws Exception {
        System.out.println("fetching row from employee table with employee ID: " + employeeId);
        return new EmployeeDo();
    }
}
