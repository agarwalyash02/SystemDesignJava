import BasePizza.BasePizza;
import BasePizza.FarmHouse;
import BasePizza.Margerita;
import ToppingDecorator.ExtraCheese;
import ToppingDecorator.Mushroom;

public class Main {
    public static void main(String[] args) {
        BasePizza customPizza = new ExtraCheese(new Mushroom(new FarmHouse()));
        System.out.println(customPizza.cost());
    }
}
