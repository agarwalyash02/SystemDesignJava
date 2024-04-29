import ShapeFactory.Shape;
import ShapeFactory.ShapeFactory;

public class Main {
    public static void main(String[] args) {
        ShapeFactory shapeFactoryObj = new ShapeFactory();
        Shape shapeObj = shapeFactoryObj.getShape("Rectangle");
        shapeObj.draw();
    }
}
