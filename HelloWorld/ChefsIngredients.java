import java.util.*;

public class ChefsIngredients {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Read inputs
        int totalDays = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        String[] ingredients = scanner.nextLine().split(" ");
        int numIngredients = scanner.nextInt();
        int numCategories = scanner.nextInt();

        // Create a map to store ingredients by category
        Map<String, Queue<Ingredient>> ingredientMap = new HashMap<>();
        ingredientMap.put("Fat", new LinkedList<>());
        ingredientMap.put("Carb", new LinkedList<>());
        ingredientMap.put("Protein", new LinkedList<>());
        ingredientMap.put("Fiber", new LinkedList<>());
        ingredientMap.put("Seasoning", new LinkedList<>());

        StringBuilder output = new StringBuilder();

        // Process each day
        for (int day = 0; day < totalDays; day++) {
            String ingredientId = ingredients[day];
            String category = extractCategory(ingredientId);

            // Update ingredient map
            updateIngredientMap(ingredientMap, ingredientId, day);

            // Check if the chef can cook a dish
            if (canCookDish(ingredientMap, numIngredients, numCategories)) {
                // Cook the dish
                for (int i = 0; i < numIngredients; i++) {
                    for (String cat : ingredientMap.keySet()) {
                        Queue<Ingredient> categoryQueue = ingredientMap.get(cat);
                        if (!categoryQueue.isEmpty()) {
                            Ingredient ingredient = categoryQueue.poll();
                            output.append(ingredient.getId()).append(": ");
                            break;
                        }
                    }
                }
                output.append("# ");
            } else {
                // Chef doesn't cook on this day
                output.append("#. ");
            }
        }

        // Print the output
        System.out.println(output.toString().trim());
    }

    private static boolean canCookDish(Map<String, Queue<Ingredient>> ingredientMap, int numIngredients,
            int numCategories) {
        Set<String> usedCategories = new HashSet<>();
        for (String cat : ingredientMap.keySet()) {
            if (!ingredientMap.get(cat).isEmpty()) {
                usedCategories.add(cat);
            }
        }
        boolean canCooks = usedCategories.size() >= numCategories && totalAvailableIngredients(ingredientMap) >= numIngredients;
        return canCooks;
    }

    private static int totalAvailableIngredients(Map<String, Queue<Ingredient>> ingredientMap) {
        int total = 0;
        for (Queue<Ingredient> queue : ingredientMap.values()) {
            total += queue.size();
        }
        return total;
    }

    private static void updateIngredientMap(Map<String, Queue<Ingredient>> ingredientMap, String ingredientId,
            int day) {
        String category = extractCategory(ingredientId);
        Queue<Ingredient> categoryQueue = ingredientMap.get(category);
        if (categoryQueue != null) {
            categoryQueue.offer(new Ingredient(ingredientId, day));
            removeExpiredIngredients(ingredientMap, day);
        }
    }

    private static void removeExpiredIngredients(Map<String, Queue<Ingredient>> ingredientMap, int currentDay) {
        for (Queue<Ingredient> queue : ingredientMap.values()) {
            Iterator<Ingredient> iterator = queue.iterator();
            while (iterator.hasNext()) {
                Ingredient ingredient = iterator.next();
                int expiry = getExpiry(extractCategory(ingredient.getId()));
                if (currentDay - ingredient.getDayReceived() >= expiry) {
                    iterator.remove();
                }
            }
        }
    }

    private static String extractCategory(String ingredientId) {
        // Extract category from ingredient id
        String[] categories = { "Fat", "Fiber", "Protein", "Carb", "Seasoning" };
        for (String category : categories) {
            if (ingredientId.contains(category)) {
                System.out.println(category);
                return category;
            }
        }
        return "Unknown";
    }

    private static int getExpiry(String category) {
        switch (category) {
            case "Fat":
                return 2;
            case "Carb":
                return 3;
            case "Protein":
                return 4;
            case "Fiber":
                return 4;
            case "Seasoning":
                return 5;
            default:
                return 0;
        }
    }

    static class Ingredient {
        private String id;
        private int dayReceived;

        public Ingredient(String id, int dayReceived) {
            this.id = id;
            this.dayReceived = dayReceived;
        }

        public String getId() {
            return id;
        }

        public int getDayReceived() {
            return dayReceived;
        }
    }
}
