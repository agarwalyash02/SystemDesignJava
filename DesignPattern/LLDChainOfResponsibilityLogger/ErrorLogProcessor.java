public class ErrorLogProcessor extends LogProcessor {
    ErrorLogProcessor(LogProcessor nexLogProcessor) {
        super(nexLogProcessor);
    }

    public void log(int level, String message) {
        if (level == ERROR) {
            System.out.println("ERROR: " + message);
        } else {
            super.log(level, message);
        }
    }
}
