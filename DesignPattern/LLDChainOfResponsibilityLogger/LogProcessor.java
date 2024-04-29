public class LogProcessor {
    public static int INFO = 1;
    public static int DEBUG = 2;
    public static int ERROR = 3;

    LogProcessor nextLoggerProcessor;

    LogProcessor(LogProcessor logProcessor) {
        this.nextLoggerProcessor = logProcessor;
    }

    public void log(int level, String message) {
        if (nextLoggerProcessor != null) {
            nextLoggerProcessor.log(level, message);
        } else {
            System.out.println("Received a un-known level: " + level + " message: " + message);
        }
    }
}
