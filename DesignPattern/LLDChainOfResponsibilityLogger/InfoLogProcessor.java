public class InfoLogProcessor extends LogProcessor {
    InfoLogProcessor(LogProcessor nexLogProcessor) {
        super(nexLogProcessor);
    }

    public void log(int level, String message) {
        if (level == INFO) {
            System.out.println("INFO: " + message);
        } else {
            super.log(level, message);
        }
    }
}
