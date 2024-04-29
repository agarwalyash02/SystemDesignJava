package Observer;

import Observable.StocksObservable;

public class EmailAlertObserverImpl implements NotificationAlertObserver {
    String emailId;
    StocksObservable observable;

    public EmailAlertObserverImpl(String emailId, StocksObservable observable) {
        this.emailId = emailId;
        this.observable = observable;
    }

    @Override
    public void update() {
        sendEmail("product is in stock hurry up!!!");
    }

    public void sendEmail(String msg) {
        System.out.println("msg sent to email: " + emailId + " " + msg);
    }

}
