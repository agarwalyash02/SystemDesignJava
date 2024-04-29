package Observer;

import Observable.StocksObservable;

public class MobileAlertObserverImpl implements NotificationAlertObserver {
    String userName;
    StocksObservable observable;

    public MobileAlertObserverImpl(String userName, StocksObservable observable) {
        this.userName = userName;
        this.observable = observable;
    }

    @Override
    public void update() {
        sendMsgOnMobile("product is in stock hurry up!!!");
    }

    public void sendMsgOnMobile(String msg) {
        System.out.println("msg sent to userName: " + userName + " " + msg);
    }

}
