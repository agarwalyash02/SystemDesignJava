import Observable.IphoneObervableImpl;
import Observable.StocksObservable;
import Observer.EmailAlertObserverImpl;
import Observer.MobileAlertObserverImpl;
import Observer.NotificationAlertObserver;

public class Store {
    public static void main(String[] args) {

        StocksObservable iphoneObserver = new IphoneObervableImpl();

        NotificationAlertObserver observer1 = new EmailAlertObserverImpl("xyz@gmail.com", iphoneObserver);
        NotificationAlertObserver observer2 = new EmailAlertObserverImpl("abc@gmail.com", iphoneObserver);
        NotificationAlertObserver observer3 = new MobileAlertObserverImpl("pqrs_username", iphoneObserver);

        iphoneObserver.add(observer1);
        iphoneObserver.add(observer2);
        iphoneObserver.add(observer3);
        iphoneObserver.setStockCount(10);
    }
}
