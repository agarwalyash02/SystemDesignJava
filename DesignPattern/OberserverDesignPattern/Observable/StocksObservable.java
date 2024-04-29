package Observable;

import Observer.NotificationAlertObserver;

/**
 * StocksObservable
 */
public interface StocksObservable {

    public void add(NotificationAlertObserver observer);

    public void remove(NotificationAlertObserver observer);

    public void notifySubscribers();

    public void setStockCount(int newStockCount);

    public int getStockCount();
}