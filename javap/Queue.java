package javap;

import java.util.LinkedList;

public class Queue<T> {
    private LinkedList<T> queue;

    public Queue() {
        queue = new LinkedList<T>();
    }

    // Add an item to the back of the queue
    public void enqueue(T item) {
        queue.addLast(item);
    }

    // Remove and return the front item from the queue
    public T dequeue() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return queue.removeFirst();
    }

    // Look at the front item without removing it
    public T peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return queue.getFirst();
    }

    // Check if the queue is empty
    public boolean isEmpty() {
        return queue.isEmpty();
    }

    // Get the size of the queue
    public int size() {
        return queue.size();
    }
}
