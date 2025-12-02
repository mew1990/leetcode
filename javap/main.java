package javap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class main {
    public static void do_array() {
        // Create an array of 5 integers
        int[] numbers = new int[5];

        // Put values in the array
        numbers[0] = 10;
        numbers[1] = 20;
        numbers[2] = 30;
        numbers[3] = 40;
        numbers[4] = 50;

        // Loop through all numbers and print them
        for (int i = 0; i < numbers.length; i++) {
            System.out.println("Number " + (i + 1) + " is: " + numbers[i]);
        }
    }

    public static void do_list() {
        // Create an ArrayList of Strings
        ArrayList<String> fruits = new ArrayList<String>();

        // Add items to the list
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");

        // Print the list
        System.out.println("Fruits: " + fruits);

        // Get an item (the second fruit)
        System.out.println("Second fruit: " + fruits.get(1));

        // Change an item
        fruits.set(0, "Apricot");

        // Remove an item
        fruits.remove("Cherry");

        // Print the updated list
        System.out.println("Updated fruits: " + fruits);
    }

    public static void do_linked_list() {
        // Create a LinkedList of Integers
        LinkedList<Integer> numbers = new LinkedList<Integer>();

        // Add items to the list
        numbers.add(10);
        numbers.add(20);
        numbers.add(30);

        // Print the list
        System.out.println("Numbers: " + numbers);

        // Add an item at the beginning
        numbers.addFirst(5);

        // Add an item at the end
        numbers.addLast(40);

        // Print the updated list
        System.out.println("Updated numbers: " + numbers);

        // Remove the first item
        numbers.removeFirst();

        // Print the final list
        System.out.println("Final numbers: " + numbers);
    }

    public static void do_map() {
        // Create a new HashMap
        HashMap<String, Integer> ages = new HashMap<>();

        // Add key-value pairs to the map
        ages.put("Alice", 25);
        ages.put("Bob", 30);
        ages.put("Carol", 35);

        // Get a value from the map
        int bobAge = ages.get("Bob");
        System.out.println("Bob's age: " + bobAge);

        // Check if a key exists
        if (ages.containsKey("David")) {
            System.out.println("David's age is known");
        } else {
            System.out.println("David's age is unknown");
        }

        // Update a value
        ages.put("Alice", 26);

        // Remove a key-value pair
        ages.remove("Carol");

        // Print all entries
        System.out.println("\nAll entries:");
        for (String name : ages.keySet()) {
            System.out.println(name + ": " + ages.get(name));
        }
    }

    public static void main(String[] args) {
        do_array();
        do_list();
        do_linked_list();
        do_map();
    }

}
