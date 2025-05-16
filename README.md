import math
import time
import matplotlib.pyplot as plt
import numpy as np
import random

# Implementation of Jump Search algorithm
def jump_search(arr, x):
    """
    Jump Search algorithm to find element x in a sorted array arr
    Returns the position of x if found, otherwise returns -1
    """
    n = len(arr)
    # Finding block size to be jumped
    step = int(math.sqrt(n))
    
    # Finding the block where element is present (if it's present)
    prev = 0
    while arr[min(step, n) - 1] < x:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Doing a linear search for x in block beginning with prev
    while arr[prev] < x:
        prev += 1
        
        # If we reached next block or end of array, element is not present
        if prev == min(step, n):
            return -1
    
    # If element is found
    if arr[prev] == x:
        return prev
    
    return -1

# Binary Search implementation for comparison
def binary_search(arr, x):
    """
    Binary Search algorithm to find element x in a sorted array arr
    Returns the position of x if found, otherwise returns -1
    """
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        # Check if x is present at mid
        if arr[mid] == x:
            return mid
        
        # If x is greater, ignore left half
        elif arr[mid] < x:
            low = mid + 1
        
        # If x is smaller, ignore right half
        else:
            high = mid - 1
    
    # Element was not present
    return -1

# Function to generate test data
def generate_test_data(size):
    """Generate a sorted array of given size with random elements"""
    arr = sorted([random.randint(0, size*10) for _ in range(size)])
    return arr

# Function to measure execution time
def measure_search_time(search_func, arr, target, iterations=10):
    """Measure average execution time of search function over multiple iterations"""
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        search_func(arr, target)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    return total_time / iterations

# Function to perform comparative analysis
def analyze_search_algorithms():
    # Define different input sizes
    small_sizes = [10, 20, 50, 100]
    medium_sizes = [500, 1000, 2000, 5000]
    large_sizes = [10000, 20000, 50000, 100000]
    
    all_sizes = small_sizes + medium_sizes + large_sizes
    
    # Initialize lists to store execution times
    jump_search_times = []
    binary_search_times = []
    
    # Measure execution times for different input sizes
    for size in all_sizes:
        arr = generate_test_data(size)
        
        # Select a random element from the array as search target
        # (to ensure the element exists in the array)
        target = random.choice(arr)
        
        # Measure Jump Search time
        jump_time = measure_search_time(jump_search, arr, target)
        jump_search_times.append(jump_time)
        
        # Measure Binary Search time
        binary_time = measure_search_time(binary_search, arr, target)
        binary_search_times.append(binary_time)
        
        print(f"Input size: {size}, Jump Search: {jump_time:.8f}s, Binary Search: {binary_time:.8f}s")
    
    # Plotting the results
    plt.figure(figsize=(15, 10))
    
    # Plot for small input sizes
    plt.subplot(2, 2, 1)
    plt.plot(small_sizes, [jump_search_times[i] for i in range(len(small_sizes))], 'ro-', label='Jump Search')
    plt.plot(small_sizes, [binary_search_times[i] for i in range(len(small_sizes))], 'bo-', label='Binary Search')
    plt.title('Search Algorithm Comparison - Small Input Sizes')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot for medium input sizes
    plt.subplot(2, 2, 2)
    plt.plot(medium_sizes, [jump_search_times[i+len(small_sizes)] for i in range(len(medium_sizes))], 'ro-', label='Jump Search')
    plt.plot(medium_sizes, [binary_search_times[i+len(small_sizes)] for i in range(len(medium_sizes))], 'bo-', label='Binary Search')
    plt.title('Search Algorithm Comparison - Medium Input Sizes')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot for large input sizes
    plt.subplot(2, 2, 3)
    plt.plot(large_sizes, [jump_search_times[i+len(small_sizes)+len(medium_sizes)] for i in range(len(large_sizes))], 'ro-', label='Jump Search')
    plt.plot(large_sizes, [binary_search_times[i+len(small_sizes)+len(medium_sizes)] for i in range(len(large_sizes))], 'bo-', label='Binary Search')
    plt.title('Search Algorithm Comparison - Large Input Sizes')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot for all input sizes (logarithmic scale)
    plt.subplot(2, 2, 4)
    plt.loglog(all_sizes, jump_search_times, 'ro-', label='Jump Search')
    plt.loglog(all_sizes, binary_search_times, 'bo-', label='Binary Search')
    plt.title('Search Algorithm Comparison - All Input Sizes (Log Scale)')
    plt.xlabel('Input Size (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('search_algorithm_comparison.png')
    plt.show()
    
    # Calculate and display asymptotic complexity
    print("\nAsymptotic Analysis:")
    print("Jump Search: O(√n) - where n is the size of the input array")
    print("Binary Search: O(log n) - where n is the size of the input array")
    
    # Verify the theoretical complexity with experimental data
    # For jump search, time should be proportional to √n
    # For binary search, time should be proportional to log n
    theory_jump = [0.00001 * math.sqrt(size) for size in all_sizes]
    theory_binary = [0.000001 * math.log2(size) for size in all_sizes]
    
    plt.figure(figsize=(12, 6))
    
    # Normalize experimental data for comparison with theoretical curves
    max_jump = max(jump_search_times)
    max_theory_jump = max(theory_jump)
    scaled_jump = [t * (max_theory_jump / max_jump) for t in jump_search_times]
    
    max_binary = max(binary_search_times)
    max_theory_binary = max(theory_binary)
    scaled_binary = [t * (max_theory_binary / max_binary) for t in binary_search_times]
    
    plt.subplot(1, 2, 1)
    plt.plot(all_sizes, scaled_jump, 'ro-', label='Jump Search (Experimental)')
    plt.plot(all_sizes, theory_jump, 'r--', label='O(√n) (Theoretical)')
    plt.title('Jump Search: Experimental vs Theoretical')
    plt.xlabel('Input Size')
    plt.ylabel('Normalized Time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(all_sizes, scaled_binary, 'bo-', label='Binary Search (Experimental)')
    plt.plot(all_sizes, theory_binary, 'b--', label='O(log n) (Theoretical)')
    plt.title('Binary Search: Experimental vs Theoretical')
    plt.xlabel('Input Size')
    plt.ylabel('Normalized Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('theoretical_comparison.png')
    plt.show()

# Run the analysis
if __name__ == "__main__":
    analyze_search_algorithms()


