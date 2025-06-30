import sys

def solve(N, A):
    def count_unique_subarray_sums(arr):
        """
        Counts the number of unique subarray sums in a given array.

        Args:
            arr: The input array.

        Returns:
            The number of unique subarray sums.
        """
        n = len(arr)
        unique_sums = set()  # Use a set to store unique sums

        # Iterate through all possible subarrays
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += arr[j]
                unique_sums.add(current_sum)

        return len(unique_sums)

    return count_unique_subarray_sums(A)

def main(): 
    N = int(sys.stdin.readline().strip())
    A = []

    for _ in range(N):
        A.append(int(sys.stdin.readline().strip()))

    result = solve(N, A)
    print(result)

if __name__ == "__main__":
    main()
