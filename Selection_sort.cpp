#include <bits/stdc++.h>
using namespace std;
// Function to find the index of the minimum element in the unsorted part of the array

int select(int &arr[], int i, int n)
    {
        // code here such that selectionSort() sorts arr[]
        int min = i;
        for(int j=i+1; j<n; j++){
            if(arr[min]>arr[j]) min=j;
        }
        return min;
    }
     
    void selectionSort(int arr[], int n)
    {
       //code here
       int min;
       for(int i=0; i<n; i++){
           min=select(arr, i, n);
           swap(&arr[min], &arr[i]);
       }
    }


// int select(int arr[], int i, int n)
// {
//     int min = i;
//     for (int j = i + 1; j < n; j++) {
//         if (arr[min] > arr[j]) 
//             min = j;
//     }
//     return min;
// }

// // Function to swap two elements in an array
// void swap(int* a, int* b)
// {
//     int temp = *a;
//     *a = *b;
//     *b = temp;
// }

// // Selection sort function
// void selectionSort(int arr[], int n)
// {
//     int min;
//     for (int i = 0; i < n - 1; i++) { // Loop till n-1
//         min = select(arr, i, n);
//         swap(&arr[min], &arr[i]); // Swapping the elements
//         for(int a=0; a<n; a++){
//             cout<<arr[a]<<" ";
//         }
//         cout<<endl;
//     }
// }

// Function to print the array
void printArray(int arr[], int size)
{
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Driver program to test the above functions
int main()
{
    int arr[] = {3, 6, 1, 8, 4, 2}; // Direct initialization of the array
    int n = sizeof(arr) / sizeof(arr[0]); // Calculate the size of the array

    selectionSort(arr, n); // Call the selection sort function
    printArray(arr, n);    // Print the sorted array

    return 0;
}
