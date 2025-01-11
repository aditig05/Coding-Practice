// C++ program to find the lower bound of a value in a
// vector using std::lower_bound()
#include <bits/stdc++.h>
using namespace std;

int main() {
    int arr[5] = {10, 20, 30, 40, 50};
      int n = sizeof(arr)/sizeof(arr[0]);

    // Finding lower bound for value 35 in array arr
    cout << lower_bound(arr, arr + n, 35)-arr;

    return 0;
}