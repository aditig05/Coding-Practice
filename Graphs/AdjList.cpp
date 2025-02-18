#include <bits/stdc++.h>
using namespace std;

// Adjacency List

int main() {
    int m;
    cin>>m;
    vector<int>adjList[m+1];
    for(int i=0; i<m; i++){
            int u, v;
            cin>>u>>v;
            adjList[u].push_back(v);
            adjList[v].push_back(u);
    }
    for(int i=0; i<m; i++){
        cout<<i<<":";
        for(int j=0; j<adjList[i].size(); j++){
            cout<<adjList[i][j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}

// //Adjecency Matrix
// int main(){
//     int m; cin>>m;
//     int adjMat[m][m]={};
//     for(int i=0; i<m; i++){
//         int u, v;
//         cin>>u>>v;
//         adjMat[u][v]=1;
//         adjMat[v][u]=1;
//     }
//     for(int i=0; i<m; i++){
//         for(int j=0; j<m; j++){
//             cout<<adjMat[i][j]<<" ";
//         }
//         cout<<endl;
//     }
// }