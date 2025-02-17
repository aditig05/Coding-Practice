#include <bits/stdc++.h>
using namespace std;

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
    cout<<"-------"<<"\n";
    for(int i=0; i<m; i++){
        cout<<i<<":";
        for(int j=0; j<adjList[i].size(); j++){
            cout<<adjList[i][j]<<" ";
        }
        cout<<endl;
    }
    vector<int> vis(m, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < adjList[i].size(); j++) { 
            int node = adjList[i][j];
            if (vis[node] == 0) {
                cout << node << " ";
                vis[node] = 1; 
            }
        }
    }
    return 0;
}