#include<bits/stdc++.h>
using namespace std;
void DFS(vector<vector<int>> &G, vector<bool> &visited, int s){ // For DFS traversal
    visited[s] = true;
    cout << s << " ";
    //visiting all adjacent vertices that are not visited yet
    for (int i : G[s])
        if (visited[i] == false)
            DFS(G, visited, i);
}


int main(){
    int n,m; //n -> nodes, m -> edges
    cin>>n>>m;
    vector<vector<int>>G(n);
    for(int i=0; i<m; i++){
        int u,v;
        cin>>u>>v;    //for 0 based nodes, for 1 based nodes either make G of size n+1 or do u--, v--
        G[u].push_back(v);
        G[v].push_back(u);
    }
    //Printing the adjacency list

    // for(int i=0; i<n; i++){
    //     cout << i << " : ";
    //     for(int j=0; j<G[i].size(); j++){
    //         cout << G[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    vector<bool> visited(G.size(), false);
    // cout<<"DFS Traversal is: ";
    // int s = 0; // starting from source = 0
    // DFS(G, visited, s);

    queue<int> q;
    cout<<"BFS Traversal is: ";
    int s=0; // starting from source = 0
    q.push(s);
    visited[s] = true;
    while(!q.empty()){
        int node = q.front();
        q.pop();
        cout<<node<<" ";
        for(int i: G[node]){
            if(!visited[i]){
                q.push(i);
                visited[i] = true;
            }
        }
    }
}