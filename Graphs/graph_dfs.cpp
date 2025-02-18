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
void BFS(vector<vector<int>> &G, vector<bool> &visited, int s){ // For BFS traversal
    queue<int> q;
    cout<<"BFS Traversal is: ";
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

    int choice, s;
    cout << "Enter 1 for DFS, 2 for BFS: ";
    cin >> choice;
    cout << "Enter starting node: ";
    cin >> s;
    
    vector<bool> visited(n, false);
    
    if (choice == 1) {
        cout << "DFS Traversal: ";
        DFS(G, visited, s);
    } else if (choice == 2) {
        cout << "BFS Traversal: ";
        BFS(G, visited, s);
    } else {
        cout << "Invalid choice!";
    }
    
    return 0;
}

