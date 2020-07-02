/*
author: diradon
24-Jun-20
19:09:24
*/

#include<bits/stdc++.h>
#define F first
#define S second
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define all(cont) cont.begin(), cont.end()
using namespace std;
#define int long long
//typedef long long int ll;

int32_t main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    map<int, int> bem;

    for (int i = 2; i < 10; i++) bem[i] = i*i;

    vector<int> bottom_a = { (*bem.begin()).first };
	vector<int> bottom_b = { (*bem.begin()).second };
	//vector<int> bottom_x = { -std::numeric_limits<float>::infinity() };

    for (auto x: bottom_a) cout << x << " ";
    cout << endl;
    for (auto x: bottom_b) cout << x << " ";
    cout << endl;

    cout << "INDEPENDENCE : ";
    cout << MAXFLOAT << endl;
    //cout << MINFLOAT << endl;
}