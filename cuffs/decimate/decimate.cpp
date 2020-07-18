#include<iostream>
#include<vector>
using namespace std;

// void decimate(vector<float>& v_in, vector<float>& v_out, int step) {
// 	//TARGET: Cython
// 	cout << "Decimating (" << step << ")... ";
// 	v_out.clear();
// 	for (int i = 0; i < (int)v_in.size() / step; i++) {
// 		v_out.push_back(v_in[i * step]);
// 	}
// 	cout << "Done!" << endl;
// }


vector<float> decimate(vector<float> v_in, int step) {
	//TARGET: Cython
	vector<float> v_out;
	cout << "Decimating (" << step << ")... ";
	for (int i = 0; i < (int)v_in.size() / step; i++) {
		v_out.push_back(v_in[i * step]);
	}
	cout << "Done!" << endl;
	return v_out;
}
