#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <queue>
#include <time.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#define rep(p, q) for (int p=0; p<q; p++)
#define PI 0
#define AND 1
#define NOT 2
#define STATE_WIDTH 1
#define CONNECT_SAMPLE_RATIO 0.1
using namespace std;

int countOnesInBinary(uint64_t num, int width) {
    int count = 0;
    rep (_, width) {
        if (num & 1) {
            count++;
        }
        num >>= 1;
    }
    return count;
}

int binaryVectorToDecimal(const std::vector<int>& vec) {
    int decimal = 0;
    int size = vec.size();

    for(int i = 0; i < size; ++i) {
        decimal += vec[size - 1 - i] * std::pow(2, i);
    }

    return decimal;
}

int main(int argc, char **argv)
{
    srand((unsigned)time(0));
    if (argc != 3) {
        cout << "Failed" << endl;
        return 1;
    }
    string in_filename = argv[1];
    string out_filename = argv[2];
    
    cout << "Read File: " << in_filename << endl;
    freopen(in_filename.c_str(), "r", stdin);
    int n;  // number of gates
    int no_patterns; 
    scanf("%d %d", &n, &no_patterns);
    cout << "Number of gates: " << n << endl;

    // Graph
    vector<vector<int> > truth_table(n);
    vector<vector<int> > fanin_list(n);
    vector<vector<int> > fanout_list(n);
    vector<int> gate_levels(n);
    vector<int> pi_list;
    vector<bool> const_0(n), const_1(n); 
    int max_level = 0;

    for (int k=0; k<n; k++) {
        int level;
        string tt_str; 
        cin >> level >> tt_str; 
        vector<int> tt; 

        // Parse truth table 
        if (tt_str[0] == 'P') {
            tt.push_back(-1);
            pi_list.push_back(k);
        }
        else if (tt_str[0] == 'C') {
            if (tt_str[1] == '0') 
                const_0[k] = true;
            else if (tt_str[1] == '1')
                const_1[k] = true;
        }
        else {
            for (char c : tt_str) {
                if (c == '1') {
                    tt.push_back(1);
                }
                else if (c == '0') {
                    tt.push_back(0);
                }
            }
        }
        reverse(tt.begin(), tt.end());
        truth_table[k] = tt;
        gate_levels[k] = level;
        if (level > max_level) {
            max_level = level;
        }
    }
    vector<vector<int> > level_list(max_level+1);
    for (int k=0; k<n; k++) {
        level_list[gate_levels[k]].push_back(k);
    }
    // Read fanin list 
    for (int k=0; k<n; k++) {
        int no_fanin;
        cin >> no_fanin;

        vector<int> fanin;
        for (int l=0; l<no_fanin; l++) {
            int fanin_gate;
            cin >> fanin_gate;
            if (fanin_gate != -1) {
                fanin.push_back(fanin_gate);
                fanout_list[fanin_gate].push_back(k);
            }
        }
        fanin_list[k] = fanin;

    }

    int no_pi = pi_list.size();
    cout << "Number of PI: " << no_pi << endl;

    cout<<"Start Simulation"<<endl;
    // Simulation
    vector<vector<uint64_t> > full_states(n); 
    int tot_clk = 0;
    int clk_cnt = 0; 

    while (no_patterns > 0) {
        no_patterns -= STATE_WIDTH; 
        vector<uint64_t> states(n);
        // generate pi patterns 
        rep(k, no_pi) {
            int pi = pi_list[k];
            states[pi] = rand() % int(std::pow(2, STATE_WIDTH)); 
            // cout << "PI: " << pi << " " << states[pi] << endl;
        }
        // Const 0 and 1
        rep (k, n) {
            if (const_0[k]) {
                states[k] = 0;
            }
            if (const_1[k]) {
                states[k] = (uint64_t)std::pow(2, STATE_WIDTH) - 1;
            }
        }
        // Logic Simulation 
        for (int l = 1; l < max_level+1; l++) {
            for (int gate: level_list[l]) {
                // Const 0 and 1 
                if (const_0[gate] or const_1[gate]) {
                    continue;
                }
                // Find truth table 
                vector<int> fanin_state; 
                for (int fanin: fanin_list[gate]) {
                    // cout << fanin << ' ' << states[fanin] << endl; // Debug
                    fanin_state.push_back(states[fanin]);
                }
                // exit(0); // Debug
                reverse(fanin_state.begin(), fanin_state.end());
                int tt_val = binaryVectorToDecimal(fanin_state);
                uint64_t tt_res = truth_table[gate][tt_val];
                states[gate] = tt_res;

                // // Debug 
                // if (gate == 256) {
                //     cout << tt_val << ' ' << tt_res << endl;
                // }
            }
        }
        // Record
        rep (k, n) {
            full_states[k].push_back(states[k]);
        }

    }

    // Probability 
    freopen(out_filename.c_str(), "w", stdout);
    vector<float> prob_list(n);
    rep(k, n) {
        int cnt = 0;
        int tot_cnt = 0;
        int all_bits = 0;
        rep(p, full_states[k].size()) {
            cnt = countOnesInBinary(full_states[k][p], STATE_WIDTH);
            tot_cnt += cnt;
            all_bits += STATE_WIDTH;
        }
        prob_list[k] = (float)tot_cnt / all_bits;
    }
    rep(k, n) {
        printf("%d %f\n", k, prob_list[k]);
    }

    // Sample node pair 
    vector<long> pi_cover_hash_list(n);
    rep(k, n) {
        scanf("%ld", &pi_cover_hash_list[k]);
    }
    vector<int> tt_pair_a; 
    vector<int> tt_pair_b;
    vector<float> tt_pair_label;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            // 1. Must have the same PI cover hash
            if (pi_cover_hash_list[i] != pi_cover_hash_list[j])
                continue;
            // 2. Must have the similar probability
            if (std::abs(prob_list[i] - prob_list[j]) > 0.1)
                continue;
            // 3. They are not connected
            bool connected = false;
            for (int k = 0; k < fanout_list[i].size(); k++) {
                if (fanout_list[i][k] == j) {
                    connected = true;
                    break;
                }
            }
            if (connected)
                continue;
            connected = false;
            for (int k = 0; k < fanout_list[j].size(); k++) {
                if (fanout_list[j][k] == i) {
                    connected = true;
                    break;
                }
            }
            if (connected)
                continue;
            // 4. Extreme Case
            int cnt = 0;
            int all_bits = 0;
            rep(p, full_states[i].size()) {
                cnt += countOnesInBinary(~(full_states[i][p] ^ full_states[j][p]), STATE_WIDTH);
                all_bits += STATE_WIDTH; 
            }
            float tt_dis = 1 - (float)cnt / all_bits;
            if (tt_dis > 0.2 and tt_dis < 0.8)
                continue;
            tt_pair_a.push_back(i);
            tt_pair_b.push_back(j);
            tt_pair_label.push_back(tt_dis);
        }
    }
    printf("!!!! %d\n", tt_pair_a.size());
    rep(k, tt_pair_a.size()) {
        printf("%d %d %f\n", tt_pair_a[k], tt_pair_b[k], tt_pair_label[k]);
    }

    // // Sample connection pairs 
    // vector<int> src_list;
    // vector<int> dst_list;
    // vector<int> pair_a; 
    // vector<int> pair_b;
    // vector<int> pair_label;
    // rep(k, n) {
    //     if (rand() % 100 < CONNECT_SAMPLE_RATIO * 100) {
    //         src_list.push_back(k);
    //     }
    // }
    // rep(src_k, src_list.size()) {
    //     int src = src_list[src_k]; 
    //     vector<int> fanin_cone(n); 
    //     vector<int> fanout_cone(n);
    //     rep(k, n){ 
    //         fanin_cone[k] = 0;
    //         fanout_cone[k] = 0;
    //     }
    //     // Find fanin cone
    //     queue<int> q; 
    //     q.push(src);
    //     while (!q.empty()) {
    //         int cur = q.front();
    //         q.pop();
    //         for (int k = 0; k < fanin_list[cur].size(); k++) {
    //             int fanin = fanin_list[cur][k];
    //             if (fanin_cone[fanin] == 0) {
    //                 fanin_cone[fanin] = 1;
    //                 q.push(fanin);
    //             }
    //         }
    //     }
    //     // Find fanout cone
    //     q.push(src);
    //     while (!q.empty()) {
    //         int cur = q.front();
    //         q.pop();
    //         for (int k = 0; k < fanout_list[cur].size(); k++) {
    //             int fanout = fanout_list[cur][k];
    //             if (fanout_cone[fanout] == 0) {
    //                 fanout_cone[fanout] = 1;
    //                 q.push(fanout);
    //             }
    //         }
    //     }
    //     // Sample dst
    //     rep(k, n) {
    //         if (rand() % 100 < CONNECT_SAMPLE_RATIO * 100 && k != src) {
    //             pair_a.push_back(src);
    //             pair_b.push_back(k);
    //             if (fanin_cone[k] == 1) {
    //                 pair_label.push_back(1);
    //             }
    //             else if (fanout_cone[k] == 1) {
    //                 pair_label.push_back(2);
    //             }
    //             else {
    //                 pair_label.push_back(0);
    //             }
    //         }
    //     }
    // }

    // // Output
    // printf("!!!! %d\n", pair_a.size());
    // rep(k, pair_a.size()) {
    //     printf("%d %d %d\n", pair_a[k], pair_b[k], pair_label[k]);
    // }
}