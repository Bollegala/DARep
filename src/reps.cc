#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <unordered_map>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <map>
#include <cmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include "parse_args.hh"
#include "omp.h"

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

using namespace std;
using namespace Eigen;

unordered_map<string, VectorXd> wS; // source non-pivots
unordered_map<string, VectorXd> wT; // target non-pivots
unordered_map<string, VectorXd> cS; // source pivots
unordered_map<string, VectorXd> cT; // target pivots

list<string> pivots;
list<string> src_nonpivots;
list<string> tgt_nonpivots;

int D = 300; // dimensionality

unordered_map<string, VectorXd> grad_wS; // Squared gradient for AdaGrad
unordered_map<string, VectorXd> grad_wT; 
unordered_map<string, VectorXd> grad_cS; 
unordered_map<string, VectorXd> grad_cT; 

struct instance{
    string pivot;
    vector<string> positives;
    vector<string> negatives;
};

vector<instance> src_instances;
vector<instance> tgt_instances;

void load_pivots(string fname, list<string> &l, string domain){
    ifstream f(fname.c_str());
    string fid, fval, dname, score;
    while (f >> fid >> fval >> dname >> score){
        if(domain == dname){
            l.push_back(fval);
        }
    }
    f.close();
}

void centralize(unordered_map<string, VectorXd> &x){
    VectorXd mean = VectorXd::Zero(D);
    VectorXd squared_mean = VectorXd::Zero(D);
    for (auto w = x.begin(); w != x.end(); ++w){
        mean += w->second;
        squared_mean += (w->second).cwiseProduct(w->second);
    }
    mean = mean / ((double) x.size());
    VectorXd sd = squared_mean - mean.cwiseProduct(mean);
    for (int i = 0; i < D; ++i){
        sd[i] = sqrt(sd[i]);
    }
    for (auto w = x.begin(); w != x.end(); ++w){
        VectorXd tmp = VectorXd::Zero(D);
        for (int i = 0; i < D; ++i){
            tmp[i] = (w->second)[i] - mean[i];
            if (sd[i] != 0)
                tmp[i] /= sd[i];
        }
        w->second = tmp;
    }
}

void initialize(){
    for (auto c = pivots.begin(); c != pivots.end(); ++c){
        cS[*c] = VectorXd::Random(D);
        cT[*c] = VectorXd::Random(D);
        grad_cS[*c] = VectorXd::Zero(D);
        grad_cT[*c] = VectorXd::Zero(D);
    }

    for (auto w = src_nonpivots.begin(); w != src_nonpivots.end(); ++w){
        wS[*w] = VectorXd::Random(D);
        grad_wS[*w] = VectorXd::Zero(D);
    }

    for (auto w = tgt_nonpivots.begin(); w != tgt_nonpivots.end(); ++w){
        wT[*w] = VectorXd::Random(D);
        grad_wT[*w] = VectorXd::Zero(D);
    }
    centralize(cS);
    centralize(cT);
    centralize(wS);
    centralize(wT);
}


void load_train_data(string fname, vector<instance> &train_data){
    ifstream train_file(fname.c_str());
    string pivot, pos_str, neg_str;
    while (train_file >> pivot >> pos_str >> neg_str){
        instance I;
        I.pivot = pivot;
        boost::split(I.positives, pos_str, boost::is_any_of(","));
        boost::split(I.negatives, neg_str, boost::is_any_of(","));
        train_data.push_back(I);
    }
    train_file.close();
}
    
void train(int epohs, double l, double alpha){
    fprintf(stderr, "\nTotal ephos to train = %d\n", epohs);
    fprintf(stderr, "Initial learning rate = %f\n", alpha);
    fprintf(stderr, "lambda = %f\n", l);
    fprintf(stderr, "Dim = %d\n", D);

    int total_train_errors, count, i, S_N, T_N;
    S_N = src_instances.size();
    T_N = tgt_instances.size();
    double score;

    VectorXd g1 = VectorXd::Zero(D);
    VectorXd g2 = VectorXd::Zero(D);
    VectorXd g3 = VectorXd::Zero(D);

    for (int t = 0; t < epohs; ++t){
        total_train_errors = count = i = 0;
        fprintf(stderr, "\nSource round = %d\n", t);
        for (auto inst = src_instances.begin(); inst != src_instances.end(); ++inst){
            fprintf(stderr, "\r%d: %2.2f", i, (100.0 * i) / (double) S_N);
            for (auto pos = inst->positives.begin(); pos != inst->positives.end(); ++pos){
                for (auto neg = inst->negatives.begin(); neg != inst->negatives.end(); ++neg){
                    score = cS[inst->pivot].dot(wS[*pos] - wS[*neg]);
                    count++;
                    if (score < 1){
                        total_train_errors++;

                        g1 = -cS[inst->pivot]; // dL/dwS
                        g2 = cS[inst->pivot]; // dL/dw*S
                        g3 = wS[*neg] - wS[*pos] + (l * (cS[inst->pivot] - cT[inst->pivot])); // dL/dcS
                        for (int k = 0; k < D; ++k){
                            wS[*pos][k] -= (alpha / sqrt(1.0 + grad_wS[*pos][k])) * g1[k];
                            wS[*neg][k] -= (alpha / sqrt(1.0 + grad_wS[*neg][k])) * g2[k];
                            cS[inst->pivot][k] -= (alpha / sqrt(1.0 + grad_cS[inst->pivot][k])) * g3[k];
                        }
                        grad_wS[*pos] += g1.cwiseProduct(g1);                          
                        grad_wS[*neg] += g2.cwiseProduct(g2);                    
                        grad_cS[inst->pivot] += g3.cwiseProduct(g3);                        
                    }
                    else{
                        g1 = l * (cS[inst->pivot] - cT[inst->pivot]);
                        for (int k = 0; k < D; ++k)
                            cS[inst->pivot][k] -= (alpha / sqrt(1.0 + grad_cS[inst->pivot][k])) * g1[k];
                        grad_cS[inst->pivot] += g1.cwiseProduct(g1);                        
                    }
                }
            }
            i++;
        }
        fprintf(stderr, "\n Source error rate = %f\n", (100 * total_train_errors) / (double) count);

        total_train_errors = count = i = 0;
        fprintf(stderr, "\rTarget round = %d\n", t);
        for (auto inst = tgt_instances.begin(); inst != tgt_instances.end(); ++inst){
            fprintf(stderr, "\r%d: %2.2f", i, (100.0 * i) / (double) T_N);
            for (auto pos = inst->positives.begin(); pos != inst->positives.end(); ++pos){
                for (auto neg = inst->negatives.begin(); neg != inst->negatives.end(); ++neg){
                    score = cT[inst->pivot].dot(wT[*pos] - wT[*neg]);
                    count++;
                    if (score < 1){
                        total_train_errors++;

                        g1 = -cT[inst->pivot]; // dL/dwT
                        g2 = cT[inst->pivot]; // dL/dw*T
                        g3 = wT[*neg] - wT[*pos] + (l * (cT[inst->pivot] - cS[inst->pivot])); // dL/dcT
                        for (int k = 0; k < D; ++k){
                            wT[*pos][k] -= (alpha / sqrt(1.0 + grad_wT[*pos][k])) * g1[k];
                            wT[*neg][k] -= (alpha / sqrt(1.0 + grad_wT[*neg][k])) * g2[k];
                            cT[inst->pivot][k] -= (alpha / sqrt(1.0 + grad_cT[inst->pivot][k])) * g3[k];
                        }
                        grad_wT[*pos] += g1.cwiseProduct(g1);
                        grad_wT[*neg] += g2.cwiseProduct(g2);                        
                        grad_cT[inst->pivot] += g3.cwiseProduct(g3);                        
                    }
                    else{
                        g1 = l * (cT[inst->pivot] - cS[inst->pivot]);
                        for (int k = 0; k < D; ++k)
                            cT[inst->pivot][k] -= (alpha / sqrt(1.0 + grad_cT[inst->pivot][k])) * g1[k];
                        grad_cT[inst->pivot] += g1.cwiseProduct(g1);                        
                    }
                }
            }
            i++;
        }
        fprintf(stderr, "\n Target error rate = %f", (100 * total_train_errors) / (double) count);
    }
}

void write_line(ofstream &reps_file, unordered_map<string,VectorXd>::iterator c, string label){
    reps_file << c->first << " " + label + " ";
    for (int i = 0; i < D; ++i)
        reps_file << c->second[i] << " ";
    reps_file << endl;
}

void save_model(string fname){
    ofstream reps_file;
    reps_file.open(fname);
    if (!reps_file){
        fprintf(stderr, "Failed to write reps to %s\n", fname.c_str());
        exit(1);
    } 
    for (auto c = cS.begin(); c != cS.end(); ++c)
        write_line(reps_file, c, "cS");
    for (auto c = cT.begin(); c != cT.end(); ++c)
        write_line(reps_file, c, "cT");
    for (auto w = wS.begin(); w != wS.end(); ++w)
        write_line(reps_file, w, "wS");
    for (auto w = wT.begin(); w != wT.end(); ++w)
        write_line(reps_file, w, "wT");
    reps_file.close();
}

  

int main(int argc, char *argv[]){
    int no_threads = 100;
    omp_set_num_threads(no_threads);
    setNbThreads(no_threads);
    initParallel(); 

    if (argc == 1) {
        fprintf(stderr, "usage: ./reps --dim=dimensionality --model=model_fname --source=source \
                            --target=target --l=lambda --alpha=alpha --ephos=rounds\n"); 
        return 0;
    }
    parse_args::init(argc, argv); 
    string source = parse_args::get<string>("--source");
    string target = parse_args::get<string>("--target");
    string fpath = "../work/" + source + "-" + target;

    D = parse_args::get<int>("--dim");
    int epohs = parse_args::get<int>("--epohs");
    double lambda = parse_args::get<double>("-l");
    double alpha = parse_args::get<double>("--alpha");
    string model = fpath + "/" + parse_args::get<string>("--model");

    cout << "Source = " << source << endl;
    cout << "Target = " << target << endl;
    load_pivots(fpath + "/pivots", pivots, "P");
    cout << "Total no. of pivots = " << pivots.size() << endl;
    load_pivots(fpath + "/nonpivots", src_nonpivots, "S");
    cout << "Total no. of src non-pivots = " << src_nonpivots.size() << endl;
    load_pivots(fpath + "/nonpivots", tgt_nonpivots, "T");
    cout << "Total no. of tgt non-pivots = " << tgt_nonpivots.size() << endl;
    initialize();
    load_train_data(fpath+"/source.train", src_instances);
    cout << "Total no. of source train instances = " << src_instances.size() << endl;
    load_train_data(fpath+"/target.train", tgt_instances);
    cout << "Total no. of target train instances = " << tgt_instances.size() << endl;
    train(epohs, lambda, alpha);
    save_model(model);
    return 0;

}