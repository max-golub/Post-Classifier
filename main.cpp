// Project UID db1f506d06d84ab787baf250c265e24e
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <string>
#include <sstream>
#include <climits>
#include <set>
#include <cmath>
#include "csvstream.h"

using namespace std;

set<string> unique_words(const string &str) {
    istringstream source(str);
    set<string> words;
    string word;
    // Read word by word from the stringstream and insert into the set
    while (source >> word) {
        words.insert(word);
    }
    return words;
}

class Classifier {
  private: 
    csvstream & csv;
    bool is_debug;
    int total_posts;
    int total_num_words;
    map<string, int> label_to_num_posts;
    map<string, int> word_to_num_posts;
    map<pair<string, string>, int> label_word_pair_to_num_posts;

    void parse_input() {
        if(is_debug) {
            cout << "training data:" << endl;
        }
        map<string, string> row;
        while(csv >> row) {
            if(is_debug) {
                cout << "  " << "label = " << row["tag"] 
                << ", content = " << row["content"] << endl;
            }
            ++total_posts;
            label_to_num_posts[row["tag"]]++;
            set<string> cur_words = unique_words(row["content"]);
            for(string word : cur_words) {
                auto find = word_to_num_posts.find(word);
                if(find == word_to_num_posts.end()) {
                    ++total_num_words;
                }
                word_to_num_posts[word]++;
                pair<string, string> new_key(row["tag"], word);
                label_word_pair_to_num_posts[new_key]++;
            }
        }
        cout << "trained on " << total_posts << " examples" << endl;
        if(is_debug) {
            cout << "vocabulary size = " << total_num_words << endl;
        }
        cout << endl;

        if(is_debug) {
        cout << "classes:" << endl;
        }
        if(is_debug) {
            for(auto pair : label_to_num_posts) {
                cout << "  " << pair.first << ", " 
                << pair.second << " examples, log-prior = "
                << log((double)pair.second / total_posts) << endl;
            }
        }
    }

  public:
    Classifier(csvstream & csv_in, const bool & debug_in) 
    : csv(csv_in), is_debug(debug_in), total_posts(0), total_num_words(0) {
        parse_input();
    }

    double make_prediction_given_label(const string & tag, 
      const set<string> & words, stringstream & params) {
        double sum = log((double)label_to_num_posts[tag] / total_posts);
        for(string word : words) {
            double cur_add;
            pair<string, string> cur_key(tag, word);
            auto find_label_word = label_word_pair_to_num_posts.find(cur_key);
            if(find_label_word == label_word_pair_to_num_posts.end()) {
                auto find_word = word_to_num_posts.find(word);
                if(find_word == word_to_num_posts.end()) {
                    cur_add = (log((double)1 / total_posts));
                }
                else {
                    cur_add = (log((double)find_word->second / total_posts));
                }
            }
            else {
                cur_add = (log((double)find_label_word->second/label_to_num_posts[tag]));
                if(is_debug) {
                    params << "  " << find_label_word->first.first
                     << ":" << find_label_word->first.second
                    << ", count = " << find_label_word->second 
                    << ", log-likelihood = " << cur_add << endl;
                }
            }
            sum += cur_add;
        }
        return sum;
    }
    pair<string,double> classify_post(const set<string> & words, stringstream & params) {
        double max_score = -numeric_limits<double>::max();
        string max_score_label = "";
        for(auto pair : label_to_num_posts) {
            double cur_val = make_prediction_given_label(pair.first, words, params);
            if(cur_val > max_score) {
                max_score = cur_val;
                max_score_label = pair.first;
            }
        }
        pair<string, double> class_log(max_score_label, max_score);
        return class_log;
    }
    void find_log_likelies(map<pair<string, string>, double> & label_word2log, 
      stringstream & params) {
        for(auto pair : label_word_pair_to_num_posts) {
            double cur_log = (log((double)pair.second
             / label_to_num_posts[pair.first.first]));
            label_word2log[pair.first] = cur_log;
            params << "  " << pair.first.first << ":" << pair.first.second
             << ", count = " << pair.second << ", log-likelihood = " << cur_log<<endl;
        }
    }
};

void print_tests(bool is_debug, Classifier & bag_of_words, csvstream & csv2) {
    map<string, string> row;
    stringstream params;
    params.precision(3);
    if(is_debug) {
        params << "classifier parameters:" << endl;
        map<pair<string, string>, double> label_word_pair_to_log_score;
        bag_of_words.find_log_likelies(label_word_pair_to_log_score, params);
        cout << params.str();
        cout << endl;
    }

    int num_test_posts = 0;
    int num_correct_class = 0;
    stringstream test_data;
    test_data.precision(3);
    test_data << "test data:" << endl;
    while(csv2 >> row) {
        pair<string, double> class_log_score;
        ++num_test_posts;
        class_log_score=bag_of_words.classify_post(unique_words(row["content"]),params);
        if(class_log_score.first == row["tag"]) {
            ++num_correct_class;
        }
        test_data << "  " << "correct = " << row["tag"] << ", predicted = " 
        << class_log_score.first
         << ", log-probability score = " << class_log_score.second << endl;
        test_data << "  " << "content = " << row["content"] << endl;
        test_data << endl;
    }

    cout << test_data.str();

    cout << "performance: " << num_correct_class << " / " << num_test_posts
     << " posts predicted correctly" << endl;
}

int main(int argc, char* argv[]) {
    cout.precision(3);
    if(!(argc == 4) && !(argc == 3)) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 1;
    }
    bool is_debug = false;
    if(argc == 4) {
        string debug(argv[3]);
        if(debug != "--debug") {
            cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
            return 1;
        }
        is_debug = true;
    }
    ifstream fin;
    string infile = argv[1];
    fin.open(infile);
    if(!fin.is_open()) {
        cout << "Error opening file: " << infile << endl;
        return 1;
    }
    fin.close();

    ifstream fin2;
    string outfile = argv[2];
    fin2.open(outfile);
    if(!fin2.is_open()) {
        cout << "Error opening file: " << outfile << endl;
        return 1;
    }
    fin2.close();

    csvstream csv1(infile);
    csvstream csv2(outfile);
    Classifier bag_of_words(csv1, is_debug);
    print_tests(is_debug, bag_of_words, csv2);

    return 0;
}



