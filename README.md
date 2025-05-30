# NoMoreNoise
Hi professore, I and Mattia (my co-worker) have worked a lot for this homework, we have tried to reach the baseline in various ways. Read the requirements.txt file for the requirements needed.

## Early Approches
Former we tried to do an hyper-parameter search on the model, changing the layer, dimensions, dropout and the gnn model. We tried also to use a lot of different loss function, implemented by us, as SOP, adaptiveSOP, NoisyCE, Asymmetric CE, Symmetric CE. We firstly tried gcn, and then we finally reached the baseline with the gin-virtual, with 3 layers, 300 as dimension, and 0.5 as dropout for A,B, D and for C 0.7, and we reached 0.833 . 

## Final Approach
Latter we tried multiple days to make the model work and reach the real baseline, but we struggled a lot and decide to change out approach. We did a fine-tune on an existing [model](https://github.com/cminuttim/Learning-with-Noisy-Graph-Labels-Competition-IJCNN_2025), and did the fine-tuning on each dataset, with the Noisy CE with noisy probability = 0.2 . 

## How to run

Run it inside your console, located in the directory

`python main.py --test_path <path_to_test.json.gz> --train_path <optional_path_to_train.json.gz> --num_cycles 5 --pretrain_paths <path_to_pretrain.txt> `

To obtain the csv output, without training or after it, run this:

`python main.py --test_path <path_to_test.json.gz> `

