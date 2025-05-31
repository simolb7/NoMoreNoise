# NoMoreNoise

![img](ImageTeaser.png)

Dear professor, this is our (Simone La Bella 1995847 and Mattia Di Marco 2019367) work for the hackaton of Deep Learning, we have tried to reach the baseline in various ways. Read the requirements.txt file for the requirements needed.

## Early Approches
Former we tried to do an hyper-parameter search on the model provided on kaggle, changing the layer, dimensions, dropout and the gnn architecture. We tried also to use a lot of different loss function, implemented by us, as SOP, adaptiveSOP, NoisyCE, Asymmetric CE, Symmetric CE. We treid first with gcn, both with and without virtual node, with poor results. At the end we reached the baseline using the gin-virtual, with 3 layers, 300 as embedding dimension, 0.5 as drop_ratio on A,B, D and 0.7 for C, and the classical CrossEntropyLoss as loss fuction. With this kind of architecture we manage to reach a f1 score of 0.833.

## Final Approach
Overcoming the real baseline proved to be much more complex. We tried multiple days to make the model work and reach the real baseline, but we struggled a lot and decide to change our approach. So, we did a fine-tune on an existing [model](https://github.com/cminuttim/Learning-with-Noisy-Graph-Labels-Competition-IJCNN_2025). The model used works thanks to the so called Variational Graph Autoencoder (VGAE), and is used to filter noisy data by retaining only real pattern in the bottleneck. For each dataset we started by using the pretrained model that can be find in the git-repo but we decided to change the loss fuction used, in fact we use the NoisyCrossEntropyLoss, provided in the kaggle file, with a p_noise of 0.2. The implementation of this loss fuction can be found in the source/models.py file.

## How to run

Run it inside your console, located in the directory

`python main.py --test_path <path_to_test.json.gz> --train_path <optional_path_to_train.json.gz> --num_cycles 5 --pretrain_paths <path_to_pretrain.txt> `

To obtain the csv output, without training or after it, run this:

`python main.py --test_path <path_to_test.json.gz> `
