import pandas as pd
import numpy as np
import wandb
import argparse
import random
import itertools
import os

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names, get_three_classes_dataset_names, get_four_classes_dataset_names, get_multiclasses_dataset_names
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=1000, help='Choose the number of ROCKET kernels to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

# Constants
n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds
n_important_weights = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def main():
    #list_of_datasets = get_binary_dataset_names()
    #list_of_datasets = get_three_classes_dataset_names()
    #list_of_datasets = get_four_classes_dataset_names()
    list_of_datasets = get_multiclasses_dataset_names()
    #list_of_datasets = ['BeetleFly'] #2 classes
    #list_of_datasets = ['CBF'] #3 classes
    #list_of_datasets = ['ACSF1'] #10 classes

    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init()

    # All the runs
    run_names = []
    runs = wandb.Api().runs("mlgroup/DECENTRALIZED FROCKS")
    for run in runs:
        run_names.append(run.name)
    #print("RUN NAMES DONE: ", run_names)
    wandb.finish()

    for ds_name in list_of_datasets:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        for experiment_seed in EXPERIMENT_SEEDS:
            print('SEED', experiment_seed)
            rng = np.random.RandomState(experiment_seed)
            np.random.seed(experiment_seed)
            random.seed(experiment_seed)
            os.environ['PYTHONHASHSEED'] = str(experiment_seed)

            if args.debug:
                wandb.init(mode="disabled")
            else:
                entity = "mlgroup"
                project = "DECENTRALIZED FROCKS"
                run_name = f"{ds_name}_{n_kernels}_KERNELS_{experiment_seed}"
                tags = ["Binary"]
                if run_name in run_names:
                    print(f"Experiment {run_name} already executed.")
                    continue
                else:
                    wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

            s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng)
            s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng)

            # Initialize different seeds for all clients
            seeds = np.arange(n_clients*n_kernels).reshape(n_clients, n_kernels)

            ts_length = len(X_train[0])
            kernels = [[RocketKernel(seed=int(seed), ts_length=ts_length, ppv_only=True) for seed in seeds[client_id]] for client_id in range(n_clients)]

            intercept = 0
            new_used_seeds = []
            all_used_seeds = []
            c_all_used_seeds = []
            prev_used_seeds = None  # Track if seeds are the same for two consecutive rounds
            
            for epoch in range(n_rounds):
                print("ROUND:", epoch)
                weights = []
                
                for client_id in range(n_clients):
                    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                    x_test, y_test = s_X_test[client_id], s_Y_test[client_id]
                    if epoch == 0:
                        if client_id == 0:
                            #print("Client", client_id, "seeds", seeds[client_id])
                            K = kernels[client_id]
                        else:
                            used_seeds = new_seeds
                            #print("Client", client_id, "seeds", used_seeds)
                            K = [RocketKernel(seed, ts_length) for seed in used_seeds]
                    else:
                        K = [RocketKernel(seed, ts_length) for seed in used_seeds]

                    # Transform data
                    x_train_transformed = np.concatenate([k.transform(x_train) for k in K], axis=1)
                    x_test_transformed = np.concatenate([k.transform(x_test) for k in K], axis=1)

                    if epoch == 0:
                        if client_id == 0:
                            lm = MLPClassifier(hidden_layer_sizes=(), solver='adam', activation='logistic', alpha = 0, max_iter=5000, learning_rate_init=0.001, batch_size=2, random_state=rng)
                        else:
                            lm = fitted_lm
                    lm.fit(x_train_transformed, y_train)
                    #print(lm.coefs_)
                    client_accuracy = accuracy_score(lm.predict(x_test_transformed), y_test)
                    if n_classes == 2:
                        client_f1 = f1_score(lm.predict(x_test_transformed), y_test)
                    else:
                        client_f1 = f1_score(lm.predict(x_test_transformed), y_test, average = 'macro')
                    print("Client: ", client_id, 
                        "accuracy: ", client_accuracy,
                        "f1-score: ", client_f1)
                    wandb.run.summary["step"] = epoch
                    wandb.log({f'Client {client_id} Test accuracy with {n_kernels} kernels': client_accuracy,
                               f'Client {client_id} Test F1 with {n_kernels} kernels': client_f1})

                    # Find best weights
                    w = lm.coefs_[0]
                    intercept += lm.intercepts_[0]
                    if n_classes == 2:
                        highest_weight_indices = np.argsort(-np.abs(w.flatten()))[:n_important_weights]
                        #print(highest_weight_indices)
                    else:
                        
                        highest_weight_indices = np.argsort(-np.abs(w), axis=None)[:n_important_weights]
                        highest_weight_indices = highest_weight_indices // w.shape[1]
                        highest_weight_indices = highest_weight_indices.tolist()
                        '''
                        row_sums = np.sum(np.abs(w), axis=1)
                        # Trovare gli indici delle righe con le somme più alte
                        highest_weight_indices = np.argsort(-row_sums)[:n_important_weights]
                        '''
                    weights.append(w[highest_weight_indices])
                    if epoch == 0:
                        if client_id == 0:
                            used_seeds = np.array(seeds[client_id])
                    for idx in highest_weight_indices:
                        if np.array(used_seeds)[idx] not in new_used_seeds:
                            new_used_seeds.append(used_seeds[idx])
                    
                    # Number of new seeds that have to be generated
                    num_new_seeds = n_kernels - len(new_used_seeds)

                    new_numbers = []
                    current_number = n_kernels
                    while len(new_numbers) < num_new_seeds:
                        if current_number not in used_seeds and current_number not in all_used_seeds:
                            new_numbers.append(current_number)
                        current_number += 1
                    
                    # Add new seeds
                    new_seeds = new_used_seeds + new_numbers
                    # Save used seeds in order to use different ones in next rounds
                    c_all_used_seeds.extend(used_seeds)
                    c_all_used_seeds = list(set(c_all_used_seeds))  # Rimuove duplicati
                    
                    #print("Used seeds", used_seeds)
                    #print("Important seeds (new used seeds)", new_used_seeds, "\n")

                    fitted_lm = lm

                all_used_seeds = c_all_used_seeds
                #print("All used seeds: ", all_used_seeds)

                # Check for convergence (two consecutive rounds with same seeds)
                if prev_used_seeds is not None and set(new_used_seeds) == set(prev_used_seeds):
                    print("Seeds are equal for two consecutive rounds. Rounds for convergence:", epoch + 1, " Exit.")
                    print(new_used_seeds)
                    wandb.log({f'Rounds for convergence': epoch+1, f'Convergence kernels': len(new_used_seeds)})
                    #print(fitted_lm.coefs_)
                    '''
                    for i in range(len(fitted_lm.coefs_)):
                        if fitted_lm.coefs_[i].shape[0] >= len(fitted_lm.coefs_):
                            fitted_lm.coefs_[i] = fitted_lm.coefs_[i][:len(new_used_seeds), :]
                    '''
                    break
                prev_used_seeds = new_used_seeds.copy()  # Update prev_used_seeds

            intercept = np.array(intercept) / n_clients
            ag_lin = MLPClassifier(hidden_layer_sizes=(), solver='adam', activation='logistic', alpha = 0, max_iter=5000, learning_rate_init=0.001, batch_size=2, random_state=rng)

            x_train_transformed = transform_seeds(x_train, used_seeds, ts_length)
            # partial_fit is used only to initialize some internals such as LabelBinarizer(), but then I will copy the previous weights in the final model
            ag_lin.partial_fit(x_train_transformed, y_train, classes=np.unique(y_train))
            
            #ag_lin.is_fitted = True
            ag_lin.coefs_ = fitted_lm.coefs_
            ag_lin.intercepts_ = intercept
            #ag_lin.classes_ = np.arange(n_classes)
            ag_lin.n_layers_ = len(fitted_lm.coefs_) + 1  # Number of layers (input + hidden/output)
            ag_lin.n_outputs_ = fitted_lm.n_outputs_
            ag_lin.out_activation_ = 'logistic'
            print("Convergence reached with:", len(new_used_seeds), "seeds")
            print(used_seeds)
            X_test_transformed = transform_seeds(X_test, used_seeds, ts_length)
            # Evaluate on test
            #print(ag_lin.coefs_)
            final_prediction = ag_lin.predict(X_test_transformed)
            final_accuracy = accuracy_score(final_prediction, Y_test)
            if n_classes == 2:
                final_f1 = f1_score(final_prediction, Y_test)
            else:
                final_f1 = f1_score(final_prediction, Y_test, average = 'macro')
            print('Final scores. Accuracy: ', final_accuracy,
                "f1-score: ", final_f1)
            wandb.log({f'Final accuracy': final_accuracy,
                        f'Final F1': final_f1})
            wandb.finish()

if __name__ == '__main__':
    main()