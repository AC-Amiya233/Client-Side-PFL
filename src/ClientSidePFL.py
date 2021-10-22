import copy
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from tqdm import tqdm

from dataset import *
from utils import *
from models import *

def reset_random_seed():
    np.random.seed(0)
    # np.random.seed(int(time.time()))

class Server():
    def __init__(self, clients):
        self.usr_idx = [i for i in range(clients)]
        self.usr_models = [CNNMnist() for i in range(clients)]
        self.usr_states = [0 for i in range(clients)]
        self.usr_participate = [i for i in range(clients)]

class Client():
    def __init__(self):
        self.model = CNNMnist()
        self.data = []
        self.optimizer = []

    def request_other_models(self, server: Server):
        pass

    def handle_self_models(self, server: Server):
        pass


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def compute_global_model(global_model: list, usr_models: list):
    usr_state_dicts = [usr_models[idx].state_dict() for idx in range(len(usr_models))]
    avg = average_weights(usr_state_dicts)
    global_model[0].load_state_dict(avg)
    return avg


def evaluate_global_model(model, test_loaders):
    loss_list = []
    correct_list = []
    samples_list = []
    for test_loader in test_loaders:
        loss, correct, samples = 0., 0., 0.
        batch_count = 0
        for batch_count, batch in enumerate(test_loader):
            img, label = tuple(batch)
            pred = model(img)
            loss += criteria(pred, label).item()
            correct += pred.argmax(1).eq(label).sum().item()
            samples += len(label)
        loss_list.append(loss / (batch_count + 1))
        correct_list.append(correct)
        samples_list.append(samples)
    avg_loss = np.mean(loss_list)
    avg_correct = np.mean(correct_list)
    avg_samples = np.mean(samples_list)
    return avg_loss, avg_correct / avg_samples


def evaluate_model(model, test_loader):
    loss, correct, samples = 0., 0., 0.
    batch_count = 0
    for batch_count, batch in enumerate(test_loader):
        img, label = tuple(batch)
        pred = model(img)
        loss += criteria(pred, label).item()
        correct += pred.argmax(1).eq(label).sum().item()
        samples += len(label)
    loss /= batch_count + 1
    return loss, correct / samples


if __name__ == '__main__':
    set_logger()
    logging.info('Client Side PFL Training Starts')

    # configs:
    clients = 10
    batch_size = 40
    select = 10
    data = 'mnist'
    path = '../data'
    criteria = torch.nn.CrossEntropyLoss()
    global_epoch = 10
    local_epoch = 1
    active_local_sv = False
    active_local_loss = True
    # R = 1
    R = 3 * clients
    # training needs:
    usr_idx = [i for i in range(clients)]
    usr_models = [CNNMnist() for i in range(clients)]
    usr_states = [0 for i in range(clients)]
    usr_participate = [i for i in range(clients)]
    usr_participate_counter = [0 for i in range(clients)]
    usr_optimizers = [torch.optim.SGD(
            usr_models[i].parameters(), lr=5e-3, momentum=.9, weight_decay=5e-5
        ) for i in range(clients)]
    # logging.info(usr_optimizers)
    usr_dataset_loaders, usr_val_loaders, usr_test_loaders = gen_random_loaders(data, path, clients, 40, 4)
    # usr_dataset_loaders, test_loader = gen_dataloaders_with_majority(data, path, clients, 40)
    # data_loaders = gen_specific_major_loaders(data, path, clients, 40)
    logging.info('User Dataset Loader Details {}'.format(len(usr_dataset_loaders)))
    global_model = [CNNMnist()]

    # eval/sv needs:
    local_update_global_acc_list = []
    loss_update_global_acc_list = []
    sv_update_global_acc_list = []

    local_acc_list = [0 for i in range(clients)]

    sv = []

    # for plot
    local_update_models_dict = {i : [] for i in range(clients)}
    local_update_acc_dict = {i : [] for i in range(clients)}
    local_update_loss_dict = {i : [] for i in range(clients)}
    sv_update_models_dict = {i : [] for i in range(clients)}
    sv_update_acc_dict = {i: [] for i in range(clients)}
    loss_update_models_dict = {i : [] for i in range(clients)}
    loss_update_acc_dict = {i : [] for i in range(clients)}

    for global_round in range(1, global_epoch + 1):
        usr_states = [0 for i in range(clients)]
        # 1 - Select users
        reset_random_seed()
        selected = random.sample(usr_idx, select)
        logging.info('Global Round {} - Participants {} ({}/{})'.format(global_round, selected, len(selected), clients))
        for idx in selected:
            usr_states[idx] = 1
            usr_participate[idx] = global_round
            usr_participate_counter[idx] += 1
        global_acc = sum(local_acc_list) / len(local_acc_list)
        local_update_global_acc_list.append(global_acc)
        logging.critical('Global Average Accuracy {}'.format(global_acc))

        # 2 - Local update
        for idx in range(clients):
            if usr_states[idx] == 1:
                # Accuracy Before Local update #
                running_loss, running_correct, running_samples = 0., 0., 0.
                batch_count = 0
                for batch_count, batch in enumerate(usr_test_loaders[idx]):
                    img, label = batch
                    pred = usr_models[idx](img)
                    running_loss += criteria(pred, label).item()
                    running_correct += pred.argmax(1).eq(label).sum().item()
                    running_samples += len(label)
                logging.info('[Before Training]Local {} loss {} - Correct {} / Samples {} = {}%'.format(idx, running_loss / batch_count, running_correct, running_samples, running_correct / running_samples * 100))

                # Local update #
                for local_round in range(1, local_epoch + 1):
                    usr_models[idx].train()
                    usr_optimizers[idx].zero_grad()
                    batch = next(iter(usr_dataset_loaders[idx]))
                    img, label = tuple(batch)
                    pred = usr_models[idx](img)
                    loss = criteria(pred, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(usr_models[idx].parameters(), 50)
                    usr_optimizers[idx].step()

                # Accuracy After Local update #
                running_loss, running_correct, running_samples = 0., 0., 0.
                for batch_count, batch in enumerate(usr_test_loaders[idx]):
                    img, label = batch
                    pred = usr_models[idx](img)
                    running_loss += criteria(pred, label).item()
                    running_correct += pred.argmax(1).eq(label).sum().item()
                    running_samples += len(label)
                logging.info('[After Training]Local {} loss {} - Correct {} / Samples {} = {}%'.format(idx, running_loss / batch_count, running_correct, running_samples, running_correct / running_samples * 100))
                # store mid-results
                local_update_models_dict[idx].append(usr_models[idx])
                local_update_acc_dict[idx].append(running_correct / running_samples)
                local_update_loss_dict[idx].append(running_loss / batch_count)
                local_acc_list[idx] = running_correct / running_samples

        # sv/loss update
        for idx in range(clients):
            if usr_states[idx] == 1:
                if active_local_sv:
                    # request other clients model
                    logging.info('[SV] Start SV Computation')
                    local_only_acc = local_update_acc_dict[idx][-1]
                    logging.info('[SV] {} Only with Accuracy {}%'.format(idx, local_only_acc * 100))
                    perm_list = []
                    perm_list += list(itertools.permutations(np.arange(clients), clients))
                    # reset_random_seed()
                    r_perm_index = np.random.choice([i for i in range(len(perm_list))], R, replace=False)
                    r_perm = []
                    for i in r_perm_index:
                        r_perm.append(perm_list[i])
                    logging.info(r_perm)
                    logging.info('[SV] Random Select permutation {} in {}'.format(len(r_perm), len(perm_list)))
                    # start sv part
                    evaluate_sv_info_dict = {i: [] for i in range(clients)}
                    evaluated_sv = []
                    for item in r_perm:
                        logging.critical('[SV] Processing {}'.format(item))
                        evaluated_sv = []
                        weights_queue = []
                        node_queue = []
                        # TODO MARKER - INIT ACCURACY
                        acc_history = [0.]
                        for member in item:
                            node_queue.append(member)
                            weights_queue.append(usr_models[member].state_dict())
                            avg = average_weights(weights_queue)
                            global_model[0].load_state_dict(avg)
                            cur_loss, cur_acc = evaluate_model(global_model[0], usr_test_loaders[idx])
                            logging.info('[EVAL] {}({}) on {} with accuracy influence {}%'.format(member, len(node_queue), idx, (cur_acc - acc_history[-1]) * 100))
                            evaluate_sv_info_dict[member].append(cur_acc - acc_history[-1])
                            acc_history.append(cur_acc)
                    # Avg sv
                    for i in range(clients):
                        evaluated_sv.append(np.mean(evaluate_sv_info_dict[i]))
                    logging.critical('[SV] Eval SV: {}'.format(evaluated_sv))
                    # calculate weights based on sv

                    # only use positive sv clients
                    positive_idx = []
                    positive_sv = []
                    for i in range(clients):
                        if evaluated_sv[i] > 0:
                            positive_idx.append(i)
                            positive_sv.append(evaluated_sv[i])
                    # norm
                    positive_sv = positive_sv / sum(positive_sv)
                    logging.info('[FedFomo] Positive Index {}'.format(positive_idx))
                    logging.info('[FedFomo] Positive Weights {}'.format(positive_sv))

                    # TODO norm
                    weights = []
                    for i in range(clients):
                        if i in positive_idx:
                            weights.append(positive_sv[positive_idx.index(i)])
                        else :
                            weights.append(0)
                    logging.critical('[FedFomo] Weights {}'.format(weights))

                    base = copy.deepcopy(usr_models[idx].state_dict())
                    for key in base.keys():
                        if idx in positive_idx:
                            base[key] = weights[positive_idx.index(idx)] * base[key]
                        else:
                            logging.critical('[FedFomo] Index {} not in its own choose'.format(idx))
                            base[key] = 0 * base[key]

                    for key in base.keys():
                        for cur in range(clients):
                            w = weights[cur]
                            state = usr_models[cur].state_dict()
                            if cur != idx:
                                base[key] += w * state[key]
                    usr_models[idx].load_state_dict(base)
                    fomo_loss, fomo_acc = evaluate_model(usr_models[idx], usr_dataset_loaders[idx])
                    logging.critical('[FedFomo] New Model Local Accuracy {}%'.format(fomo_acc * 100))
                    loss_update_acc_dict[idx].append(fomo_acc)

                if active_local_loss:
                    logging.info('[ICLR] Start Loss Computation')
                    local_only_acc = local_update_acc_dict[idx][-1]
                    local_only_loss = local_update_loss_dict[idx][-1]
                    logging.info('[ICLR] {} Only with Accuracy {}% & Loss {}'.format(idx, local_only_acc * 100, local_only_loss))

                    # calculate weights
                    weights = []

                    loss_diff = []
                    loss_diff.append(local_update_loss_dict[idx][-1])
                    models_diff = [usr_models[idx].state_dict() for i in range(clients)]
                    for request in range(clients):
                        if request != idx:
                            cur_loss, cur_acc = evaluate_model(usr_models[request], usr_test_loaders[idx])
                            logging.info('[ICLR-EVAL] {} on {} dataset with accuracy {}% & Loss {}'.format(request, idx, cur_acc * 100, cur_loss))
                            loss_diff.append(loss_diff[0] - cur_loss)
                            for key in models_diff[0].keys():
                                models_diff[request][key] = models_diff[0][key] - usr_models[request].state_dict()[key]


                    loss_diff = loss_diff[1:]
                    norm_models_diff = []
                    for request in range(1, clients):
                        plant = []
                        for key in models_diff[0].keys():
                            plant += torch.reshape(models_diff[request][key], (-1,)).detach().numpy().tolist()
                        norm_plant = np.linalg.norm(plant)
                        norm_models_diff.append(norm_plant)
                        logging.info('[ICLR-EVAL] {} on {} dataset with models different {}'.format(request, idx, norm_plant))

                    for item in range(0, clients - 1):
                        weights.append(loss_diff[item] / norm_models_diff[item])

                    # exclude negative value
                    for item in range(len(weights)):
                        weights[item] = max(0, weights[item])
                    # normalize weights
                    base = sum(weights)
                    if base == 0:
                        loss_update_models_dict[idx].append(usr_models[idx].state_dict())
                    else:
                        for item in range(len(weights)):
                            weights[item] = weights[item] / base

                        # update local model
                        local_update_model = usr_models[idx].state_dict()
                        for request in range(0, clients - 1):
                            if request != idx:
                                for key in local_update_model.keys():
                                    local_update_model[key] += weights[request] * models_diff[request][key]
                        usr_models[idx].load_state_dict(local_update_model)
                        loss_update_models_dict[idx].append(local_update_model)

                    iclar_loss, iclr_acc = evaluate_model(usr_models[idx], usr_test_loaders[idx])
                    logging.critical('[ICLR - EVAL] ICLR Accuracy {}% Loss {}'.format(iclr_acc * 100, iclar_loss))
                    loss_update_acc_dict[idx].append(iclr_acc)


    fig1 = plt.figure('Fig 1', figsize=(20, 10))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[0], color='red', linestyle='--',
             marker='x')
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[0], color='blue', linestyle='--',
             marker='x')
    ax1.set_title('Client 0')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[1], color='red', linestyle='--',
             marker='x')
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[1], color='blue', linestyle='--',
             marker='x')
    ax2.set_title('Client 1')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')

    fig2 = plt.figure('Fig 2', figsize=(20, 10))
    ax1 = fig2.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[2], color='red', linestyle='--',
             marker='x')
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[2], color='blue', linestyle='--',
             marker='x')
    ax1.set_title('Client 2')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2 = fig2.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[3], color='red', linestyle='--',
             marker='x')
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[3], color='blue', linestyle='--',
             marker='x')
    ax2.set_title('Client 3')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')

    fig3 = plt.figure('Fig 3', figsize=(20, 10))
    ax1 = fig3.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[4], color='red', linestyle='--',
             marker='x')
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[4], color='blue', linestyle='--',
             marker='x')
    ax1.set_title('Client 4')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2 = fig3.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[5], color='red', linestyle='--',
             marker='x')
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[5], color='blue', linestyle='--',
             marker='x')
    ax2.set_title('Client 5')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')

    fig4 = plt.figure('Fig 4', figsize=(20, 10))
    ax1 = fig4.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[6], color='red', linestyle='--',
             marker='x')
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[6], color='blue', linestyle='--',
             marker='x')
    ax1.set_title('Client 6')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2 = fig4.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[7], color='red', linestyle='--',
             marker='x')
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[7], color='blue', linestyle='--',
             marker='x')
    ax2.set_title('Client 7')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')

    fig5 = plt.figure('Fig 5', figsize=(20, 10))
    ax1 = fig5.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[8], color='red', linestyle='--',
             marker='x')
    ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[8], color='blue', linestyle='--',
             marker='x')
    ax1.set_title('Client 8')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')

    ax2 = fig5.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[9], color='red', linestyle='--',
             marker='x')
    ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[9], color='blue', linestyle='--',
             marker='x')
    ax2.set_title('Client 9')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')

    plt.show()

    with open('{}.csv'.format('iclr_update_accuracy'), 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in loss_update_acc_dict.items()]

    with open('{}.csv'.format('iclr_local_update_accuracy'), 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in local_update_acc_dict.items()]