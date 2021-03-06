import copy
import heapq
import itertools
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from tqdm import tqdm

from dataset import *
from utils import *
from models import *

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
    set_seed(40)
    logging.info('Client Side PFL Training Starts')

    # todo Configs:
    task_repeat_time = 3
    global_epoch = 20
    local_epoch = 10
    batch_size = 40
    classes_per_client = 2

    loss_repeating_val = 10

    # participant
    clients = 15
    select = 15

    # dataset
    data = 'cifar10'
    path = '../data'
    # loss function
    criteria = torch.nn.CrossEntropyLoss()

    # download models
    active_partial_download = True
    download = 5
    exploit = 4
    explore = 1

    # todo SV for personalization
    active_local_sv = True
    sv_eval_method = 'acc'
    whether_free_space = False
    whether_delta = False
    if active_partial_download:
        R = 3 * (download + 1)
    else:
        R = 3 * clients

    # todo FedFomo for personalization
    active_local_loss = False

    multi_task_avg_accuracy_list = [0 for i in range(task_repeat_time)]

    for task in range(task_repeat_time):
        # training needs:
        prob_download_dict = {idx: [float('-inf') if i == idx else 0 for i in range(clients)] for idx in range(clients)}

        usr_idx = [i for i in range(clients)]
        # client model
        if data == 'mnist':
            usr_models = [CNNMnist() for i in range(clients)]
        elif data == 'fashion-mnist':
            usr_models = [CNNFashion_Mnist() for i in range(clients)]
        elif data == 'cifar10' or 'cifar100':
            usr_models = [CNNCifar() for i in range(clients)]

        usr_states = [0 for i in range(clients)]
        usr_participate = [i for i in range(clients)]
        usr_participate_counter = [0 for i in range(clients)]

        # todo optimizer setting
        usr_optimizers = [torch.optim.SGD(usr_models[i].parameters(), lr=0.1, momentum=0, weight_decay=1e-4) for i in range(clients)]
        # usr_optimizers = [torch.optim.Adam(usr_models[i].parameters(), lr=0.001) for i in range(clients)]
        # logging.info(usr_optimizers)

        usr_dataset_loaders, usr_val_loaders, usr_test_loaders = gen_random_loaders(data, path, clients, batch_size, classes_per_client)
        # usr_dataset_loaders, test_loader = gen_dataloaders_with_majority(data, path, clients, 40)
        # data_loaders = gen_specific_major_loaders(data, path, clients, 40)
        logging.info('User Dataset Loader Details {}'.format(len(usr_dataset_loaders)))

        # the global model for sv evaluation
        global_model = []
        if data == 'mnist':
            global_model = [CNNMnist()]
        elif data == 'fashion-mnist':
            global_model = [CNNFashion_Mnist()]
        elif data == 'cifar10' or 'cifar100':
            global_model = [CNNCifar()]


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

        download_history = {i: [] for i in range(clients)}
        selection_history = {i: [] for i in range(clients)}
        selection_weighst_history = {i: [] for i in range(clients)}
        before_selection_acc_history = {i: [] for i in range(clients)}
        after_selection_acc_history = {i: [] for i in range(clients)}
        before_selection_loss_history = {i: [] for i in range(clients)}
        after_selection_loss_history = {i: [] for i in range(clients)}

        for global_round in range(1, global_epoch + 1):
            usr_states = [0 for i in range(clients)]
            # 1 - Select users
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
                        for i, batch_data in enumerate(usr_dataset_loaders[idx], 0):
                            usr_models[idx].train()
                            usr_optimizers[idx].zero_grad()
                            img, label = batch_data
                            pred = usr_models[idx](img)
                            loss = criteria(pred, label)
                            loss.backward()
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
                        local_only_loss = local_update_loss_dict[idx][-1]
                        logging.critical('[SV] {} Only with Accuracy {}%'.format(idx, local_only_acc * 100))

                        # MAKER
                        before_selection_acc_history[idx].append(local_only_acc)
                        before_selection_loss_history[idx].append(local_only_loss)

                        # compute probs:
                        participate = [i for i in range(clients) if i != idx]
                        num_pull_models = clients - 1

                        # todo choose to download other models
                        if active_partial_download:
                            downloaded_usrs = []
                            topk_idx = heapq.nlargest(exploit, range(len(prob_download_dict[idx])), prob_download_dict[idx].__getitem__)
                            downloaded_usrs += topk_idx
                            while len(downloaded_usrs) != download:
                                ch = random.choice(range(clients))
                                if ch not in downloaded_usrs and ch != idx:
                                    downloaded_usrs.append(ch)
                            logging.critical('[SV] Download {} from the server'.format(downloaded_usrs))

                            participate = downloaded_usrs
                            num_pull_models = len(downloaded_usrs)

                        backup = [i for i in participate]
                        download_history[idx].append(backup)

                        local_total_models_num = num_pull_models + 1
                        participate.append(idx)

                        perm_list = []
                        perm_list += list(itertools.permutations(participate, local_total_models_num))
                        r_perm_index = np.random.choice([i for i in range(len(perm_list))], R, replace=False)
                        r_perm = []
                        for i in r_perm_index:
                            r_perm.append(perm_list[i])
                        logging.info(r_perm)
                        logging.info('[SV] Random Select permutation {} in {}'.format(len(r_perm), len(perm_list)))
                        # start sv part
                        evaluate_sv_info_dict = {i: [] for i in participate}
                        evaluated_sv_dict = {i: 0. for i in participate}
                        for item in r_perm:
                            logging.info('[SV-{}-FreeSpace:{}] Current Task {}, Global Round {}, Index {} Processing {}'.format(sv_eval_method, whether_free_space, task+1, global_round, idx, item))
                            evaluated_sv = []
                            weights_queue = []
                            node_queue = []
                            acc_history = [0.]
                            loss_history = [2.4]
                            for member in item:
                                node_queue.append(member)
                                weights_queue.append(usr_models[member].state_dict())
                                avg = average_weights(weights_queue)
                                global_model[0].load_state_dict(avg)
                                cur_loss, cur_acc = evaluate_model(global_model[0], usr_test_loaders[idx])
                                if sv_eval_method == 'acc':
                                    # logging.info('[EVAL] {}({}) on {} with accuracy influence {} - {} = {}%'.format(member, len(node_queue), idx, cur_acc * 100, acc_history[-1] * 100, (cur_acc - acc_history[-1]) * 100))
                                    evaluate_sv_info_dict[member].append(cur_acc - acc_history[-1])
                                    acc_history.append(cur_acc)
                                if sv_eval_method == 'loss':
                                     #logging.info('[EVAL] {}({}) on {} with loss influence {} - {} = {}'.format(member, len(node_queue), idx, loss_history[-1], cur_loss, loss_history[-1] - cur_loss))
                                    evaluate_sv_info_dict[member].append(loss_history[-1] - cur_loss)
                                    loss_history.append(cur_loss)
                        logging.info('[EVAL] Info gained {}'.format(evaluate_sv_info_dict))
                        # Avg sv
                        for i in participate:
                            # logging.critical('[EVAL] Write {} in {}'.format(np.mean(evaluate_sv_info_dict[i]), i))
                            evaluated_sv_dict[i] = np.mean(evaluate_sv_info_dict[i])
                        logging.info('[SV] Eval SV: {}'.format(evaluated_sv_dict))
                        # calculate weights based on sv

                        # update local downlaad dictionary
                        for d in backup:
                            logging.debug('[Prob] Update {}->{}: pro from {} to {} with change {}'.format(idx, d, prob_download_dict[idx][d], prob_download_dict[idx][d] + evaluated_sv_dict[d], evaluated_sv_dict[d]))
                            prob_download_dict[idx][d] += evaluated_sv_dict[d]

                        # calculate model differences

                        # TODO NEW EVAL METHOD
                        models_difference = {i: 0. for i in participate}
                        for request in participate:
                            params_diff = []
                            params_diff = torch.Tensor(params_diff)
                            if request != idx:
                                for param_cur, param_request in zip(usr_models[idx].parameters(), usr_models[request].parameters()):
                                    params_diff = torch.cat((params_diff, ((param_cur - param_request).view(-1))), 0)
                                    # logging.info('[SV] {} <-> {}'.format(torch.norm(param_cur), torch.norm(param_request)))
                                models_difference[request] = torch.norm(params_diff).detach().numpy()
                                # logging.info('[SV - EVAL] {} with models_difference {}'.format(request, models_difference[request]))
                            else:
                                models_difference[request] = 1.
                                # logging.info('[SV - EVAL] {} with models_difference {}'.format(request, models_difference[request]))
                        logging.info('[SV - EVAL] Model Difference {}'.format(models_difference))

                        # only use positive sv clients
                        # logging.info('[SV - EVAL] Before Exclude Negative Values&Itself SV: {}'.format(evaluated_sv_dict))

                        positive_idx = []
                        positive_sv = []
                        for i in participate:
                            if evaluated_sv_dict[i] > 0:
                                positive_idx.append(i)
                                # positive_sv.append(evaluated_sv[i])
                                # logging.info('[SV/MODELS] Request {} ==> {} / {} = {}'.format(i, evaluated_sv_dict[i], models_difference[i], evaluated_sv_dict[i] / models_difference[i]))
                                positive_sv.append(evaluated_sv_dict[i] / models_difference[i])
                                # logging.info('[SV/MODELS - EVAL] New Strategy Calculation {} / {} = {}'.format(evaluated_sv[i], models_difference[i], evaluated_sv[i] / models_difference[i]))

                        # norm with idx itself
                        positive_sv = [i / sum(positive_sv) for i in positive_sv]
                        logging.info('[SV] Positive Index {}'.format(positive_idx))
                        logging.info('[SV] Positive Weights {}'.format(positive_sv))

                        free_space = 1.0 - positive_sv[positive_idx.index(idx)]
                        logging.info('[SV] Owner {} leaves {} space for other model aggregation'.format(idx, free_space))

                        # self weights
                        self_weights = positive_sv[-1]
                        if not whether_delta:
                            logging.info('[SV] Self info {}'.format(self_weights))
                            for param in usr_models[idx].parameters():
                                param.data = param.data.clone() * self_weights

                        # norm with idx itself
                        positive_idx_noitself = positive_idx
                        positive_sv_noitself = positive_sv
                        positive_idx_noitself.pop()
                        positive_sv_noitself.pop()
                        positive_sv_noitself = [i / sum(positive_sv_noitself) for i in positive_sv_noitself]

                        logging.info('[SV] Positive norm weights without idx itself {}'.format(positive_sv_noitself))

                        # MAKER
                        selection_history[idx].append(positive_idx_noitself)
                        selection_weighst_history[idx].append(positive_sv_noitself)

                        # free_space to modify the weights or not
                        weights = {i: 0. for i in participate}
                        for i in participate:
                            if i in positive_idx and i != idx:
                                if whether_free_space:
                                    weights[i] = free_space * positive_sv[positive_idx.index(i)]  # add free_space
                                else:
                                    weights[i] = positive_sv[positive_idx.index(i)]               # not add free_space
                        logging.info('[SV] {} Allocated Weights {}'.format(idx, weights))

                        logging.critical('[Before SV Aggregation] {} Only with Accuracy {}% and Local loss {}'.format(idx, local_only_acc * 100, local_only_loss))
                        for request in participate:
                            for param, param_request in zip(usr_models[idx].parameters(), usr_models[request].parameters()):
                                if whether_delta:
                                    param.data += (param_request.data.clone() - param.data.clone()) * weights[request]
                                else:
                                    param.data += param.data.clone() * weights[request]
                        sv_loss, sv_acc = evaluate_model(usr_models[idx], usr_test_loaders[idx])
                        logging.critical('[After SV Aggregation] New Model Local Accuracy {}% and Local Loss {}'.format(sv_acc * 100, sv_loss))
                        loss_update_acc_dict[idx].append(sv_acc)

                        # MAKER
                        after_selection_acc_history[idx].append(sv_acc)
                        after_selection_loss_history[idx].append(sv_loss)

                    if active_local_loss:
                        '''
                        logging.info('[Fomo-Foresee] Models similarity')
                        motivation_s2 = {i: 0. for i in range(clients)}
                        for request in range(clients):
                            params_diff = []
                            tensor_diff = torch.Tensor(params_diff)
                            for param_cur, param_request in zip(usr_models[idx].parameters(), usr_models[request].parameters()):
                                # logging.info('[Fomo-EVAL] {} <-> {}'.format(torch.norm(param_cur), torch.norm(param_request)))
                                tensor_diff = torch.cat((tensor_diff, (param_cur - param_request).view(-1)), 0)
                            motivation_s2[request] = torch.norm(tensor_diff)
                            # logging.critical('[Fomo-compare] Model({}) <-> Model({}): ({})'.format(idx, request, torch.norm(tensor_diff)))
                        soretd_s2 = sorted(motivation_s2.items(), key = lambda kv:(kv[1], kv[0]))
                        rank = 1
                        for key, value in soretd_s2:
                            logging.critical('[Fomo-Foresee-Usr {}] Rank {}: {}, {}'.format(idx, rank, key, value))
                            rank += 1
                        '''

                        logging.info('[Fomo] Start Loss Computation')
                        local_only_acc = local_update_acc_dict[idx][-1]
                        local_only_loss = local_update_loss_dict[idx][-1]
                        logging.info('[Fomo] {} Only with Accuracy {}% & Loss {}'.format(idx, local_only_acc * 100, local_only_loss))

                        # FOR INNER RELATION
                        before_selection_acc_history[idx].append(local_only_acc)
                        before_selection_loss_history[idx].append(local_only_loss)

                        # compute probs:
                        downloaded_usrs = []
                        topk_idx = heapq.nlargest(exploit, range(len(prob_download_dict[idx])), prob_download_dict[idx].__getitem__)
                        downloaded_usrs += topk_idx
                        while len(downloaded_usrs) != download:
                            ch = random.choice(range(clients))
                            if ch not in downloaded_usrs and ch != idx:
                                downloaded_usrs.append(ch)
                        logging.critical('[Fomo] Download {} from the server'.format(downloaded_usrs))

                        download_history[idx].append(downloaded_usrs)

                        # calculate weights
                        weights = {i: 0. for i in downloaded_usrs}
                        models_diff_dict = {i: [] for i in downloaded_usrs}
                        for request in downloaded_usrs:
                            params_diff = []
                            tensor_diff = torch.Tensor(params_diff)
                            models_diff = []
                            request_loss, request_acc = evaluate_model(usr_models[request], usr_test_loaders[idx])
                            logging.info('[Fomo-EVAL] {} on {} dataset with accuracy {}% & Loss {}'.format(request, idx, request_acc * 100, local_only_loss - request_loss))
                            for param_cur, param_request in zip(usr_models[idx].parameters(), usr_models[request].parameters()):
                                # logging.info('[Fomo-EVAL] {} <-> {}'.format(torch.norm(param_cur), torch.norm(param_request)))
                                tensor_diff = torch.cat((tensor_diff, (param_cur - param_request).view(-1)), 0)
                                models_diff.append(param_cur - param_request)
                            w = (local_only_loss - request_loss) / torch.norm(tensor_diff)
                            prob_download_dict[idx][request] += w
                            weights[request] = w
                            models_diff_dict[request] = models_diff
                        # exclude negative value, only maintain the positive weight
                        for item in downloaded_usrs:
                            weights[item] = max(0, weights[item])
                        logging.info('[Fomo] weights {}'.format(weights))
                        # print the positive weight and its idx of download user, the weights are not norm
                        positive_idx = []
                        for request in downloaded_usrs:
                            if weights[request] > 0:
                                positive_idx.append(request)
                                logging.info('[Fomo] the client idx is {} and its positive weight is {}'.format(request, weights[request]))

                        selection_history[idx].append(positive_idx if positive_idx != [] else 'Empty')

                        # normalize weights
                        base = sum([i for i in weights.values()])
                        if base != 0:
                            selection_weights = []
                            for item in downloaded_usrs:
                                weights[item] = weights[item] / base
                            for o in positive_idx:
                                selection_weights.append(weights[o].item())
                            selection_weighst_history[idx].append(selection_weights)
                            logging.critical('[Fomo] weights {}'.format(weights))
                            # model aggregation to update local model
                            # local_update_model = usr_models[idx].state_dict()
                            for request in downloaded_usrs:
                                for param, param_request in zip(usr_models[idx].parameters(), usr_models[request].parameters(),):
                                    param.data += (param_request.data.clone() - param.data.clone()) * weights[request]
                        else:
                            selection_weighst_history[idx].append('Empty')
                            # usr_models[idx].load_state_dict(local_update_model)
                            # loss_update_models_dict[idx].append(local_update_model)
                        iclar_loss, iclr_acc = evaluate_model(usr_models[idx], usr_test_loaders[idx])
                        logging.critical('[Fomo - EVAL] Fomo {} Accuracy {}% Loss {}'.format(idx, iclr_acc * 100, iclar_loss))
                        loss_update_acc_dict[idx].append(iclr_acc)
                        after_selection_acc_history[idx].append(iclr_acc);
                        after_selection_loss_history[idx].append(iclar_loss)
        if active_local_sv or active_local_loss:
            record = [loss_update_acc_dict[i][-1] for i in range(clients)]
            logging.critical('Task {} Avg Accuracy {}'.format(task, record))
            multi_task_avg_accuracy_list[task] = np.mean(record)
        else:
            record = [local_update_acc_dict[i][-1] for i in range(clients)]
            logging.critical('Task {} Avg Accuracy {}'.format(task, record))
            multi_task_avg_accuracy_list[task] = np.mean(record)

        print('---------------------------')
        print(selection_history)
        print(selection_weighst_history)
        print('---------------------------')
        print(before_selection_acc_history)
        print(after_selection_acc_history)
        print('---------------------------')
        print(before_selection_loss_history)
        print(after_selection_loss_history)
        print('---------------------------')

        filter = 'sv' if active_local_sv else 'fomo'
        writing_name_lst = ['download', 'selection', 'weights', 'before_acc_history', 'after_acc_history', 'before_loss_history', 'after_loss_history']
        writing_lst = [download_history, selection_history, selection_weighst_history, before_selection_acc_history, after_selection_acc_history, before_selection_loss_history, after_selection_loss_history]
        for item in range(len(writing_name_lst)):
            with open('task{}_{}_{}.csv'.format(task, filter, writing_name_lst[item]), 'w') as f:
                for key, value in writing_lst[item].items():
                    if value != 'Empty':
                        f.write('{},'.format(key))
                        for da in value:
                            if writing_name_lst[item] == 'selection' \
                                    or writing_name_lst[item] == 'weights'\
                                    or writing_name_lst[item] == 'download':
                                if da != 'Empty':
                                    for element in da:
                                        f.write('{} '.format(element))
                                    f.write(',')
                                else:
                                    f.write('Empty,')
                            else:
                                f.write('{1},'.format(key, da))
                        f.write('\n')
                    else:
                        f.write('{0},{1}\n'.format(key, 'Empty'))
        '''
        if clients % 2 == 0:
            draw_clients = clients // 2
        else:
            draw_clients = clients // 2 + 1
        figs = [plt.figure('Fig {}'.format(i), figsize=(20, 10)) for i in range(draw_clients)]
        for i in range(draw_clients):
            fig = figs[i]
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[i * 2], color='red',
                     linestyle='--',
                     marker='x')
            if active_local_loss or active_local_sv:
                ax1.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[i * 2], color='blue',
                         linestyle='--',
                         marker='x')
            ax1.set_title('Client {}'.format(i * 2))
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy')

            if i == draw_clients - 1 and clients % 2 != 0:
                break

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), local_update_acc_dict[i * 2 + 1], color='red',
                     linestyle='--',
                     marker='x')
            if active_local_loss or active_local_sv:
                ax2.plot(np.arange(1, global_epoch + 1).astype(dtype=np.str), loss_update_acc_dict[i * 2 + 1], color='blue',
                         linestyle='--',
                         marker='x')
            ax2.set_title('Client {}'.format(i * 2 + 1))
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Accuracy')
        plt.show()
        '''

        if active_local_sv:
            with open('{}.csv'.format('./SV/T{}_sv_client{}_round{}_update_accuracy'.format(task, clients, global_epoch)), 'w') as f:
                [f.write('{0},{1}\n'.format(key, '{}'.format(value).strip('[]'))) for key, value in loss_update_acc_dict.items()]

            with open('{}.csv'.format('./SV/T{}_sv_client{}_round{}_local_update_accuracy'.format(task, clients, global_epoch)), 'w') as f:
                [f.write('{0},{1}\n'.format(key, '{}'.format(value).strip('[]'))) for key, value in local_update_acc_dict.items()]
        elif active_local_loss:
            with open('{}.csv'.format('./ICLR/T{}_iclr_client{}_round{}_update_accuracy'.format(task, clients, global_epoch)), 'w') as f:
                [f.write('{0},{1}\n'.format(key, '{}'.format(value).strip('[]'))) for key, value in loss_update_acc_dict.items()]

            with open('{}.csv'.format('./ICLR/T{}_iclr_client{}_round{}_local_update_accuracy'.format(task, clients, global_epoch)), 'w') as f:
                [f.write('{0},{1}\n'.format(key, '{}'.format(value).strip('[]'))) for key, value in local_update_acc_dict.items()]
        else:
            with open('{}.csv'.format('./PureLU/T{}_pure_local_client{}_round{}_update_accuracy'.format(task, clients, global_epoch)), 'w') as f:
                [f.write('{0},{1}\n'.format(key, '{}'.format(value).strip('[]'))) for key, value in local_update_acc_dict.items()]

    multitask_fig = plt.figure('Multi-task Figure', figsize=(10, 10))
    ax = multitask_fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, task_repeat_time + 1).astype(dtype=np.str), multi_task_avg_accuracy_list, color='red',
                     linestyle='--',
                     marker='x')
    ax.set_title('Repeat {} Task'.format(task_repeat_time))
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    plt.show()

    if active_local_sv:
        with open('{}_exper2.csv'.format("SV"), 'w') as f:
            [f.write('{0},{1}\n'.format(key, '{}'.format(multi_task_avg_accuracy_list).strip('[]'))) for key in range(task_repeat_time)]
    elif active_local_loss:
        with open('{}_exper2.csv'.format("Fomo"), 'w') as f:
            [f.write('{0},{1}\n'.format(key, '{}'.format(multi_task_avg_accuracy_list).strip('[]'))) for key in range(task_repeat_time)]
    else:
        with open('{}_exper2.csv'.format("Local"), 'w') as f:
            [f.write('{0},{1}\n'.format(key, '{}'.format(multi_task_avg_accuracy_list).strip('[]'))) for key in range(task_repeat_time)]