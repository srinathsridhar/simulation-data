import json
import random
import numpy as np
import scipy.stats as st
import itertools


def estimate_n_levels(link, n_it, n_st):
    L_max = 1000
    for i in range(n_it):
        L_item = 0
        for j in range(n_st):
            if [i, j] in link:
                L_item += 1
        #print('L_item :', L_item)
        L_max = min(L_item, L_max)
    L_max = min(3, L_max)
   # print(L_max)
    print('n_levels : ', L_max)
    return int(L_max)


def read_data(st, instance, ty) -> list:
    f = open('Data_MI_dcsg/Data(' + str(st) + ')(' + str(ty) + ')(' + str(instance) + ').json', 'r')
    data = json.load(f)
    f.close()
    return data


# Our Simulated Algorithm
def set_cover(data, I, L, BigC, n_runs):
    link = data[1]
    data = data[0]
    J = range(len(data))
    I = range(I)
    cost_vector = []
    for _ in range(n_runs):
        cost = 0
        items_shipped = []
        stores_tried_previous_level = [[] for _ in I]
        for level in range(L):
            n_items_in_store = [[] for _ in J] # list of items in each store that are not shipped or tried at that store in previous levels
            store_expected_cost = [1E6 for _ in J]
            for store in J:
                for item in I:
                    if [item, store] in link:
                        if item not in items_shipped:
                            if store not in stores_tried_previous_level[item]:
                                n_items_in_store[store].append(item)
                                store_expected_cost[store] += (data[store]['c'][item] + (1 - data[store]['p'][item]) * data[store]['s'])/(1-data[store]['p'][item] )

            # print('level', level)
            sort_cost_cardinality = sorted(range(len(n_items_in_store)), key=lambda k: store_expected_cost[k] - 1000 * len(n_items_in_store[k]))
            #  Sort by expected cost - the maximum number of items a store can supply

            # print('sort_max_cardinality : ', sort_max_cardinality)
            # print([len(items_in_store[sort_max_cardinality[j]]) for j in J])
            assign_item = [-1] * len(I)  # Store to which each item is assigned
            stores_assigned = set()         # set of stores assigned (just for recoed keeping)
            for store in sort_cost_cardinality:    #  try each store one by one
                for item in n_items_in_store[store]:
                    if store not in stores_tried_previous_level[item]:
                        if assign_item[item] == -1:         # if item should be tried at the store
                            assign_item[item] = store       # assign item to the store
                            stores_assigned.add(store)


            #  print('Assignment : ', assign_item)

            # We begin trials at each store for the items assigned to that store
            for store in stores_assigned:
                shipped = False
                for item in n_items_in_store[store]:
                    if item not in items_shipped:
                        if assign_item[item] == store:  # if item has been assigned to the store
                            cost += data[store]['c'][item]
                            stores_tried_previous_level[item].append(
                                store)  # the store has been tried for item, hence cannot be tried at next level
                            if random.random() > data[store]['p'][item]:    # if edge success
                                items_shipped.append(item)
                                # print('Item ', item, 'shipped.')
                                shipped = True

                if shipped:
                    cost += data[store]['s']
        cost += BigC * (len(I) - len(items_shipped))
        # print('Cost : ', cost)
        cost_vector.append(cost)

    lc, uc = st.t.interval(0.95, len(cost_vector) - 1, loc=np.mean(cost_vector), scale=st.sem(cost_vector))
    print('Our Cost : ', sum(cost_vector) / len(cost_vector), '(', lc, uc, ')')
    return sum(cost_vector) / len(cost_vector), lc, uc


# Simulation based on sorting by Picking Cost
def solve_pick(cost_data, n_item, L, BigC, n_runs):
    link = cost_data[1]
    cost_data = cost_data[0]
    n_stores = range(len(cost_data))
    n_item = range(n_item)
    cost_vector = []
    for _ in range(n_runs):
        items_shipped = []
        stores_shipped = set()
        stores_used = [set() for _ in n_item]
        cost = 0
        for item in n_item:
            for level in range(L):
                min_value = 100000
                for store in n_stores:
                    if [item, store] in link:
                        if cost_data[store]['c'][item] < min_value:
                            if store not in stores_used[item]:
                                selected_store = store


                stores_used[item].add(selected_store)
                cost += cost_data[selected_store]['c'][item]
                if random.random() > cost_data[selected_store]['p'][item]:
                    items_shipped.append(item)
                    stores_shipped.add(selected_store)
                    break

        for stores in stores_shipped:
            cost += cost_data[stores]['s']

        cost += BigC * (len(n_item) - len(stores_shipped))
        cost_vector.append(cost)

    lc, uc = st.t.interval(0.95, len(cost_vector) - 1, loc=np.mean(cost_vector), scale=st.sem(cost_vector))
    print('Pick Cost : ', sum(cost_vector) / len(cost_vector), '(', lc, uc, ')')
    return sum(cost_vector) / len(cost_vector), lc, uc


# Simulation based on sorting by Shipping Cost
def solve_ship(cost_data, n_items, n_levels, BigC, n_runs):
    link = cost_data[1]
    cost_data = cost_data[0]
    n_stores = range(len(cost_data))
    n_items = range(n_items)
    cost_vector = []
    for _ in range(n_runs):
        items_shipped = []
        stores_shipped = set()
        stores_used = [set() for _ in n_items]
        cost = 0
        for item in n_items:
            for level in range(n_levels):
                min_value = 100000
                for store in n_stores:
                    if [item, store] in link:
                        if cost_data[store]['s'] < min_value:
                            if store not in stores_used[item]:
                                selected_store = store

                stores_used[item].add(selected_store)
                cost += cost_data[selected_store]['c'][item]
                if random.random() > cost_data[selected_store]['p'][item]:
                    items_shipped.append(item)
                    stores_shipped.add(selected_store)
                    break

        for stores in stores_shipped:
            cost += cost_data[stores]['s']

        cost += BigC * (len(n_items) - len(stores_shipped))
        cost_vector.append(cost)

    lc, uc = st.t.interval(0.95, len(cost_vector) - 1, loc=np.mean(cost_vector), scale=st.sem(cost_vector))
    print('Ship Cost : ', sum(cost_vector) / len(cost_vector), '(', lc, uc, ')')
    return sum(cost_vector) / len(cost_vector), lc, uc


# Simulation based on sorting by Picking & Shipping Cost
def solve_pick_ship(cost_data, n_items, n_levels, BigC, n_runs):
    link = cost_data[1]
    cost_data = cost_data[0]
    J = range(len(cost_data))
    n_items = range(n_items)
    cost_vector = []
    for _ in range(n_runs):
        items_shipped = []
        stores_shipped = set()
        stores_used = [set() for _ in n_items]
        cost = 0
        for item in n_items:
            for level in range(n_levels):
                min_value = 100000
                for store in J:
                    if [item, store] in link:
                        if cost_data[store]['c'][item]+cost_data[store]['s'] < min_value:
                            if store not in stores_used[item]:
                                selected_store = store

                stores_used[item].add(selected_store)
                cost += cost_data[selected_store]['c'][item]
                if random.random() > cost_data[selected_store]['p'][item]:
                    items_shipped.append(item)
                    stores_shipped.add(selected_store)
                    break

        for stores in stores_shipped:
            cost += cost_data[stores]['s']

        cost += BigC * (len(n_items) - len(stores_shipped))
        cost_vector.append(cost)

    lc, uc = st.t.interval(0.95, len(cost_vector) - 1, loc=np.mean(cost_vector), scale=st.sem(cost_vector))
    print('Pick and Ship Cost : ', sum(cost_vector) / len(cost_vector), '(', lc, uc, ')')
    return sum(cost_vector) / len(cost_vector), lc, uc


def main():
    types = []
    for string in map(''.join, itertools.product('LH', repeat=5)):
        #print(string)
        types.append(string)

    for t in types:
        cost_adap_vector = 0
        obj_pick_vector = 0
        obj_ship_vector = 0
        obj_pick_ship_vector = 0
        f_charts = open('Results_MI/Bounds_All.txt', 'a+')
        total_runs = 10
        for run in range(total_runs):  # iter is the instance of the type we wish to run
            print('iter : ', run)
            print('type : ', t)
            n_stores = 100
            data = read_data(n_stores, run, t)
            #  data = [[{'p': [0.324], 's': 6.75, 'c': [1.85]}, {'p': [0.318], 's': 6.41, 'c': [1.45]}], [[0, 0], [0, 1]]]
            n_stores = len(data[0])
            n_items = len(data[0][0]['p'])
            n_samples = 10000
            #  n_levels = 4
            n_levels = estimate_n_levels(data[1], n_items, n_stores)
            big_penalty = 100
            f = open('Results_MI/Individual_Type/Bounds_All_Iter1.txt', 'a+')
            f.write(t+'\t')

            cost_adap, low_conf, high_conf = set_cover(data, n_items, n_levels, big_penalty, n_samples)
            cost_adap_vector += cost_adap
            f.write(str(cost_adap) + '\t' + str(low_conf) + '\t' + str(high_conf)+'\t')

            obj_pick, low_con_pick, high_con_pick = solve_pick(data, n_items, n_levels, big_penalty, n_samples)
            f.write(str(obj_pick) + '\t' + str(low_con_pick) + '\t' + str(high_con_pick)+'\t')
            obj_pick_vector += obj_pick

            obj_ship, low_con_ship, high_con_ship = solve_ship(data, n_items, n_levels, big_penalty, n_samples)
            f.write(str(obj_ship) + '\t' + str(low_con_ship) + '\t' + str(high_con_ship) + '\t')
            obj_ship_vector += obj_ship

            obj_pick_ship, low_con_pick_ship, high_con_pick_ship = solve_pick_ship(data, n_items, n_levels, big_penalty, n_samples)
            f.write(str(obj_pick_ship) + '\t' + str(low_con_pick_ship) + '\t' + str(high_con_pick_ship) + '\t')
            obj_pick_ship_vector += obj_pick_ship

            f.write('\n')
            f.close()

        cost_adap_vector /= total_runs
        obj_pick_vector /= total_runs
        obj_ship_vector /= total_runs
        obj_pick_ship_vector /= total_runs
        f_charts.write(str(cost_adap_vector) + '\t' + str(
            obj_pick_vector) + '\t' + str(obj_ship_vector) + '\t' + str(obj_pick_ship_vector) + '\n')


if __name__ == "__main__":
    main()
