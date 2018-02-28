import json
import random
import numpy as np
import scipy.stats as st
import itertools
from functools import reduce
import sys
from data_generator import *

data_to_sample = read_tsv(sys.argv[2])
tf_model = sys.argv[1]
sample_obj = Sample(sys.argv[3])


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

def read_data(i):
    '''
    #f = open('Data_MI/Data(' + str(st) + ')(' + str(ty) + ')(' + str(instance) + ').json', 'r')
    f = open('Data_MI/0', 'r')
    data = json.load(f)
    f.close()
    '''
    data = multi_item_sampling(data_to_sample, sys.argv[4], sys.argv[5], sample_obj)
    #data = multi_item_sampling(model, data_to_sample)
    with open('Data_MI/{}_{}_noTF'.format(i, tf_model), 'w') as f:
        f.write(json.dumps(data))
    #data = [[{'p': [0.324], 's': 6.75, 'c': [1.85]}, {'p': [0.318], 's': 6.41, 'c': [1.45]}], [[0, 0], [0, 1]]] 
    #data = [[{'p': [0.1, 0.1], 's': 7.27, 'c': [12.0, 9.2]}], [[0, 0], [1, 0]]]

    print(data)
    print('Iter: ', i)
    return data


# Solving the LP relaxation for No Repeat nad then Iterative rounding to get UB
def solve_lp_noRepeat(data, I, L, BigC):
    def con_rule_1(model_1, i):
        return sum(model_1.x[i, j, 0] for j in model.J) == 1

    def con_rule2(model_2, i, j, l):
        return (1 - model_2.p[i, j]) * model_2.x[i, j, l] <= model_2.y[j, l]

    def con_rule3(model_3, i, l):  # Flow conservation constraint
        return sum(model_3.p[i, j] * model_3.x[i, j, l] for j in model_3.J) == sum(
            model_3.x[i, j, l + 1] for j in model_3.J)

    def con_rule4(model_4, i, j, l):
        return model_4.x[i, j, l] <= model_4.z[i, j, l]

    def con_rule5(model_5, i, j):
        return sum(model_5.z[i, j, l] for l in model_5.L) <= L

    def obj_rule(model):
        return sum(
            model.Cost[i, j] * model.x[i, j, l]  for i in model.I for j in model.J for l in
            model.L_con) + \
               sum(model.BigCost[i, j] * model.x[i, j, L - 1] for i in model.I for j in
                   model.J) + \
               sum(model.s[j] * model.y[j, l] for j in model.J for l in model.L)

    link = data[1]
    data = data[0]
    model = ConcreteModel()
    model.I = RangeSet(0, I - 1)
    model.L = RangeSet(0, L - 1)
    model.L_con = RangeSet(0, L - 2)
    model.J = RangeSet(0, len(data) - 1)
    # model.J = RangeSet(0, 999)
    model.y = Var(model.J * model.L, domain=NonNegativeReals, bounds=(0, 1))
    # model.pprint()
    model.s = Param(model.J, initialize={j: data[j]['s'] for j in model.J})
    model.p = Param(model.I, model.J, initialize={(i, j): data[j]['p'][i] for i, j in model.I * model.J})
    model.x = Var(model.I * model.J * model.L, domain=NonNegativeReals, bounds=(0, 1))
    model.z = Var(model.I * model.J * model.L, domain=Integers, bounds=(0, 1))

    model.c = Param(model.I, model.J, initialize={(i, j): data[j]['c'][i] for i in model.I for j in model.J})
    model.Cost = Param(model.I, model.J,
                       initialize={(i, j):  data[j]['c'][i] for i in model.I for j
                                   in model.J})

    model.BigCost = Param(model.I, model.J, initialize={
        (i, j): data[j]['c'][i] + data[j]['p'][i] * BigC for i in model.I for j in
        model.J})
    model.obj = Objective(rule=obj_rule, sense=minimize)
    model.con_1 = Constraint(model.I, rule=con_rule_1)
    model.con_2 = Constraint(model.I * model.J * model.L, rule=con_rule2)
    model.con_3 = Constraint(model.I * model.L_con, rule=con_rule3)
    model.con_4 = Constraint(model.I*model.J*model.L, rule=con_rule4)
    model.con_5 = Constraint(model.I * model.J, rule=con_rule5)

    opt = SolverFactory('cplex')
    time_lp_start = time.clock()
    opt.solve(model)
    time_lp_end = time.clock()
    time_lp = time_lp_end - time_lp_start
    print('LB : ', model.obj())
    y_hat = {}

    for j, l in model.J * model.L:
        prod = 1
        for i in model.I:
            if [i, j] in link:
                if model.x[i, j, l].value > 0:
                    #print('x_values : ', model.x[i, j, l].value)
                    #print('z_values : ', model.z[i, j, l].value)
                    prod *= model.p[i, j] * model.x[i, j, l].value
        y_hat[j, l] = 1 - prod

    UB = 0
    for j, l in model.J * model.L:
        UB += model.s[j] * y_hat[j, l]
        for i in model.I:
            if [i, j] in link:
                UB += model.x[i, j, l].value*data[j]['c'][i]
                if l == L - 1:
                    UB += data[j]['p'][i] * BigC * model.x[i, j, l].value
    print('UB : ', UB)
    return model.obj(), UB



def set_cover(data, I_new, L, BigC, n_runs):
    link = data[1]
    data = data[0]
    J = range(len(data))
    I = range(I_new)
    #print "I", "J"
    #print I, J
    cost_vector = []
    for _ in range(n_runs):
        cost = 0
        items_shipped = []
        stores_tried_previous_level = [[] for _ in I]
        for level in range(L):
            n_items_in_store = [[] for _ in J] # list of items in each store that are not shipped or tried at that store in previous levels
            store_expected_cost = [1E6 for _ in J]
            # print store_expected_cost
            for store in J:
                for item in I:
                    if [item, store] in link:
                        if item not in items_shipped:
                            #print store
                            #print stores_tried_previous_level
                            #if item not in stores_tried_previous_level[store]:
                            if store not in stores_tried_previous_level[item]:
                                n_items_in_store[store].append(item)
                                # print store, store_expected_cost, data, item
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
                                
                                min_value = cost_data[store]['c'][item]


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
                                min_value = cost_data[store]['s']

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
                        if 100 * cost_data[store]['c'][item] + cost_data[store]['s'] < min_value:
                            if store not in stores_used[item]:
                                selected_store = store
                                min_value = 100 * cost_data[store]['c'][item] + cost_data[store]['s']

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

    f = open('Results_MI/Individual_Type/Bounds_All_Iter1_{}_noTF.txt'.format(tf_model), 'w')
    for iter in range(100):  # iter is the instance of the type we wish to run
        lb_vector = 0
        ub_vector = 0
        cost_adap_vector = 0
        obj_pick_vector = 0
        obj_ship_vector = 0
        obj_pick_ship_vector = 0
        f_charts = open('Results_MI/Bounds_All.txt', 'w')
        total_runs = 1
        print('iter : ', iter)
        data = read_data(iter)
        n_stores = len(data[0])
        #print(data[1])
        n_items = len(data[0][0]['p'])
        #print(data[0]['p'])
        n_samples = 1000
        n_levels = 3
        n_levels = estimate_n_levels(data[1], n_items, n_stores)
        big_penalty = 25
        f.write('Run {}:'.format(iter)+'\t')
        lb, ub = 0, 0 #solve_lp_noRepeat(data, n_items, n_levels, big_penalty)
        lb_vector += lb
        ub_vector += ub
        f.write(str(lb) + '\t' + str(ub) + '\t')
        #lb is the lower bound obtained by solving the IP


        cost_adap, low_conf, high_conf = set_cover(data, n_items, n_levels, big_penalty, n_samples)
        #f.write(str(cost_adap) + '\t')
        cost_adap_vector += cost_adap
        f.write(str(cost_adap) + '\t' + str(low_conf) + '\t' + str(high_conf)+'\t')

        #for theta in [20, 50, 80]:
         #   print('theta : ', theta)
         #   cost_adap_param, low_con_param, high_con_param = parameterized_set_cover(data, n_items, n_levels, big_penalty, theta)
          #  f.write(str(cost_adap_param) + '\t' + str(low_con_param) + '\t' + str(high_con_param)+'\t')

        obj_pick, low_con_pick, high_con_pick = solve_pick(data, n_items, n_levels, big_penalty, n_samples)
        #f.write(str(obj_pick) +'\t')
        f.write(str(obj_pick) + '\t' + str(low_con_pick) + '\t' + str(high_con_pick)+'\t')
        obj_pick_vector += obj_pick

        obj_ship, low_con_ship, high_con_ship = solve_ship(data, n_items, n_levels, big_penalty, n_samples)
        #f.write(str(obj_ship) + '\t')
        f.write(str(obj_ship) + '\t' + str(low_con_ship) + '\t' + str(high_con_ship) + '\t')
        obj_ship_vector += obj_ship

        obj_pick_ship, low_con_pick_ship, high_con_pick_ship = solve_pick_ship(data, n_items, n_levels, big_penalty, n_samples)
        f.write(str(obj_pick_ship) + '\t' + str(low_con_pick_ship) + '\t' + str(high_con_pick_ship) + '\t')
        #f.write(str(obj_pick_ship)+ '\t')
        obj_pick_ship_vector += obj_pick_ship

        f.write('\n')
        lb_vector /= total_runs
        ub_vector /= total_runs
        cost_adap_vector /= total_runs
        obj_pick_vector /= total_runs
        obj_ship_vector /= total_runs
        obj_pick_ship_vector /= total_runs
        f_charts.write(str(cost_adap_vector) + '\t' + str(
            obj_pick_vector) + '\t' + str(obj_ship_vector) + '\t' + str(obj_pick_ship_vector) + '\n')


    f.close()
if __name__ == "__main__":
    main()
