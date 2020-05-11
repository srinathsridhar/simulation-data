# This code contains all heuristics for multi-item orders, namely LSC, SPS and Greedy.
# It takes the data as input and returns the cost and store assignment as output.

import itertools
from functools import reduce
import operator
from pyomo.environ import *
from pyomo.opt import SolverFactory


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def power_set(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


# This data is taken directly from https://pe.usps.com/text/dmm300/Notice123.htm
#  Look for Commercial under Parcel Select - Ground, in multiple of 4lbs.
ship_cost = [[7.84, 8.35, 8.97, 10.13, 13.91, 15.76, 17.79, 17.79],
                 [8.35, 10.07, 11.39, 17.15, 22.22, 26.33, 30.59, 30.59],
                 [11.9, 14.29, 16.79, 23.98, 31.56, 38.1, 44.38, 44.38],
                 [14.29, 17.87, 20.68, 29.83, 39.7, 45.54, 52.78, 52.78],
                 [16.00, 20.04, 23.64, 34.2, 47.13, 54.63, 64.04, 64.04],
                 [18.49, 22.5, 27.06, 37.59, 49.11, 57.43, 67.63, 67.63],
                 [22.25, 26.47, 34.7, 45.81, 52.46, 63.18, 76.43, 76.43],
                 [24.57, 27.98, 37.67, 48.22, 56.37, 68.99, 83.52, 83.52],
                 [25.78, 31.09, 40.69, 52.09, 61.95, 74.32, 89.66, 89.66],
                 [27.84, 35.24, 44.44, 59.73, 73.19, 84.77, 100.41, 100.41],
                 [28.72, 36.26, 46.00, 63.56, 76.59, 88.45, 105.17, 105.17],
                 [29.93, 37.28, 47.54, 67.07, 80.59, 93.21, 109.65, 109.65],
                 [31.85, 38.29, 49.09, 69.16, 83.08, 99.01, 115.3, 115.3],
                 [33.74, 39.3, 50.63, 70.83, 85.11, 103.15, 119, 119],
                 [35.65, 40.31, 52.18, 72.2, 86.87, 105.19, 126.41, 126.41],
                 [37.63, 41.33, 54.53, 73.21, 88.95, 107.02, 133.4, 133.4]]


def shipping_cost(zone, numItems):
    if numItems == 0:
        return 0

    return round(ship_cost[numItems - 1][zone]/4, 2)
    # We scale these constants by 4 to be realistic.


# Given a sequence of stores for each item, this function calculates the exact expected cost (in exponential time)
def compute_exact_cost(_assignment, _data, _nStores, _nItems, _nStages, _penaltyCost):
    ub = 0
    probabilityOfTrying = [1 for _item in range(_nItems)]
    for stage in range(_nStages):
        # Shipping Cost
        for store in range(_nStores):
            itemsInStore = set()
            for _item in range(_nItems):
                if _assignment[stage][_item] == store:
                    itemsInStore.add(_item)
            for itemsToTry in power_set(itemsInStore):
                shipCost = 0
                for itemsToShip in power_set(itemsToTry):
                    itemsNotToShip = set(itemsToTry) - set(itemsToShip)
                    probShipping = 1
                    for it in itemsToShip:
                        probShipping *= 1 - _data[0][store]['p'][it]
                    for it in itemsNotToShip:
                        probShipping *= _data[0][store]['p'][it]
                    shipCost += probShipping * shipping_cost(_data[0][store]['s'], len(itemsToShip))
                probTryingInSTore = prod(probabilityOfTrying[_it] for _it in itemsToTry)
                itemsNotToTry = itemsInStore - set(itemsToTry)
                probTryingInSTore *= prod(1 - probabilityOfTrying[_it] for _it in itemsNotToTry)
                ub += probTryingInSTore * shipCost
        # Picking Cost & Updating Trying Probabilities
        for _item in range(_nItems):
            if _assignment[stage][_item] != -1:
                ub += probabilityOfTrying[_item] * _data[0][_assignment[stage][_item]]['c'][_item]
                probabilityOfTrying[_item] *= _data[0][_assignment[stage][_item]]['p'][_item]

    # Last Level
    for _item in range(_nItems):
        ub += probabilityOfTrying[_item] * _penaltyCost

    return ub


# Given a sequence of stores for each item, this function calculates the exact expected cost (in exponential time)
# This one is a bit faster than compute_exact_cost()
# used for computations in draft
def compute_exact_cost_2(_assignment, _data, _nStores, _nItems, _nStages, _penaltyCost):
    ub = 0
    probabilityOfTrying = [1 for _item in range(_nItems)]
    for stage in range(_nStages):
        # Shipping Cost
        for store in range(_nStores):
            itemsInStore = set()
            for _item in range(_nItems):
                if _assignment[stage][_item] == store:
                    itemsInStore.add(_item)

            for itemsToShip in power_set(itemsInStore):
                itemsNotToShip = itemsInStore - set(itemsToShip)
                probShipping = 1
                for it in itemsToShip:
                    probShipping *= probabilityOfTrying[it] * (1 - _data[0][store]['p'][it])
                for it in itemsNotToShip:
                    probShipping *= 1 - probabilityOfTrying[it] * (1 - _data[0][store]['p'][it])
                ub += probShipping * shipping_cost(_data[0][store]['s'], len(itemsToShip))

        # Picking Cost & Updating Trying Probabilities
        for _item in range(_nItems):
            if _assignment[stage][_item] != -1:
                ub += probabilityOfTrying[_item] * _data[0][_assignment[stage][_item]]['c'][_item]
                probabilityOfTrying[_item] *= _data[0][_assignment[stage][_item]]['p'][_item]

    # Last Level
    for _item in range(_nItems):
        ub += probabilityOfTrying[_item] * _penaltyCost

    return ub


# Given a sequence of stores for each item, this function calculates an upper bound to exact expected cost
# in polynomial time
# Not used in draft
def compute_upper_bound_polytime(_assignment, _data, _nStores, _nItems, _nStages, _penaltyCost):
    ub = 0
    probabilityOfTrying = [1 for _item in range(_nItems)]
    for stage in range(_nStages):
        for store in range(_nStores):
            shipProbability = 0
            for _itemInStore in range(_nItems):
                if _assignment[stage][_itemInStore] == store:
                    shipProbability += probabilityOfTrying[_itemInStore] * (1 - _data[0][store]['p'][_itemInStore])

            ub += shipping_cost(_data[0][store]['s'], shipProbability)

        for _itemI in range(_nItems):
            if _assignment[stage][_itemI] != -1:
                ub += probabilityOfTrying[_itemI] * _data[0][_assignment[stage][_itemI]]['c'][_itemI]

                #  Update the probability of trying at store at the next stage
                probabilityOfTrying[_itemI] *= _data[0][_assignment[stage][_itemI]]['p'][_itemI]

    #  Adding the last failure cost after last stage
    for __item in range(_nItems):
        ub += probabilityOfTrying[__item] * _penaltyCost

    return ub


# For a single item this function computes the opt fulfillment policy and cost under linear shipping cost
def solve_dp_single_item(_item, _data, _nStores, _nItems, _nStages, _penaltyCost):
    nItemsPerStore = [0 for _ in range(_nStores)]
    for [_, _store] in _data[1]:
        nItemsPerStore[_store] += 1
    perUnitShippingCost = [shipping_cost(_data[0][_store]['s'], nItemsPerStore[_store]) / nItemsPerStore[_store]
                           for _store in range(_nStores)]

    def ratio(_store):
        return (_data[0][_store]['c'][_item]
                + (1 - _data[0][_store]['p'][_item]) * perUnitShippingCost[_store]) / (1 - _data[0][_store]['p'][_item])

    # Eligible Stores for Item
    eligibleStores = set()
    for [__item, __store] in _data[1]:
        if __item == _item:
            eligibleStores.add(__store)

    # Sort the Stores into a Sorted List
    indexSorted = list(sorted(eligibleStores, key=lambda k: ratio(k)))

    if len(indexSorted) < _nStages + 1:
        # No need to solve the DP
        _cost = _data[0][indexSorted[0]]['c'][_item] + (1 - _data[0][indexSorted[0]]['p'][_item]) \
                * perUnitShippingCost[indexSorted[0]]
        product = _data[0][indexSorted[0]]['p'][_item]
        for stage in range(len(indexSorted) - 1):
            _cost += product * (
                        _data[0][indexSorted[stage + 1]]['c'][_item] + (1 - _data[0][indexSorted[stage + 1]]['p'][_item]) *
                        perUnitShippingCost[indexSorted[stage + 1]])
            product *= _data[0][indexSorted[stage + 1]]['p'][_item]
        _cost += product * _penaltyCost

        if len(indexSorted) == _nStages:
            return indexSorted, _cost
        else:
            emptyLevels = [-1 for _ in range(_nStages - len(indexSorted))]
            indexSorted.extend(emptyLevels)
        return indexSorted, _cost

    else:
        #  Solve the DP
        #  Initializing the value Function
        valueFunction = [[50 * _penaltyCost for _ in range(1 + len(indexSorted))] for _ in range(1 + _nStages)]
        # One extra stage to handle the penalty
        # the "1+" in 1+len(indexSorted) is to handle #stores = #stages
        for _store in range(1 + len(indexSorted)):
            valueFunction[_nStages][_store] = _penaltyCost

        selectedStoresLocalIndex = [[-1 for _ in range(1 + len(indexSorted))] for _ in range(_nStages)]
        #  Writing the recursion
        for stage in reversed(range(_nStages)):
            # s is position
            for s in range(1 + len(indexSorted) - _nStages + stage):
                # Initializing with huge cost
                theCost = _nStores * _penaltyCost
                for ss in range(s, 1 + len(indexSorted) - _nStages + stage):
                    #             if _item == 3:
                    #                print('ss', ss)
                    currentCost = _data[0][indexSorted[ss]]['c'][_item] + (1 - _data[0][indexSorted[ss]]['p'][_item]) \
                                  * perUnitShippingCost[indexSorted[ss]] + _data[0][indexSorted[ss]]['p'][_item] \
                                  * valueFunction[stage + 1][ss + 1]
                    if currentCost < theCost:
                        theCost = currentCost
                        selectedStoresLocalIndex[stage][s] = ss
                valueFunction[stage][s] = theCost

        selectedStore = [-1 for _ in range(_nStages)]
        currentStoreIndex = 0
        for stage in range(_nStages):
            selectedStore[stage] = indexSorted[selectedStoresLocalIndex[stage][currentStoreIndex]]
            currentStoreIndex = selectedStoresLocalIndex[stage][currentStoreIndex] + 1

        return selectedStore, valueFunction[0][0]


# A lower bound to optimal cost
def solve_lb_multi_item_no_repeat(_data, _nStores, _nItems, _nStages, _penaltyCost):
    lb = 0
    lbAssignment = {stage: [-1 for _ in range(_nItems)] for stage in range(_nStages)}
    for _item in range(_nItems):
        itemAssignment, itemCost = solve_dp_single_item(_item, _data, _nStores, _nItems, _nStages, _penaltyCost)
        lb += itemCost
        for stage in range(_nStages):
            lbAssignment[stage][_item] = itemAssignment[stage]
    return lb, lbAssignment


# Linear Underestimator Heuristic that breaks multi-item order into single-item orders
def solve_lu(_data, _nStores, _nItems, _nStages, _penaltyCost):
    lbAssignment = {stage: [-1 for _ in range(_nItems)] for stage in range(_nStages)}
    for _item in range(_nItems):
        itemAssignment, itemCost = solve_dp_single_item(_item, _data, _nStores, _nItems, _nStages, _penaltyCost)
        for stage in range(_nStages):
            lbAssignment[stage][_item] = itemAssignment[stage]

    return compute_exact_cost_2(lbAssignment, _data, _nStores, _nItems, _nStages, _penaltyCost), lbAssignment


# Greedy heuristic that chooses L stores with lowest labor and shipping costs
def greedy(_data, _nStores, _nItems, _nStages, _penaltyCost):
    storesAssignment = {stage: [-1 for _ in range(_nItems)] for stage in range(_nStages)}
    for _item in range(_nItems):

        # Eligible Stores for Item
        eligibleStores = set()
        for [__item, __store] in _data[1]:
            if __item == _item:
                eligibleStores.add(__store)

        # Sort stores for each item based on pick cost only
        index_sorted = sorted(eligibleStores,
                              key=lambda k: shipping_cost(_data[0][k]['s'], 1) + _data[0][k]['c'][_item])

        _adjLevels = min(len(index_sorted), _nStages)

        for stage in range(_adjLevels):
            storesAssignment[stage][_item] = index_sorted[stage]
    return compute_exact_cost_2(storesAssignment, _data, _nStores, _nItems, _nStages, _penaltyCost), storesAssignment


# This heuristic allows k stores to be tried at each stage.
def solve_k_sps(k, _data, _nStores, _nItems, _nStages, _penaltyCost):
    k = min(k, _nItems)
    nItemsPerStore = [0 for _ in range(_nStores)]
    for [_, _store] in _data[1]:
        nItemsPerStore[_store] += 1
    perUnitShippingCost = [shipping_cost(_data[0][_store]['s'], nItemsPerStore[_store]) / nItemsPerStore[_store]
                           for _store in range(_nStores)]

    ratio = [{} for _ in range(_nItems)]
    for _item in range(_nItems):
        for _store in range(_nStores):
            if [_item, _store] in _data[1]:
                ratio[_item][_store] = (_data[0][_store]['c'][_item] + (1 - _data[0][_store]['p'][_item])
                                        * perUnitShippingCost[_store]) / (1 - _data[0][_store]['p'][_item])

    # Now we need to have an assignment policy based on 'k' stores per stage

    kAssignment = {stage: [-1 for _ in range(_nItems)] for stage in range(_nStages)}

    true_k = k
    stageAssigned = [False for _ in range(_nStages)]

    for stage in range(_nStages):
        while not stageAssigned[stage]:
            allEligibleStores = set()
            itemsHavingStores = set(range(_nItems))
            for _item in range(_nItems):
                allEligibleStores |= set(ratio[_item].keys())
                if len(ratio[_item]) == 0:
                    itemsHavingStores.remove(_item)

            true_k = min(true_k, len(itemsHavingStores), len(allEligibleStores))

            if not len(itemsHavingStores):
                stageAssigned[stage] = True
                break

            def varLinkConstraint_rule(_model, _item, _store):
                return _model.x[_item, _store] <= _model.y[_store]

            def oneStorePerItem_rule(_model, _item):
                return sum(_model.x[_item, store] for store in ratio[_item].keys()) == 1

            def numStoresConstraint_rule(_model):
                return sum(_model.y[store] for store in model.stores) == true_k

            def obj_rule(_model):
                return sum(_model.ratio[i, j] * _model.x[i, j] for [i, j] in _model.itemStorePairs)

            model = ConcreteModel()
            model.products = Set(initialize=list(itemsHavingStores))
            model.stores = Set(initialize=list(allEligibleStores))
            model.itemStorePairs = Set(initialize=list((i, j) for i in model.products for j in ratio[i].keys()))
            model.ratio = Param(model.itemStorePairs,
                                initialize={(i, j): ratio[i][j] for [i, j] in model.itemStorePairs})
            model.x = Var(model.itemStorePairs, within=Binary)
            model.y = Var(model.stores, within=Binary)

            model.obj = Objective(rule=obj_rule, sense=minimize)
            model.varLinkConstraint = Constraint(model.itemStorePairs, rule=varLinkConstraint_rule)
            model.oneStorePerItem = Constraint(model.products, rule=oneStorePerItem_rule)
            model.numStoresConstraint = Constraint(rule=numStoresConstraint_rule)
            #  model.pprint()
            opt = SolverFactory('cplex')

            results = opt.solve(model)

            if results.solver.status == SolverStatus.ok and \
                    results.solver.termination_condition == TerminationCondition.optimal:
                # Do something when the solution in optimal and feasible
                stageAssigned[stage] = True
                for [_item, _store] in model.itemStorePairs:
                    if model.x[_item, _store].value > 0.5:
                        kAssignment[stage][_item] = _store
                        del ratio[_item][_store]

                usedStores = []
                for _store in model.stores:
                    if model.y[_store].value > 0.5:
                        usedStores.append(_store)

            elif results.solver.termination_condition == TerminationCondition.infeasible:
                # Do something when model in infeasible
                true_k += 1

                if true_k > len(itemsHavingStores):
                    print('Warning Something Wrong!')
                    input('pause')
            else:
                # Something else is wrong
                print('Solver Status: ', results.solver.status)
                input('pause')
    # print('kAssignment', kAssignment)
    #  input('pause')
    return compute_exact_cost_2(kAssignment, _data, _nStores, _nItems, _nStages, _penaltyCost), kAssignment


# OPT using enumeration, too slow
# Not used at all in draft
def enumerate_opt(_data, _nStores, _nItems, _nStages, _penaltyCost):
    eligibleStores = [set() for _ in range(_nItems)]
    for _item in range(_nItems):
        for [__item, __store] in _data[1]:
            if __item == _item:
                eligibleStores[_item].add(__store)

    print('Eligible Stores:', eligibleStores)
    commonStores = {(a, b): eligibleStores[a] & eligibleStores[b]
                    for (a, b) in list(itertools.combinations(range(_nItems), 2))}

    c, pol = solve_lu(_data, _nStores, _nItems, _nStages, _penaltyCost)
    if not any(commonStores.values()):
        print('No Common Stores')
        return c, pol
    else:
        print('Common Stores: ', commonStores)

    combEligibleStores = [[] for _ in range(_nItems)]
    for _item in range(_nItems):
        if len(eligibleStores[_item]) == 0:
            combEligibleStores[_item] = [tuple([-1 for _ in range(_nStages)])]

        elif len(eligibleStores[_item]) < _nStages:
            combEligibleStores[_item] = list(itertools.permutations(eligibleStores[_item], len(eligibleStores[_item])))

            for _ in range(_nStages - len(eligibleStores[_item])):
                for store_permutations in combEligibleStores[_item]:
                    #             print(type(store_permutations))
                    store_permutations += (1,)
        else:
            combEligibleStores[_item] = list(itertools.permutations(eligibleStores[_item], _nStages))
    #      print('Eligible Items ', eligibleStores)
    #  print('combEligible Items : ', combEligibleStores)

    optCost = c
    optPolicy = tuple([tuple([pol[stage][_item] for stage in range(_nStages)]) for _item in range(_nItems)])
    print('Policy:', optPolicy)
    count = 0
    print('Total # Policies: ', len(list(itertools.product(*combEligibleStores))))
    for policy in itertools.product(*combEligibleStores):
        #   print('Policy', policy)
        count += 1
        print(count, end=' ')
        checkPolicy = False
        for stage in range(_nStages):
            for (i1, i2) in itertools.combinations(range(_nItems), 2):
                if policy[i1][stage] in commonStores[(i1, i2)] and policy[i2][stage] in commonStores[(i1, i2)]:
                    checkPolicy = True
                    break
        if checkPolicy:
            storesAssignment = [[policy[_item][_stage] for _item in range(_nItems)] for _stage in range(_nStages)]
            costOfPolicy = compute_exact_cost_2(storesAssignment, _data, _nStores, _nItems, _nStages, _penaltyCost)
            #     print(costOfPolicy)
            if costOfPolicy < optCost:
                optCost = costOfPolicy
                optPolicy = policy
    print('OptPolicy: ', optPolicy, optCost)
    optAssignment = {stage: [optPolicy[_item][stage] for _item in range(_nItems)] for stage in range(_nStages)}
    return optCost, optAssignment


def solve_multi_item(algorithm, data):
    nStores = len(data[0])
    nItems = len(data[0][0]['p'])
    nStages = 3
    penaltyCost = 25

    if algorithm == 'LB':
        cost, assignment = solve_lb_multi_item_no_repeat(data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == 'LSC':
        cost, assignment = solve_lu(data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == 'Greedy':
        cost, assignment = greedy(data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == '1-SPS':
        cost, assignment = solve_k_sps(1, data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == '2-SPS':
        cost, assignment = solve_k_sps(2, data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == '4-SPS':
        cost, assignment = solve_k_sps(4, data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == '6-SPS':
        cost, assignment = solve_k_sps(6, data, nStores, nItems, nStages, penaltyCost)

    elif algorithm == 'OPT':
        cost, assignment = enumerate_opt(data, nStores, nItems, nStages, penaltyCost)

    else:
        cost, assignment = -1, -1
        print('No Algorithm')
        input()

    return cost, assignment
