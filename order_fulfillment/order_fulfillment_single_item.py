import sys
import json
import itertools

ship_cost = [7.84, 8.35, 8.97, 10.13, 13.91, 15.76, 17.79, 17.79]
ship_cost = [round(cost/4, 2) for cost in ship_cost]
#  We scale these constants by 4 to be realistic.


def read_data(dataFile):
    file = open(dataFile, 'r')
    inst = json.load(file)
    for store in inst:
        ship_zone = store['s']
        store['s'] = ship_cost[ship_zone]
    file.close()
    return inst


def calculate_expected_cost(cost_data, store_indices, _penalty_cost):
    cost = cost_data[store_indices[0]]['c'] + (1 - cost_data[store_indices[0]]['p']) * cost_data[store_indices[0]]['s']
    product = cost_data[store_indices[0]]['p']
    for stage in range(len(store_indices) - 1):
        cost += product * (cost_data[store_indices[stage + 1]]['c'] + (1 - cost_data[store_indices[stage + 1]]['p']) *
                           cost_data[store_indices[stage + 1]]['s'])
        product *= cost_data[store_indices[stage + 1]]['p']
    cost += product * _penalty_cost
    return cost


# Sort by labor and shipping Cost, Stores not repeated
def solve_greedy(_data, n_stages, _penalty_cost):
    n_stages_adjusted = min(len(_data), n_stages)
    index_sorted = sorted(range(len(_data)), key=lambda k: (_data[k]['s'] + _data[k]['c']))
    # Store indices in ascending order of ratio
    val = calculate_expected_cost(_data, index_sorted[:n_stages_adjusted], _penalty_cost)
    #  print('Ship & Pick Obj :', val, 'Stores : ', index_sorted[:n_stages_adjusted])
    return val, index_sorted[:n_stages_adjusted]


# Enumerate in sorted list for OPT, Stores not repeated
# Can be used to test if DP is implemented correctly.
# Not used in draft
def solve_opt_enumerate(_data, n_stages, _penalty_cost):
    def ratio(store):
        num = _data[store]['c'] + (1 - _data[store]['p']) * float(_data[store]['s'])
        den = 1 - _data[store]['p']
        return num / den

    index_sorted = sorted(range(len(_data)), key=lambda k: ratio(k))
    n_stores = len(index_sorted)
    if n_stores < n_stages + 1:
        val = calculate_expected_cost(_data, index_sorted, _penalty_cost)
        order_opt = index_sorted
    else:
        tuples_indices = list(itertools.combinations(list(index_sorted), n_stages))
        cost = {s: calculate_expected_cost(_data, list(s), _penalty_cost) for s in tuples_indices}
        order_opt = min(cost, key=cost.get)
        val = min(cost.values())
    #  print('DP OPT :', val, 'Stores : ', list(order_opt))
    return val, order_opt


# Solve DP for OPT, Stores not repeated
def solve_opt_dp(_data, n_stages, _penalty_cost):
    def ratio(store):
        num = _data[store]['c'] + (1 - _data[store]['p']) * _data[store]['s']
        den = 1 - _data[store]['p']
        return num / den

    index_sorted = list(sorted(range(len(_data)), key=lambda k: ratio(k)))
    n_stores = len(index_sorted)

    if n_stores < n_stages + 1:
        val = calculate_expected_cost(_data, index_sorted, _penalty_cost)
        order_opt = index_sorted
    else:
        valueFunction = [[50 * _penalty_cost for _ in range(1 + n_stores)] for _ in range(1 + n_stages)]
        # One extra stage to handle the penalty
        # the "1+" in 1+len(indexSorted) to handle #stores = #stages
        for _store in range(1 + len(index_sorted)):
            valueFunction[n_stages][_store] = _penalty_cost

        selectedStoresLocalIndex = [[-1 for _ in range(1 + n_stores)] for _ in range(n_stages)]
        #  Writing the recursion
        for stage in reversed(range(n_stages)):
            # s is position
            for s in range(1 + len(index_sorted) - n_stages + stage):
                # Initializing with huge cost
                theCost = n_stores * _penalty_cost
                for ss in range(s, 1 + len(index_sorted) - n_stages + stage):
                    currentCost = _data[index_sorted[ss]]['c'] + (1 - _data[index_sorted[ss]]['p']) \
                                  * _data[index_sorted[ss]]['s'] + _data[index_sorted[ss]]['p'] \
                                  * valueFunction[stage + 1][ss + 1]
                    if currentCost < theCost:
                        theCost = currentCost
                        selectedStoresLocalIndex[stage][s] = ss
                valueFunction[stage][s] = theCost

        selectedStore = [-1 for _ in range(n_stages)]
        currentStoreIndex = 0
        for stage in range(n_stages):
            selectedStore[stage] = index_sorted[selectedStoresLocalIndex[stage][currentStoreIndex]]
            currentStoreIndex = selectedStoresLocalIndex[stage][currentStoreIndex] + 1

        val = valueFunction[0][0]
        order_opt = selectedStore

    return val, order_opt


if __name__ == "__main__":
    retailer = sys.argv[1]
    instance = sys.argv[2]

    filename = 'data/si/retailer_{0}/{0}.{1}.json'.format(retailer, instance)
    data = read_data(filename)

    nStores = len(data)
    nStages = 3
    penalty_cost = 25

    optDPCost, optDPAssignment = solve_opt_dp(data, nStages, penalty_cost)
    greedyCost, greedyAssignment = solve_greedy(data, nStages, penalty_cost)

    fileout = 'summary_stats/costs_single_item.txt'
    f = open(fileout, "a+")
    row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(retailer, instance, nStores, optDPCost, greedyCost,
                                                optDPAssignment, greedyAssignment)
    f.write(row)
    f.close()
