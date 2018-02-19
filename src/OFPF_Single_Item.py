import json
import itertools


def calculate_expected_cost(cost_data, store_indices, big_penalty):
    cost = cost_data[store_indices[0]]['c'] + (1 - cost_data[store_indices[0]]['p']) * cost_data[store_indices[0]]['s']
    product = cost_data[store_indices[0]]['p']
    for l in range(len(store_indices)-1):
        cost += product * (cost_data[store_indices[l + 1]]['c'] + (1 - cost_data[store_indices[l + 1]]['p']) * cost_data[store_indices[l + 1]]['s'])
        product *= cost_data[store_indices[l + 1]]['p']
    cost += product * big_penalty
    return cost


def read_data(n_items, instance) -> list:
    file = open('Data_SI/Data(' + instance + ')(' + str(n_items) + ').json', 'r')
    cost_data = json.load(file)
    file.close()
    return cost_data


# Sort by pick cost, stores cannot be repeated
def solve_picking_norepeat(data, n_levels, big_penalty):
    index_sorted = sorted(range(len(data)), key=lambda k: (data[k]['c'], data[k]['s']))  # Store indices in ascending order of 'c'
    n_levels_adjusted = min(len(data), n_levels)
    val = calculate_expected_cost(data, index_sorted[:n_levels_adjusted], big_penalty)
    print('Picking Obj :', val, 'Stores : ', index_sorted[:n_levels_adjusted])
    return val


# Sort by Shipping Cost, stores cannot be repeated
def solve_shipping_norepeat(data, n_levels, big_penalty):
    index_sorted = sorted(range(len(data)), key=lambda k: (data[k]['s'], data[k]['c']))  # Store indices in ascending order of 's'
    n_levels_adjusted = min(len(data), n_levels)
    val = calculate_expected_cost(data, index_sorted[:n_levels_adjusted], big_penalty)
    print('Shipping Obj :', val, 'Stores : ', index_sorted[:n_levels_adjusted])
    return val


# Sort by Pick and Shipping Cost, Stores not repeated
def solve_pick_and_ship_norepeat(data, n_levels, big_penalty):
    n_levels_adjusted = min(len(data), n_levels)
    index_sorted = sorted(range(len(data)), key=lambda k: (data[k]['s'] + data[k]['c']))  # Store indices in ascending order of ratio
    val = calculate_expected_cost(data, index_sorted[:n_levels_adjusted], big_penalty)
    print('Ship & Pick Obj :', val, 'Stores : ', index_sorted[:n_levels_adjusted])
    return val


# Solve DP for OPT, Stores not repeated
def solve_opt_dp_norepeat(data, n_levels, big_penalty):
    def ratio(store):
        return (data[store]['c'] + (1 - data[store]['p']) * data[store]['s']) / (1 - data[store]['p'])

    index_sorted = sorted(range(len(data)), key=lambda k: ratio(k))
    n_levels_adjusted = min(len(data), n_levels)
    tuples_indices = list(itertools.combinations(list(index_sorted), n_levels_adjusted))
    cost = {s: calculate_expected_cost(data, list(s), big_penalty) for s in tuples_indices}
    order_opt = min(cost, key=cost.get)
    val = min(cost.values())
    print('DP OPT :', val, 'Stores : ', list(order_opt))
    return val


def main():
    n_levels = 3
    big_penalty = 100
    types = []
    for string in map(''.join, itertools.product('LH', repeat=4)):
        types.append(string)
    gap = {ty: 0 for ty in types}

    for t in types:
        gap_pick = []
        gap_ship = []
        gap_pick_ship = []
        opt = []
        for i in range(10):
            stores = read_data(i, t)  # Argument: Number of items
            print('Type : ', t, ', Instance : ', i)

            dp_opt = solve_opt_dp_norepeat(stores, n_levels, big_penalty)
            gap_pick.append(solve_picking_norepeat(stores, n_levels, big_penalty))
            gap_ship.append(solve_shipping_norepeat(stores, n_levels, big_penalty))
            gap_pick_ship.append(solve_pick_and_ship_norepeat(stores, n_levels, big_penalty))
            opt.append(dp_opt)
            print('\n')

        avg_opt = sum(opt)/len(opt)
        avg_pick = sum(gap_pick)/len(gap_pick)
        avg_ship = sum(gap_ship)/len(gap_ship)
        avg_pick_ship = sum(gap_pick_ship) / len(gap_pick_ship)
        gap[t] = [avg_opt, avg_pick, avg_ship, avg_pick_ship]

    f = open('Results_Single_Item/Costs.txt', 'w')

    for g in gap:
        f.write(g+'g'+str(gap[g])+'\n')
    for g in gap:
        print(g, gap[g])
    f.close()


if __name__ == "__main__":
    main()
