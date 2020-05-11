import json
import order_fulfillment_multi_item as ofmi
import random


def read_data(dataFile, ship_cost):
    file = open(dataFile, 'r')
    inst = json.load(file)
    file.close()
    if ship_cost == 'H':
        for _store in inst[0]:
            for _item in range(len(_store['p'])):
                if _store['p'][_item] < 0.9:
                    _store['p'][_item] = random.uniform(0.5, 0.8)
    else:
        for _store in inst[0]:
            for _item in range(len(_store['p'])):
                if _store['p'][_item] > 0.1:
                    _store['p'][_item] = random.uniform(0.05, 0.1)

    return inst


algorithms = ['LB', 'LSC', '6-SPS', '4-SPS', '2-SPS', '1-SPS', 'Greedy']
retailers = ['A', 'B', 'C', 'D', 'E']


for pickFailProb in ['H', 'L']:
    if pickFailProb == 'H':
        fileout = 'summary_stats/costs_multi_item_high_pickfailprob.txt'
    else:
        fileout = 'summary_stats/costs_multi_item_low_pickfailprob.txt'

    for retailer in retailers:
        for instance in range(1, 101):
            for alg in ['LSC', '2-SPS', 'Greedy']:
                filename = 'data/mi/retailer_{0}/{0}.{1}.json'.format(retailer, instance)
                print('file', filename)
                data = read_data(filename, pickFailProb)
                cost, assignment = ofmi.solve_multi_item(alg, data)

                nStores = len(data[0])
                nItems = len(data[0][0]['p'])

                storeDistribution = [0 for _ in range(nItems)]
                for item, store in data[1]:
                    storeDistribution[item] += 1

                numItemsPerStore = [-1 for _ in range(nStores)]
                for [item, store] in data[1]:
                    numItemsPerStore[store] += 1
                overlap_factor = round(sum(numItemsPerStore) / (nStores * (nItems - 1)), 3)

                f = open(fileout, "a+")
                row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    retailer, instance, nItems, storeDistribution,overlap_factor,alg, cost, assignment)

                f.write(row)
                f.close()
