import json
import order_fulfillment_multi_item as ofmi


def read_data(dataFile):
    file = open(dataFile, 'r')
    inst = json.load(file)
    file.close()
    return inst


fileout = 'summary_stats/costs_multi_item.txt'
algorithms = ['LB', 'LSC', '6-SPS', '4-SPS', '2-SPS', '1-SPS', 'Greedy']
retailers = ['A', 'B', 'C', 'D', 'E']


for retailer in retailers:
    for instance in range(1, 101):
        for alg in algorithms:
            filename = 'data/mi/retailer_{}/{}.{}.json'.format(retailer, retailer, instance)
            print('File: ', filename)
            data = read_data(filename)
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
                retailer, instance, nItems, storeDistribution, overlap_factor, alg, cost, assignment)

            f.write(row)
            f.close()
