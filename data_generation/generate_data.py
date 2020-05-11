import json
import sys
import math
import random
import numpy as np


class Sample:
    def __init__(self, retailer):
        coeffs = dict(A=dict(eslope=-0.81343218, eintercept=13.48328815, pintercept=10.48634123, pslope=-0.87142997),
                      B=dict(eslope=-0.79160416, eintercept=12.46597743, pintercept=11.83223274, pslope=-1.2682217),
                      C=dict(eslope=-1.21334127, eintercept=12.88765399, pintercept=10.19523483, pslope=-1.20083862),
                      D=dict(eslope=-1.09965963, eintercept=10.77349155, pintercept=9.64185248, pslope=-1.13472406),
                      E=dict(eslope=-1.10382723, eintercept=12.47153335, pintercept=11.94578295, pslope=-1.39746398))

        self.pslope = coeffs[retailer]['pslope']
        self.pintercept = coeffs[retailer]['pintercept']
        self.eslope = coeffs[retailer]['eslope']
        self.eintercept = coeffs[retailer]['eintercept']
        self.pxmax = 500

        self.py = [math.exp(self.pslope * math.log(x) + self.pintercept) for x in
                   range(1, self.pxmax)]
        self.ptotal = sum(self.py)
        self.pprob = [a / self.ptotal for a in self.py]

        self.eslope = coeffs[retailer]['eslope']
        self.eintercept = coeffs[retailer]['eintercept']
        self.exmax = 10

        self.ey = [math.exp(self.eslope * x + self.eintercept) for x in range(1, self.exmax)]
        self.etotal = sum(self.ey)
        self.eprob = [a / self.etotal for a in self.ey]

    def sample_power(self):
        r = random.random()
        cumulative = 0
        for i in range(1, self.pxmax):
            cumulative += self.pprob[i - 1]
            if r < cumulative:
                return i
        return self.pxmax

    def sample_exponential(self):
        r = random.random()
        cumulative = 0
        for i in range(1, self.exmax):
            cumulative += self.eprob[i - 1]
            if r < cumulative:
                return i
        return self.exmax


zone_dist = [.058, .060, .173, .268, .144, .091, .206]
#  zone_cost = [[4.78, 5.21, 5.99, 6.27, 6.41, 6.75, 7.27],
#             [6.70, 7.15, 7.30, 7.42, 7.59, 7.77, 8.27],
#             [7.25, 7.70, 8.75, 9.56, 10.33, 11.52, 12.72],
#             [7.90, 8.80, 10.15, 11.40, 12.68, 14.24, 16.69],
#             [8.50, 9.90, 11.15, 12.05, 14.56, 17.08, 19.60],
#             [9.85, 10.95, 11.95, 13.08, 16.29, 19.50, 22.71]]


def sample_shipping():
    r = random.random()
    cumulative = 0

    zone = len(zone_dist) - 1

    for i in range(len(zone_dist)):
        cumulative += zone_dist[i]
        if r < cumulative:
            zone = i
            break

    return zone
    #  weight = int(5. * random.random())
    #  weight = min(weight, 5)
    #  return zone_cost[weight][zone] / 4.0


# store_dist=[.0007, .0053, .0679, .1696, .3161, .4405, .0]
# store_cost=[2.74, 2.31, 1.92, 1.40, 2.89, 1, 1]
# store_dist=[.058, .060, .173, .268, .144, .091, .206]
# store_cost=[4.78, 5.21, 5.99, 6.27, 6.41, 6.75, 7.27]


def sample_labor():
    data = [a.strip().split(',') for a in open('data_to_sample/store_volume_labor2.csv').readlines()]

    total_volume = 0
    for _, volume, _ in data: # _ id, _ labor
        total_volume += int(volume)

    for i in range(len(data)):
        data[i][1] = float(data[i][1]) / total_volume
    r = random.random()

    cumulative = 0
    for i in range(len(data)):
        cumulative += data[i][1]
        if r < cumulative:
            return float(data[i][2]) / 4.
    return float(data[len(data) - 1][2]) / 4.


def read_tsv(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = []
    header = lines[0].split('\t')
    for line in lines[1:]:
        data.append(dict(zip(header, line.split('\t'))))
    return data


def get_s(num_stores):
    return [sample_shipping() for _ in range(num_stores)]


def get_c(items_to_sample, num_stores, single):
    if single:
        return [sample_labor() for _ in range(num_stores)]
    else:
        return [[sample_labor() for _ in range(items_to_sample)] for _ in range(num_stores)]


def get_sample(data, fields, num_samples):
    samples = []
    for _ in range(num_samples):
        r = random.randint(0, len(data) - 1)
        samples.append(dict(filter(lambda x: x[0] in fields,
                                   data[r].items())))
    return samples


def sample_store(_store_hist_file):
    hist = [line.strip('\n').split('\t') for line in open(_store_hist_file).readlines()]
    x_grid = [x[0] for x in hist]
    pdf = [float(x[1]) for x in hist]
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return x_grid[np.searchsorted(cdf, random.random())]


def load_pick_rates(pick_rate_file):
    hist = [line.strip('\n').split('\t') for line in open(pick_rate_file).readlines()]
    return {h[0]: round(max(min(0.9, 1 - float(h[1])), 0.1), 4) for h in hist}


def generate_data_multiple_sampling(data_to_sample, items_to_sample,
                                    store_vol_hist_file, _pick_hist_file, sample_object, _single):

    #  print('#Items', items_to_sample)
    sampled_items = get_sample(data_to_sample, ['product_unique_identifier',
                                                'price_discount_level',
                                                'location_inventory_on_hand',
                                                'product_code',
                                                'vendor',
                                                'division_id',
                                                'department_id',
                                                'category_id',
                                                'subcategory_id',
                                                'advertised_price',
                                                'success'], items_to_sample)

    # Store map is a dictionary that maps real store numbers to integers 0,1,2,....
    store_map = {}
    links = []

    for i, _ in enumerate(sampled_items):
        num_stores = sample_object.sample_power()
        sampled_stores = [sample_store(store_vol_hist_file) for _ in range(num_stores)]
        #  print('For item ', i, 'sampled stores : ', sampled_stores)
        for store in sampled_stores:
            if store not in store_map:
                store_map[store] = len(store_map)
            new_link = [i, store_map[store]]
            if new_link not in links:
                links.append(new_link)

    pick_rate = load_pick_rates(_pick_hist_file)

    if _single:
        p = [pick_rate[store] for store in store_map]
    else:
        p = [[pick_rate[store] for _ in range(items_to_sample)] for store in store_map]

    s = get_s(len(store_map))
    c = get_c(items_to_sample, len(store_map), _single)

    return [[{"p": p[i], "s": s[i], "c": c[i]} for i in range(len(store_map))], links]


def item_sampling(data_to_sample, store_vol_hist, pick_hist, _sample_obj, single):
    if single:
        num_items = 1
    else:
        num_items = _sample_obj.sample_exponential()
        while num_items == 1:
            num_items = _sample_obj.sample_exponential()
    data = generate_data_multiple_sampling(data_to_sample, num_items, store_vol_hist, pick_hist, _sample_obj, single)

    if single:
        return data[0]
    else:
        return data


if __name__ == '__main__':
    retailer = sys.argv[1]
    instance = sys.argv[2]
    isSingleItem = {'True': True, 'False': False}.get(sys.argv[3])

    store_hist_file = 'data_to_sample/retailer_{}/store_vol_hist'.format(retailer)
    pick_hist_file = 'data_to_sample/retailer_{}/pick_hist'.format(retailer)

    master_file = 'data_to_sample/retailer_{}/ld.tsv'.format(retailer)
    dts = read_tsv(master_file)

    sample_obj = Sample(retailer)

    generated_data = item_sampling(dts, store_hist_file, pick_hist_file, sample_obj, isSingleItem)

    if isSingleItem:
        output_file = 'generated_data/si/retailer_{}/{}.{}.json'.format(retailer, retailer, instance)
    else:
        output_file = 'generated_data/mi/retailer_{}/{}.{}.json'.format(retailer, retailer, instance)

    with open(output_file, "w") as outfile:
        json.dump(generated_data, outfile)
    outfile.close()
