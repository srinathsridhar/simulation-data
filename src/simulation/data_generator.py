import json
import os
import sys
import random
import subprocess
import uuid

import math
import random
import numpy as np

class Sample:
    def __init__(self, merchant):
      coeffs = {'merchant1': {'eslope': -0.81343218,
                              'eintercept': 13.48328815,
                              'pintercept': 10.48634123,
                              'pslope': -0.87142997
                              },
                'merchant2': {'eslope': -0.79160416,
                             'eintercept': 12.46597743,
                             'pintercept': 11.83223274,
                             'pslope': -1.2682217
                             },
                'merchant3': {'eslope':-1.21334127,
                            'eintercept': 12.88765399,
                            'pintercept': 10.19523483,
                            'pslope': -1.20083862
                            },
                'merchant4': {'eslope': -1.09965963,
                           'eintercept': 10.77349155,
                           'pintercept': 9.64185248,
                           'pslope': -1.13472406
                           },
                'merchant5': {'eslope': -1.10382723,
                         'eintercept': 12.47153335,
                         'pintercept': 11.94578295,
                         'pslope': -1.39746398
                         },
                }
      self.pslope = coeffs[merchant]['pslope']
      self.pintercept = coeffs[merchant]['pintercept']
      self.eslope = coeffs[merchant]['eslope']
      self.eintercept = coeffs[merchant]['eintercept']
      self.pxmax = 500

      self.py = [math.exp(self.pslope * math.log(x) + self.pintercept) for x in
                 range(1, self.pxmax)]
      self.ptotal = sum(self.py)
      self.pprob = [a / self.ptotal for a in self.py]

      self.eslope = eslope
      self.eintercept = eintercept
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

pslope=--0.87142997
pintercept=10.48634123
pxmax=500

py=[math.exp(pslope * math.log(x) + pintercept) for x in range(1, pxmax)]
ptotal=sum(py)
pprob=[a / ptotal for a in py]

def sample_power():
  r = random.random()
  cumulative = 0
  for i in range(1, pxmax):
    cumulative += pprob[i - 1]
    if r < cumulative:
      return i
  return pxmax

eslope=-0.81343218
eintercept=13.48328815
exmax=10

ey=[math.exp(eslope * x + eintercept) for x in range(1, exmax)]
etotal=sum(ey)
eprob=[a / etotal for a in ey]

def sample_exponential():
  r = random.random()
  cumulative = 0
  for i in range(1, exmax):
    cumulative += eprob[i - 1]
    if r < cumulative:
      return i
  return exmax


zone_dist=[.058, .060, .173, .268, .144, .091, .206]
zone_cost=[[4.78, 5.21, 5.99, 6.27, 6.41, 6.75, 7.27], \
           [6.70, 7.15, 7.30, 7.42, 7.59, 7.77, 8.27], \
           [7.25, 7.70, 8.75, 9.56, 10.33, 11.52, 12.72], \
           [7.90, 8.80, 10.15, 11.40, 12.68, 14.24, 16.69], \
           [8.50, 9.90, 11.15, 12.05, 14.56, 17.08, 19.60], \
           [9.85, 10.95, 11.95, 13.08, 16.29, 19.50, 22.71]]


def sample_shipping():
  r = random.random()
  cumulative = 0
  zone = len(zone_dist) - 1

  for i in range(len(zone_dist)):
    cumulative += zone_dist[i]
    if r < cumulative:
      zone = i
      break

  weight = int(5. * random.random())
  weight = min(weight, 5)

  return zone_cost[weight][zone]/4.0

#store_dist=[.0007, .0053, .0679, .1696, .3161, .4405, .0]
#store_cost=[2.74, 2.31, 1.92, 1.40, 2.89, 1, 1]
#store_dist=[.058, .060, .173, .268, .144, .091, .206]
#store_cost=[4.78, 5.21, 5.99, 6.27, 6.41, 6.75, 7.27]

data = [a.strip().split(',') for a in open('./store_volume_labor2.csv').readlines()]
total_volume = 0
for id, volume, labor in data:
    total_volume += int(volume)

for i in range(len(data)):
    data[i][1] = float(data[i][1]) / total_volume

def sample_labor():
    r = random.random()

    cumulative = 0
    for i in range(len(data)):
        cumulative += data[i][1]
        if r < cumulative:
            return float(data[i][2]) / 4.
    return float(data[len(data) - 1][2]) / 4.

#def sample_labor():
#  r = random.random()

#  cumulative = 0
#  for i in range(len(store_dist)):
#    cumulative += store_dist[i]
#    if r < cumulative:
#      return store_cost[i]
#  return store_cost[len(store_dist) - 1]


def sample_store(store_hist_file):
  hist = map(lambda line: line.strip('\n').split('\t'),
             open(store_hist_file).readlines())
  x_grid = map(lambda x: x[0], hist)
  pdf = map(lambda x: float(x[1]), hist)
  cdf = np.cumsum(pdf)
  cdf = cdf / cdf[-1]
  return x_grid[np.searchsorted(cdf, random.random())]

def load_pick_rates(pick_rate_file):
  hist = map(lambda line: line.strip('\n').split('\t'),
             open(pick_rate_file).readlines())
  return {h[0]: max(min(0.9, 1-float(h[1])), 0.1) for h in hist}


def get_sample(data, fields, num_samples):
  samples = []
  for _ in range(num_samples):
    r = random.randint(0, len(data)-1)
    samples.append(dict(filter(lambda x:x[0] in fields,
                               data[r].items())))
  return samples

def get_p(sampled_items, sampled_stores, model, single):
  prediction_format = {"fulfillment_location":str,
                       "product_unique_identifier":str,
                       "price_discount_level":str,
                       "location_inventory_on_hand":int,
                       "product_code":str,
                       "vendor":str,
                       "division_id":str,
                       "department_id":str,
                       "category_id":str,
                       "subcategory_id":str,
                       "advertised_price":float}

  uuid_str = uuid.uuid1().urn.split(':')[-1]
  f = open('/tmp/{}.json'.format(uuid_str), 'w')
  samples = {"instances":[]}
  for store in sampled_stores:
    sample = {}
    sample['fulfillment_location'] = store['fulfillment_location']
    for item in sampled_items:
      for k in item:
        val = item[k]
        if prediction_format[k] == float and item[k] == '':
            val = 0.0
        if prediction_format[k] == int and item[k] == '':
            val = 0
        #print k,val
        sample[k] = prediction_format[k](val)
      samples['instances'].append(sample.copy())

  f.write(json.dumps(samples))
  f.close()
  out = subprocess.check_output(['/usr/bin/java',
                                 '-Xmx22g',
                                 '-cp',
                                 '/mnt/release/atp-1.0-SNAPSHOT-jar-with-dependencies.jar',
                                 'com.onera.atp.estimator.Predictions',
                                 '-f',
                                 '/tmp/{}.json'.format(uuid_str),
                                 '-m',
                                 model]).strip()
  #res = map(lambda x: map(max(0.1,(min(round(1-float(x), 3), 0.9)))), out.split('\n'))
  res = map(lambda x: max(0.1, min(0.9, round(1-float(x), 3))), out.split('\n'))
  n = len(sampled_items)
  if not single:
    result = [res[i:i+n] for i in range(0, len(res), n)]
    return result
  else:
    return res

'''
def replace_p_single(data, pick_probs):
  for i,d in enumerate(data):
    d['p'] = round(float(pick_probs[i]), 3)
  return data

def replace_p_multi(data, pick_probs):
  count = 0

  for i,d in enumerate(data[0]):
    l = len(d['p'])
    d['p'] = [round(float(p), 3) for p in pick_probs[count:count+l]]
    count += l
  return data
'''

def read_tsv(filename):
  f = open(filename)
  lines = f.readlines()
  f.close()
  data = []
  header = lines[0].split('\t')
  for line in lines[1:]:
    data.append(dict(zip(header, line.split('\t'))))
  return data

def get_s(num_stores, single):
  return [sample_shipping() for _ in range(num_stores)]

def get_c(items_to_sample, num_stores, single):
  if single:
    return [sample_labor() for _ in range(num_stores)]
  else:
    return [[sample_labor() for _ in range(items_to_sample)] for _ in range(num_stores)]


def generate_data(model, data_to_sample, items_to_sample=1, single=True):
  sampled_item = get_sample(data_to_sample, ['product_unique_identifier',
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
  num_stores = sample_power()
  sampled_stores = get_sample(data_to_sample, ['fulfillment_location'], num_stores)
  p = get_p(sampled_item, sampled_stores, model, single)
  s = get_s(num_stores, single)
  c = get_c(items_to_sample, num_stores, single)
  return [{"p": p[i], "s": s[i], "c": c[i]} for i in range(num_stores)]

def generate_data_sampling(store_vol_hist_file, pick_hist_file, sample_obj):
  '''
  sampled_item = get_sample(data_to_sample, ['product_unique_identifier',
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
  '''
  num_stores = sample_obj.sample_power()
  sampled_stores = [sample_store(store_vol_hist_file) for _ in range(num_stores)]
  pick_rate = load_pick_rates(pick_hist_file)
  p = [pick_rate[store] for store in sampled_stores]
  s = get_s(num_stores, True)
  c = get_c(1, num_stores, True)
  return [{"p": p[i], "s": s[i], "c": c[i]} for i in range(num_stores)]

def generate_data_multiple(model, data_to_sample, items_to_sample=1, single=False):
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
    store_map = {}
    links = []
  
    for i,item in enumerate(sampled_items):
      num_stores = sample_power()
      sampled_stores = get_sample(data_to_sample, ['fulfillment_location'], num_stores)

      for store in sampled_stores:
        if store['fulfillment_location'] not in store_map:
            store_map[store['fulfillment_location']] = len(store_map)
        new_link = [i, store_map[store['fulfillment_location']]]
        if new_link not in links:
            links.append(new_link)

    store_list = [{'fulfillment_location': store_number} for store_number, store_id in sorted(store_map.items(), key=lambda x: x[1])]

    #print store_list
    #print sampled_items
    #print items_to_sample
    p = get_p(sampled_items, store_list, model, False)
    s = get_s(len(store_list), False)
    c = get_c(items_to_sample, len(store_list), False)
    return [[{"p": p[i], "s": s[i], "c": c[i]} for i in range(len(store_list))], links]

def generate_data_multiple_sampling(data_to_sample, items_to_sample, store_vol_hist_file, pick_hist_file, sample_obj):
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
    store_map = {}
    links = []
  
    for i,item in enumerate(sampled_items):
      num_stores = sample_obj.sample_power()
      #sampled_stores = get_sample(data_to_sample, ['fulfillment_location'], num_stores)
      sampled_stores = [sample_store(store_vol_hist_file) for _ in range(num_stores)]

      for store in sampled_stores:
        if store not in store_map:
            store_map[store] = len(store_map)
        new_link = [i, store_map[store]]
        if new_link not in links:
            links.append(new_link)

    pick_rate = load_pick_rates(pick_hist_file)
    p = [[pick_rate[store] for _ in range(items_to_sample)] for store in store_map]
    s = get_s(len(store_map), False)
    c = get_c(items_to_sample, len(store_map), False)

    return [[{"p": p[i], "s": s[i], "c": c[i]} for i in range(len(store_map))], links]



def single_item(model, data_to_sample):
  return generate_data(model, data_to_sample)

def single_item_sampling(store_vol_hist, pick_hist, sample_obj):
  return generate_data_sampling(store_vol_hist, pick_hist, sample_obj)


def multi_item(model, data_to_sample):
  num_items = sample_exponential()
  while num_items == 1:
      num_items = sample_exponential()

  return generate_data_multiple(model, data_to_sample, num_items, False)

def multi_item_sampling(data_to_sample, store_vol_hist, pick_hist, sample_obj):
  num_items = sample_obj.sample_exponential()
  while num_items == 1:
      num_items = sample_obj.sample_exponential()

  return generate_data_multiple_sampling(data_to_sample, num_items, store_vol_hist, pick_hist, sample_obj)



if __name__ == '__main__':
  model = sys.argv[1]
  data = read_tsv(sys.argv[2])
  single_item(model, data)
