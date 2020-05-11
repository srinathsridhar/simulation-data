# Order Fulfillment in Omnichannel SFS Programs

This repository containing data ad codes serves as supporting material for the article *Order Fulfillment in Omnichannel SFS Programs*.

The codes can be divided into two components: data_generation and solve_order_fulfillment. Data generation essentially takes raw data provided by Onera and converts it into instances of orders. There are 5 retailers, namely A, B, C, D and E. solve_order_fulfillment takes each instance of the data and applies the algorithms/heuristics that we have developed to that instance. It also features plot generation codes, which essentially takes summary statistics and converts them into meaningful plots.

## Data Generation

All files concerned with data deneration are located in the folder *data_generation*.

* The raw data provided by Onera is in the folder *data_to_sample*. 
  * The *store_volume_labor2.csv* file has 3 columns, namely store, volume and minimum wages respectively. This file is used to generate labor costs at stores.
  * Each retailer has a folder with 3 files: *ld.tsv*, *pick_hist* and *store_vol_hist*
    * *ld.tsv* is a master file. A row represents an item that is ordered. The rest can be interpretedd using the informative headers.
    * *pick_hist* has two columns, first being the store and the second being the pick failure probability. This file is used to generate pick failure probabilities for stores.
    * *store_vol_hist* has two columns one being the store and the second being the volume of items shipped from that store. This file is used to sample stores for the items in an order.
* *generate_data.py* is the primary code that takes 3 arguments: retailer, instance and a boolean variable indicating whether the order is a single-item order or a multi-item order. 
* *iterate_generate_data.py* just calls the *generate_data.py* file with appropriate arguments.
* The generated data is stored in the folder generated_data. There are subfolders for single-item/muli-item and the retailers. For example,  the 65th single-item instance for retailer D is in the folder *generated_data/si/retailer_D/D.65.json*
  * Each data instance of a retailer is a *.json* file. 
  * A single-item data instance is a list of stores, each store represented by a dictionary with keys *p*, *c* and *s* representing the pick failure probability, labor cost and shipping zone respectively.
  * A multi-item data instance is a list of two lists:
    * First is a list of stores, each store represented by a dictionary of lists with keys *p*, *c* and *s* representing the pick failure probability, labor cost and shipping zone respectively. An element in the list of the dictionary corressponds to item in the order.
    * Second is a list of (item-store) tuples, representing whether that item can be shipped from that store.

## Solving

All files are located in the folder *order_fulfillment*

* The generated data is copied into the folder *data*.
* Single-item
  * *order_fulfillment_single_item.py* is the primary code that takes as input the retailer, instance and algorithm, and gives as output the cost and the store allocation per stage.
  * *iterate_single_item.py* just calls the *order_fulfillment_single_item.py* with the appropriate arguments
  * All summary statistics or output is stored in the file *summary_stats/costs_single_item.txt*.
  * *costs_single_item.txt* has 7 columns representing retailer, instance, # stores, cost of OPT, cost of greedy, store allocation of opt and store allocation of greedy.
* Multi-item
  * *order_fulfillment_multi_item.py* is the primary code that takes as input the data instance and algorithm, and gives as output the cost and the store allocation per stage per item.
  * *iterate_multi_item.py* just calls the *order_fulfillment_single_item.py* with the appropriate arguments
  * All summary statistics or output is stored in the file *summary_stats/costs_multi_item.txt*.
  * *costs_multi_item.txt* has 8 columns representing retailer, instance, # stores, # store per item, *ovf*, algorithm, fulfillment cost, fulfillment policy (store assignment per item per stage).

## Plot Generation

The folder *order_fulfillment* contains all the relevant files. All the functions to generate plots are in *generate_plots.py* file.

* *box_plot_single_item()* takes summary statistics from the file *summary_stats/costs_single_item.txt* and plots the box plots for each retailer.
* *box_plots_multi_item()* takes summary statistics from the file *summary_stats/costs_multi_item.txt* and plots the box plots for each retailer.
* *box_plots_full_overlap()* modifies the data by making each store connected to each item for all instances, storing summary staistics in *summary_stats/costs_multi_item_full_overlap.txt* and plots the box plots for each retailer.
* *box_plots_shipcost_lowhigh()* modifies the data by making two classes one with higher and one with lower shipping costs, and then storing summary staistics in *summary_stats/costs_multi_item_shipcost_lowhigh.txt* and plots the box plots for each retailer.
* *box_plots_pickfailprob_lowhigh()* modifies the data by making two classes one with higher and one with lower pick failure probabilities, and then storing summary staistics in *summary_stats/costs_multi_item_pickfailprob_lowhigh.txt* and plots the box plots for each retailer.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.