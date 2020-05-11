from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf
import json
import numpy as np
import statistics

retailers = ['A', 'B', 'C', 'D', 'E']
cancellation_cost = 25

# box plot and generate tables for single item
def box_plot_single_item():
    data_opt = [[] for _ in retailers]
    data_greedy = [[] for _ in retailers]
    data_saving = {retailer: [] for retailer in retailers}
    filename = 'summary_stats/costs_single_item.txt'
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    retailer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for line in lines[1:]:

        splitLine = line.split('\t')
        retailer = splitLine[0]
        retailer_id = retailer_map[retailer]
        opt = float(splitLine[3])
        data_opt[retailer_id].append(opt)
        greedy = float(splitLine[4])
        data_greedy[retailer_id].append(greedy)
        saving = round((greedy-opt)*100/greedy, 5)

        data_saving[retailer].append(saving)

    ticks = retailers

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    pdf_box = matplotlib.backends.backend_pdf.PdfPages('plots/si/box_plot_single_item.pdf')

    plt.figure()

    bpl = plt.boxplot(data_opt, positions=np.array(range(len(data_opt))) * 2.0 - 0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_greedy, positions=np.array(range(len(data_greedy))) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    formatter = ticker.FormatStrFormatter('$%1.0f')

    plt.axes().yaxis.set_major_formatter(formatter)
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='OPT')
    plt.plot([], c='#2C7BB6', label='Greedy')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
 #   plt.xlim(-2, len(ticks) * 2)
 #   plt.ylim(0, 8)
    plt.tight_layout()

  #  plt.title()
    pdf_box.savefig()
    plt.close()
    pdf_box.close()

    average_saving = {retailer: sum(data_saving[retailer])/100 for retailer in retailers}
    print('Savings', average_saving)


# box plot for Retailer E fulfillment and cancellation costs
def box_plot_single_item_cancellation_probability():
    data_opt = [[], []]
    data_greedy = [[], []]
    filename = 'summary_stats/costs_single_item.txt'
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    retailer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for line in lines[1:]:
        splitLine = line.split('\t')
        retailer = splitLine[0]
        if retailer == 'E':
            instance = int(splitLine[1])
            optCost = float(splitLine[3])
            greedyCost = float(splitLine[4])
            optAssignment = json.loads(splitLine[5])
            greedyAssignment = json.loads(splitLine[6])
            file = 'data/si/retailer_{0}/{0}.{1}.json'.format(retailer, instance)
            with open(file) as json_file:
                data = json.load(json_file)

            optCancellationCost = cancellation_cost
            for store in optAssignment:
                optCancellationCost *= data[store]['p']
            greedyCancellationCost = cancellation_cost
            for store in greedyAssignment:
                greedyCancellationCost *= data[store]['p']

            data_opt[1].append(optCancellationCost)
            data_greedy[1].append(greedyCancellationCost)

            data_opt[0].append(optCost - optCancellationCost)
            data_greedy[0].append(greedyCost - greedyCancellationCost)

    ticks = ['Fulfillment Costs', 'Cancellation Costs']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    pdf_box = matplotlib.backends.backend_pdf.PdfPages\
        ('plots/si/box_plot_single_item_retailer_E_fulfillment_cancellations_costs.pdf')

    plt.figure()

    bpl = plt.boxplot(data_opt, positions=np.array(range(len(data_opt))) * 2.0 - 0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_greedy, positions=np.array(range(len(data_greedy))) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    formatter = ticker.FormatStrFormatter('$%1.0f')

    plt.axes().yaxis.set_major_formatter(formatter)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='OPT')
    plt.plot([], c='#2C7BB6', label='Greedy')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
 #   plt.xlim(-2, len(ticks) * 2)
 #   plt.ylim(0, 8)
    plt.tight_layout()

  #  plt.title()
    pdf_box.savefig()
    plt.close()
    pdf_box.close()


def read_data(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    cost_data = {}
    header = lines[0].split('\t')
    for line in lines[1:]:

        splitLine = line.split('\t')
        retailer = splitLine[0]
        instance = int(splitLine[1])
        alg = splitLine[5]
        cost = float(splitLine[6])
        cost_data[retailer, instance, alg] = cost
    return cost_data


#  box plot and generate table for multi-item original data
def box_plots_multi_item():
    cost = read_data('summary_stats/costs_multi_item.txt')

    for retailer in retailers:
        pdf_box = matplotlib.backends.backend_pdf.PdfPages('plots/mi/box_plot_retailer_{}.pdf'.format(retailer))
        print('filename', 'plots/box_plot_retailer_{}.pdf'.format(retailer))
        plt.figure()

        algorithms = ['LB', 'LSC', '6-SPS', '4-SPS', '2-SPS', '1-SPS', 'Greedy']
        plot_data = {alg: [] for alg in algorithms}
        for key, value in cost.items():
            if key[0] == retailer:
              #  print('alf',key[2])
                if key[2] in algorithms:
                    plot_data[key[2]].append(value)

        data_compressed = [plot_data[alg] for alg in algorithms]
     #   print('data_compressed', data_compressed)
        plt.boxplot(data_compressed, showfliers=False, medianprops={'color': 'black'})
        label_algorithms = ['LB', 'LSC', '6-SPS', '4-SPS', '2-SPS', '1-SPS', 'Greedy']
        plt.title('Retailer {}'.format(retailer))
        formatter = ticker.FormatStrFormatter('$%1.0f')
        plt.axes().yaxis.set_major_formatter(formatter)
        plt.xticks(range(1, 8), label_algorithms)
        plt.tight_layout()
        pdf_box.savefig()
        plt.close()
        pdf_box.close()

        savingsOverGreedy = {alg: 0 for alg in algorithms[1:6]}
        for alg in algorithms[1:6]:
            for inst in range(100):
                savingsOverGreedy[alg] += (plot_data['Greedy'][inst] - plot_data[alg][inst])*100\
                                          /plot_data['Greedy'][inst]
            savingsOverGreedy[alg] /= 100
        print('Retailer', retailer, 'savings', savingsOverGreedy)


#  We divide the original data into high and low overlap classes
#  Not in draft anymore
#  Deprecated
def box_plots_overlap():
    cost = read_data('summary_stats/costs_multi_item.txt')
    overlap_factor = {retailer: [] for retailer in retailers}

    for retailer in retailers:
        for instance in range(1, 101):
            data_file = 'data/mi/retailer_{0}/{0}.{1}.json'.format(retailer, instance)
            file = open(data_file, 'r')
            data = json.load(file)
            file.close()
            nItems = len(data[0][0]['p'])
            nStores = len(data[0])
            numItemsPerStore = [-1 for _ in range(nStores)]
            for [_, store] in data[1]:
                numItemsPerStore[store] += 1
            overlap_factor[retailer].append(round(sum(numItemsPerStore) / (nStores * (nItems-1)), 3))

    avg_ov_fac = {retailer: statistics.mean(overlap_factor[retailer]) for retailer in retailers}
    std_ov_fac = {retailer: statistics.stdev(overlap_factor[retailer]) for retailer in retailers}
    print('Avg Overlap', avg_ov_fac)
    algorithms = ['LSC', '2-SPS', 'Greedy']
    overlaps = ['L', 'H']
    overlap_map = {'L': 0, 'H': 1}
    lift = {retailer: {overlap: 0 for overlap in overlaps} for retailer in retailers}

    for retailer in retailers:
        data_SPS = [[] for _ in overlaps]
        data_LSC = [[] for _ in overlaps]
        data_Greedy = [[] for _ in overlaps]

        for key, value in cost.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    if overlap_factor[key[0]][key[1]-1] < avg_ov_fac[retailer]:
                        data_SPS[overlap_map['L']].append(value)
                    elif overlap_factor[key[0]][key[1]-1] > avg_ov_fac[retailer]+std_ov_fac[retailer]:
                        data_SPS[overlap_map['H']].append(value)
                if key[2] == 'LSC':
                    if overlap_factor[key[0]][key[1]-1] < avg_ov_fac[retailer]:
                        data_LSC[overlap_map['L']].append(value)
                    elif overlap_factor[key[0]][key[1]-1] > avg_ov_fac[retailer]+std_ov_fac[retailer]:
                        data_LSC[overlap_map['H']].append(value)
                if key[2] == 'Greedy':
                    if overlap_factor[key[0]][key[1]-1] < avg_ov_fac[retailer]:
                        data_Greedy[overlap_map['L']].append(value)
                    elif overlap_factor[key[0]][key[1]-1] > avg_ov_fac[retailer]+std_ov_fac[retailer]:
                        data_Greedy[overlap_map['H']].append(value)
        ticks = overlaps

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        pdf_box = matplotlib.backends.backend_pdf.PdfPages('plots/mi/overlap/box_plot_overlap_{}.pdf'.format(retailer))

        plt.figure()

        bpl = plt.boxplot(data_LSC, positions=np.array(range(len(data_LSC))) * 2 - 0.5, sym='', widths=0.3)
        bpm = plt.boxplot(data_SPS, positions=np.array(range(len(data_SPS))) * 2, sym='', widths=0.3)
        bpr = plt.boxplot(data_Greedy, positions=np.array(range(len(data_Greedy))) * 2 + 0.5, sym='', widths=0.3)
        set_box_color(bpl, 'blue')  # colors are from http://colorbrewer2.org/
        set_box_color(bpm, 'red')
        set_box_color(bpr, 'green')

        formatter = ticker.FormatStrFormatter('$%1.0f')
        plt.axes().yaxis.set_major_formatter(formatter)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label='LSC')
        plt.plot([], c='red', label='SPS')
        plt.plot([], c='green', label='SPS')
        plt.legend()
        plt.title('Retailer {}'.format(retailer))
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.tight_layout()

        #  plt.title()
        pdf_box.savefig()
        plt.close()
        pdf_box.close()

        for overlap in overlaps:
            lift_inst = 0
            for i in range(len(data_LSC[overlap_map[overlap]])):

                LSC = data_LSC[overlap_map[overlap]][i]
                SPS = data_SPS[overlap_map[overlap]][i]
                lift_inst += round((SPS-LSC)*100/SPS, 5)

            lift_inst /= 100

            lift[retailer][overlap] = round(lift_inst, 3)

    print('Lift Matrix A', lift['A'])
    print('Lift Matrix B', lift['B'])
    print('Lift Matrix C', lift['C'])
    print('Lift Matrix D', lift['D'])
    print('Lift Matrix E', lift['E'])


# We compare original data with full overlap data
def box_plots_full_overlap():
    cost_high = read_data('summary_stats/costs_multi_item_full_overlap.txt')
    cost_low = read_data('summary_stats/costs_multi_item.txt')
    algorithms = ['LSC', '2-SPS', 'Greedy']
    high_low = ['L', 'H']
    high_low_map = {'L': 0, 'H': 1}
    lift = {retailer: {hl: 0 for hl in high_low} for retailer in retailers}

    for retailer in retailers:
        data_SPS = [[] for _ in high_low]
        data_LSC = [[] for _ in high_low]
        data_Greedy = [[] for _ in high_low]

        for key, value in cost_high.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['H']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['H']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['H']].append(value)

        for key, value in cost_low.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['L']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['L']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['L']].append(value)

        #  print('data_sps', data_SPS)

        ticks = high_low

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        pdf_box = matplotlib.backends.backend_pdf.PdfPages('plots/mi/overlap_full/box_plot_overlap_{}.pdf'.
                                                           format(retailer))

        plt.figure()

        bpl = plt.boxplot(data_LSC, positions=np.array(range(len(data_LSC))) * 2.0 - 0.5, sym='', widths=0.3)
        bpm = plt.boxplot(data_SPS, positions=np.array(range(len(data_SPS))) * 2.0, sym='', widths=0.3)
        bpr = plt.boxplot(data_Greedy, positions=np.array(range(len(data_Greedy))) * 2.0 + 0.5, sym='', widths=0.3)
        set_box_color(bpl, 'blue')  # colors are from http://colorbrewer2.org/
        set_box_color(bpm, 'red')
        set_box_color(bpr, 'green')
        formatter = ticker.FormatStrFormatter('$%1.0f')

        plt.axes().yaxis.set_major_formatter(formatter)
        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label='LSC')
        plt.plot([], c='red', label='SPS')
        plt.plot([], c='green', label='Greedy')
        plt.legend()
        plt.title('Retailer {}'.format(retailer))
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.tight_layout()

        #  plt.title()
        pdf_box.savefig()
        plt.close()
        pdf_box.close()

        for highlow in high_low:
            lift_inst = 0
            #  print('size', len(data_LSC[high_low_map[hl]]))
            for i in range(len(data_LSC[high_low_map[highlow]])):
                LSC = data_LSC[high_low_map[highlow]][i]
                SPS = data_SPS[high_low_map[highlow]][i]
                Greedy = data_Greedy[high_low_map[highlow]][i]
                lift_inst += round((LSC - SPS) * 100 / LSC, 5)

            lift_inst /= 100

            lift[retailer][highlow] = round(lift_inst, 3)

    print('Lift Matrix A', lift['A'])
    print('Lift Matrix B', lift['B'])
    print('Lift Matrix C', lift['C'])
    print('Lift Matrix D', lift['D'])
    print('Lift Matrix E', lift['E'])


#  box plot for multi items with data modified with high and low ship cost classes
def box_plots_shipcost_lowhigh():
    cost_high = read_data('summary_stats/costs_multi_item_high_shipcost.txt')
    cost_low = read_data('summary_stats/costs_multi_item_low_shipcost.txt')
    algorithms = ['LSC', '2-SPS']
    high_low = ['L', 'H']
    high_low_map = {'L': 0, 'H': 1}
    lift = {retailer: {highlow: 0 for highlow in high_low} for retailer in retailers}

    for retailer in retailers:
        data_SPS = [[] for _ in high_low]
        data_LSC = [[] for _ in high_low]
        data_Greedy = [[] for _ in high_low]

        for key, value in cost_high.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['H']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['H']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['H']].append(value)

        for key, value in cost_low.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['L']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['L']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['L']].append(value)

        #  print('data_sps', data_SPS)

        ticks = high_low

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        pdf_box = matplotlib.backends.backend_pdf.PdfPages('plots/mi/ship_cost_high_low/box_plot_shipcost_{}.pdf'.
                                                           format(retailer))

        plt.figure()

        bpl = plt.boxplot(data_LSC, positions=np.array(range(len(data_LSC))) * 2.0 - 0.5, sym='', widths=0.3)
        bpm = plt.boxplot(data_SPS, positions=np.array(range(len(data_SPS))) * 2.0, sym='', widths=0.3)
        bpr = plt.boxplot(data_Greedy, positions=np.array(range(len(data_Greedy))) * 2.0 + 0.5, sym='', widths=0.3)
        set_box_color(bpl, 'blue')  # colors are from http://colorbrewer2.org/
        set_box_color(bpm, 'red')
        set_box_color(bpr, 'green')
        formatter = ticker.FormatStrFormatter('$%1.0f')

        plt.axes().yaxis.set_major_formatter(formatter)
        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label='LSC')
        plt.plot([], c='red', label='SPS')
        plt.plot([], c='green', label='Greedy')
        plt.legend()
        plt.title('Retailer {}'.format(retailer))
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.tight_layout()

        #  plt.title()
        pdf_box.savefig()
        plt.close()
        pdf_box.close()

        for highlow in high_low:
            lift_inst = 0
            #  print('size', len(data_LSC[high_low_map[highlow]]))
            for i in range(len(data_LSC[high_low_map[highlow]])):
                LSC = data_LSC[high_low_map[highlow]][i]
                SPS = data_SPS[high_low_map[highlow]][i]
                Greedy = data_Greedy[high_low_map[highlow]][i]
                lift_inst += round((Greedy-SPS)*100/Greedy, 5)

            lift_inst /= 100

            lift[retailer][highlow] = round(lift_inst, 3)

    print('Lift Matrix A', lift['A'])
    print('Lift Matrix B', lift['B'])
    print('Lift Matrix C', lift['C'])
    print('Lift Matrix D', lift['D'])
    print('Lift Matrix E', lift['E'])


#  box plot for multi items with data modified with high and low pick failure prob classes
def box_plots_pickfailprob_lowhigh():
    cost_high = read_data('summary_stats/costs_multi_item_high_pickfailprob.txt')
    cost_low = read_data('summary_stats/costs_multi_item_low_pickfailprob.txt')
    algorithms = ['LSC', '2-SPS', 'Greedy']
    high_low = ['L', 'H']
    high_low_map = {'L': 0, 'H': 1}
    lift = {retailer: {highlow: 0 for highlow in high_low} for retailer in retailers}

    for retailer in retailers:
        data_SPS = [[] for _ in high_low]
        data_LSC = [[] for _ in high_low]
        data_Greedy = [[] for _ in high_low]

        for key, value in cost_high.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['H']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['H']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['H']].append(value)

        for key, value in cost_low.items():
            if key[0] == retailer:
                if key[2] == '2-SPS':
                    data_SPS[high_low_map['L']].append(value)
                elif key[2] == 'LSC':
                    data_LSC[high_low_map['L']].append(value)
                elif key[2] == 'Greedy':
                    data_Greedy[high_low_map['L']].append(value)

        #  print('data_sps', data_SPS)

        ticks = high_low

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        pdf_box = matplotlib.backends.backend_pdf.PdfPages(
            'plots/mi/pick_fail_prob_high_low/box_plot_pickfailprob_{}.pdf'.format(retailer))

        plt.figure()

        bpl = plt.boxplot(data_LSC, positions=np.array(range(len(data_LSC))) * 2.0 - 0.5, sym='', widths=0.3)
        bpm = plt.boxplot(data_SPS, positions=np.array(range(len(data_SPS))) * 2.0, sym='', widths=0.3)
        bpr = plt.boxplot(data_Greedy, positions=np.array(range(len(data_Greedy))) * 2.0 + 0.5, sym='', widths=0.3)
        set_box_color(bpl, 'blue')  # colors are from http://colorbrewer2.org/
        set_box_color(bpm, 'red')
        set_box_color(bpr, 'green')
        formatter = ticker.FormatStrFormatter('$%1.0f')

        plt.axes().yaxis.set_major_formatter(formatter)
        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label='LSC')
        plt.plot([], c='red', label='SPS')
        plt.plot([], c='green', label='Greedy')
        plt.legend()
        plt.title('Retailer {}'.format(retailer))
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.tight_layout()

        #  plt.title()
        pdf_box.savefig()
        plt.close()
        pdf_box.close()

        for highlow in high_low:
            lift_inst = 0
            #  print('size', len(data_LSC[high_low_map[highlow]]))
            for i in range(len(data_LSC[high_low_map[highlow]])):
                LSC = data_LSC[high_low_map[highlow]][i]
                SPS = data_SPS[high_low_map[highlow]][i]
                Greedy = data_Greedy[high_low_map[highlow]][i]
                lift_inst += round((Greedy-SPS)*100/Greedy, 5)

            lift_inst /= 100

            lift[retailer][highlow] = round(lift_inst, 3)

    print('Lift Matrix A', lift['A'])
    print('Lift Matrix B', lift['B'])
    print('Lift Matrix C', lift['C'])
    print('Lift Matrix D', lift['D'])
    print('Lift Matrix E', lift['E'])


if __name__ == "__main__":
    #  box_plot_single_item()
    #  box_plot_single_item_cancellation_probability()
    box_plots_multi_item()
    box_plots_full_overlap()
    box_plots_shipcost_lowhigh()
    box_plots_pickfailprob_lowhigh()
