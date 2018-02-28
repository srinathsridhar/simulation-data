import json
import itertools
import sys
from data_generator import *

data_to_sample = read_tsv(sys.argv[2])
tf_model = sys.argv[1]
sample_obj = Sample(sys.argv[3])

def calculate_expected_cost(cost_data, store_indices, big_penalty):
    cost = cost_data[store_indices[0]]['c'] + (1 - cost_data[store_indices[0]]['p']) * cost_data[store_indices[0]]['s']
    product = cost_data[store_indices[0]]['p']
    for l in range(len(store_indices)-1):
        cost += product * (cost_data[store_indices[l + 1]]['c'] + (1 - cost_data[store_indices[l + 1]]['p']) * cost_data[store_indices[l + 1]]['s'])
        product *= cost_data[store_indices[l + 1]]['p']
    cost += product * big_penalty
    return cost

def read_data(i):
    #file = open('Data_SI/Data(' + t + ')(' + str(i) + ').json', 'r')
    #stu = json.load(open('Data_SI/6'))
    #file.close()
    #stu = single_item(model, data_to_sample)
    stu = single_item_sampling(sys.argv[4], sys.argv[5], sample_obj)
    with open('Data_SI/{}_{}_noTF'.format(tf_model, i), 'w') as f:
        f.write(json.dumps(stu))
    print(stu)
    print('Iter:', i)
    return stu

#  solving LP to get OPT stores
def solve_lp_repeat(data, L, BigC):
    def con_rule3(model_3, l):
        return sum(model_3.p[j]*model_3.x[j, l] for j in model_3.J) == sum(model_3.x[j, l+1] for j in model_3.J)

    def obj_rule(model):
        return sum(model.Cost[j]*model.x[j, l] for j in model.J for l in model.L_con) + sum(model.BigCost[j]*model.x[j, L-1] for j in model.J)

    model = ConcreteModel()
    model.L = RangeSet(0, L-1)
    model.L_con = RangeSet(0, L-2)
    model.J = RangeSet(0, len(data) - 1)
    model.x = Var(model.J * model.L, domain=NonNegativeReals, bounds=(0, 1))
    model.s = Param(model.J, initialize={j: data[j]['s'] for j in model.J})
    model.p = Param(model.J, initialize={j: data[j]['p'] for j in model.J})
    model.c = Param(model.J, initialize={j: data[j]['c'] for j in model.J})
    model.Cost = Param(model.J, initialize={j: (1-data[j]['p'])*data[j]['s'] + data[j]['c'] for j in model.J})
    model.BigCost = Param(model.J, initialize={j: model.Cost[j] + model.p[j] * BigC for j in model.J})
    #model.BigCost = Param(model.J, initialize={j: model.Cost[j] for j in model.J})
    model.obj = Objective(rule=obj_rule, sense=minimize)
    model.con_1 = Constraint(expr=sum(model.x[j, 0] for j in model.J) == 1)
    model.con_3 = Constraint(model.L_con, rule=con_rule3)
   # model.pprint()
    opt = SolverFactory('cplex')
    time_lp_start = time.clock()

    opt.solve(model)
    time_lp_end = time.clock()
    time_lp = time_lp_end - time_lp_start
    # print([(j, model.w[j].value / (model.v[j] * model.w0.value)) for j in model.J if model.w[j].value > 0])
    # # print('Obj : ',model.obj())
    ss = [0] * L
    for j, l in model.J * model.L:
        if float(model.x[j, l].value) > 0:
            ss[l] = j
    print('LP Stores : ', ss)
    return model.obj(), time_lp


#  solving DP to get OPT stores
def solve_dp_repeat(data, L, BigC):
    J = len(data)
    Cost = [(1-store['p'])*store['s'] + store['c'] for store in data]
    BigCost = [Cost[j] + data[j]['p']*BigC for j in range(J)]
    time_start = time.clock()
    Value = [0] * L
    sol = [0]*L
    Value[L-1] = min(BigCost)
    sol[L-1] = BigCost.index(Value[L-1])
    for l in reversed(range(L-1)):
        Flow = [Cost[j] + data[j]['p'] * Value[l+1] for j in range(J)]
        Value[l] = min(Flow)
        sol[l] = Flow.index(Value[l])
    # print(Value)
    print('DP Stores : ', sol)
    time_end = time.clock()
    timer = time_end - time_start
    return Value[0]


# solving IP for no repeat (irrelevant now becuse we have a DP)
def solve_ip_norepeat(data, L, BigC):
    def con_rule3(model_3, l):
        return sum(model_3.p[j] * model_3.x[j, l] for j in model_3.J) == sum(model_3.x[j, l + 1] for j in model_3.J)

    def con_rule2(model_2, j):
        return sum(model_2.z[j, le] for le in model_2.L) <= 1

    def con_rule4(model_4, j, l):
        return model_4.x[j, l] <= model_4.z[j, l]

    def obj_rule(model):
        return sum(model.Cost[j] * model.x[j, l] for j in model.J for l in model.L_con) + sum(
            model.BigCost[j] * model.x[j, L - 1] for j in model.J)

    model = ConcreteModel()
    model.L = RangeSet(0, L - 1)
    model.L_con = RangeSet(0, L - 2)
    model.J = RangeSet(0, len(data) - 1)
    model.x = Var(model.J * model.L, domain=NonNegativeReals, bounds=(0, 1))
    model.z = Var(model.J * model.L, domain=Integers, bounds=(0, 1))
    model.s = Param(model.J, initialize={j: data[j]['s'] for j in model.J})
    model.p = Param(model.J, initialize={j: data[j]['p'] for j in model.J})
    model.c = Param(model.J, initialize={j: data[j]['c'] for j in model.J})
    model.Cost = Param(model.J, initialize={j: (1 - data[j]['p']) * data[j]['s'] + data[j]['c'] for j in model.J})
    model.BigCost = Param(model.J, initialize={j: model.Cost[j] + model.p[j] * BigC for j in model.J})
    model.obj = Objective(rule=obj_rule, sense=minimize)
    model.con_1 = Constraint(expr=sum(model.x[j, 0] for j in model.J) == 1)
    model.con_2 = Constraint(model.J, rule=con_rule2)
    model.con_3 = Constraint(model.L_con, rule=con_rule3)
    model.con_4 = Constraint(model.J*model.L, rule=con_rule4)
    # model.pprint()
    opt = SolverFactory('cplex')
    #opt.options['opt.emphasis.numerical'] = 1

    time_start = time.clock()
    opt.solve(model)
    time_end = time.clock()
    timed = time_end - time_start
    # print([(j, model.w[j].value / (model.v[j] * model.w0.value)) for j in model.J if model.w[j].value > 0])
    # # print('Obj : ',model.obj())
    x_sol = [0] * L

    for j, l in model.J * model.L:
       # print('z', model.z[j, l].value)
        if float(model.z[j, l].value) > 0.5:
            #print(l,j,model.z[j, l].value)
            x_sol[l] = j

    opt_eval = calculate_expected_cost(data, x_sol, BigC)


    print('IP OBJ            : ', model.obj(), 'Stores : ', x_sol)
    print('IP OBJ Calculated : ', opt_eval, 'Stores : ', x_sol)
    return model.obj(), timed


# Sort by pick cost, stores cannot be repeated
def solve_picking_norepeat(data, L, BigC):
    ss = sorted(range(len(data)), key=lambda k: (data[k]['c']))  # Store indices in ascending order of 'c'
    # print('ss_pick : ', ss)
    L_adjusted = min(len(data), L)
    val = calculate_expected_cost(data, ss[:L_adjusted], BigC)
    print('Picking Obj :', val, 'Stores : ', ss[:L_adjusted])
    return val


# Sort by Shipping Cost, stores cannot be repeted
def solve_shipping_norepeat(data, L, BigC):
    ss = sorted(range(len(data)), key=lambda k: (data[k]['s']))  # Store indices in ascending order of 's'
    # print('ss_ship : ', ss)
    L_adjusted = min(len(data), L)
    val = calculate_expected_cost(data, ss[:L_adjusted], BigC)
    print('Shipping Obj :', val, 'Stores : ', ss[:L_adjusted])
    return val


# Sort by Pick and Shipping Cost, Stores not repeated

def solve_pick_and_ship_norepeat(data, L, BigC):
    def ratio(m):
        return 100 * data[m]['c'] + data[m]['s']
    L_adjusted = min(len(data), L)
    ss = sorted(range(len(data)), key=lambda k: ratio(k))  # Store indices in ascending order of ratio
    val = calculate_expected_cost(data, ss[:L_adjusted], BigC)
    print('Ship & Pick Obj :', val, 'Stores : ', ss[:L_adjusted])
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

    L = 3   # # levels
    BigC = 25 # Penalty Cost if order is unfulfilled
    type=[]
    for string in map(''.join, itertools.product('LH', repeat=4)):
        #print(string)
        type.append(string)
    gap = []
    rep_gap = {ty: 0 for ty in type}
    for i in range(100):
        gap_pick = []
        gap_ship = []
        gap_pick_ship = []
        opt = []



        rep_gap_pick = []
        rep_gap_ship = []
        rep_gap_pick_ship = []
        stores = read_data(i)  # Argument: Number of items

        print('File : ', i)
        print stores

        dp_opt = solve_opt_dp_norepeat(stores, L, BigC)
        gap_pick.append(solve_picking_norepeat(stores, L, BigC))
        gap_ship.append(solve_shipping_norepeat(stores, L, BigC))
        gap_pick_ship.append(solve_pick_and_ship_norepeat(stores, L, BigC))
        opt.append(dp_opt)

        #lp_opt = solve_dp_repeat(stores, L, BigC)
        #rep_gap_pick.append(solve_picking_repeat(stores, L, BigC) / lp_opt - 1)
        #rep_gap_ship.append(solve_shipping_repeat(stores, L, BigC) / lp_opt - 1)
        #rep_gap_pick_ship.append(solve_pick_and_ship_repeat(stores, L, BigC) / lp_opt - 1)
        #rep_opt.append(lp_opt)

        #solve_ip_norepeat(stores, L, BigC)  # Argument: data, Time Periods
        #obj_dp, time_dp = solve_dp_Repeat(stores, L, BigC)  # Argument: data, Time Periods
        # obj_lp, time_lp = solve_lp_Repeat(stores, L, BigC)  # Argument: data, Time Periods
        print('\n')
        avg_opt = sum(opt)/len(opt)
        avg_pick = sum(gap_pick)/len(gap_pick)
        avg_ship = sum(gap_ship)/len(gap_ship)
        avg_pick_ship = sum(gap_pick_ship) / len(gap_pick_ship)
        gap.append([avg_opt, avg_pick, avg_ship, avg_pick_ship])

       # rep_avg_pick = sum(rep_gap_pick) / len(rep_gap_pick)
        #rep_avg_ship = sum(rep_gap_ship) / len(rep_gap_ship)
        #rep_avg_pick_ship = sum(rep_gap_pick_ship) / len(rep_gap_pick_ship)
       # rep_gap[t] = [rep_avg_pick, rep_avg_ship, rep_avg_pick_ship]
    f = open('Results_Single_Item/Costs_{}_noTF.txt'.format(tf_model),'w')

    for run_count,g in enumerate(gap):
        f.write('Run ' + str(run_count) + ': '+str(g)+'\n')
    for run_count,g in enumerate(gap):
        print(run_count, g)
    f.close()


if __name__ == "__main__":
    main()
