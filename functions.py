import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import time
from tabulate import tabulate

def clustering_missmatch(result: pd.DataFrame,
                  sales_original: pd.DataFrame,
                  count_of_move: int):

    # +1 kvoli cislovaniu clustrov od 1
    clusters_matrix = np.zeros((count_of_move+1,count_of_move+1))
    sales_original.sort_values(by="SalesDate", inplace=True)
    #Orezane na 1000 objednavok pre skratenie casu algoritmu
    orders = sales_original["EcommOrderNumber"].unique()[:1000]
    result_products = result["Product_SK"].unique()

    #vytvori matici vzdalenosti clustru
    for order_number in orders:
        order_products = list(sales_original[sales_original["EcommOrderNumber"] == order_number]["Product_SK"].unique())
        for product1 in order_products:
            for product2 in order_products:
                if product1 != product2:
                    if product1 in result_products and product2 in result_products:
                        #result obsahuje stlce Product_SK a Solution
                        cluster1 = result.loc[result["Product_SK"] == product1].Solution.item()
                        cluster2 = result.loc[result["Product_SK"] == product2].Solution.item()
                        if cluster1 != cluster2:
                            clusters_matrix[cluster1][cluster2] += 1
                            clusters_matrix[cluster2][cluster1] += 1

    #prevedie maticu na list a zosrtuje podla poctu prepojeni clustrov
    matrix_to_list = []
    for i in range(1, count_of_move+1):
        for j in range(1+i, count_of_move+1):
                matrix_to_list.append({'from': i,
                                       'to': j,
                                       'relation': clusters_matrix[i][j]})
    matrix_to_list_sorted = sorted(list(filter(lambda x: x.get('relation') != 0, matrix_to_list)), key=lambda x: x.get('relation'), reverse=True)

    #list clustrov zoradeni v poradi od najviac prepojenych
    maping = []
    for element in matrix_to_list_sorted:
        if element.get('from') not in maping:
            maping.append(element.get('from'))
        if len(maping) == count_of_move:
            break
        if element.get('to') not in maping:
            maping.append(element.get('to'))
        if len(maping) == count_of_move:
            break

    #dictionary[povodny_cluster] = novy_cluster
    final_maping = {}
    for i in range(0, len(maping)):
        final_maping[i+1] = maping[i]

    result['New_Solution'] = result.apply(lambda row : final_maping[row['Solution']], axis = 1)
    del result['Solution']
    return result



def loss_function(volumes_original : pd.DataFrame,
                  sales_original : pd.DataFrame,
                  solution : np.array,
                  time_sequence : list,
                  one_move_size : int):

    volumes = volumes_original.copy()
    sales = sales_original.copy()

    start = time.time()
    volumes[["Stock"]] = 0   # bool -> 0 = nepřestěhované , 1 = přestěhované
    if len(volumes) == len(solution):
        volumes[["StepMove"]] = solution
    else:
        print("Error - solution has a different dimension than volumes \nsolution must be a 1xN vector")
        return 1e100, []

    if max(volumes[["StepMove","Volumes"]].groupby(["StepMove"]).sum().Volumes.values) > one_move_size*1.1:
        print("Warning - inadmissible solution - the volume of goods does not fit in the truck")


    LOG = list()
    loss = 0
    for i in range(1,len(time_sequence)):

        volumes.loc[volumes["StepMove"] == i, "Stock"] = 1  # přesune zboží ze skaldu 1 do skaldu 2 (zboží které se má přesouvat ve fázi 1)
        sales_subset = sales[(sales.SalesDate >= time_sequence[i-1]) &
                             (sales.SalesDate < time_sequence[i])][["Product_SK","EcommOrderNumber"]].copy() # subset objednávek na dané období

        loss_function_value = _loss_function_main(sales_subset, volumes)

        total_subset_orders_count = len(list(set(sales_subset.EcommOrderNumber)))
        LOG.append([time_sequence[i-1],loss_function_value,total_subset_orders_count]) # udává přírůstky loss function a total orders v jednotlivých dnech
        loss += loss_function_value

    print("loss function count time:",round(time.time()-start,2),"s" )
    return loss, np.array(LOG)


def create_time_sequence(time = '15-10-2020',count_of_move = 30):
    start_date = datetime.datetime.strptime(time,"%d-%m-%Y")
    time_sequence = [start_date + datetime.timedelta(days = 2*i) for i in range(count_of_move+1)]
    return time_sequence


def loss_function_2(volumes_original : pd.DataFrame,
                    sales_original : pd.DataFrame,
                    solution : pd.DataFrame,     #  potřebuje sloupec Product_SK, a clusterovací sloupec
                    time_sequence : list,
                    one_move_size : int):

    start = time.time()

    volumes = volumes_original.copy()
    sales = sales_original.copy()

    try:
        solution.columns = ["StepMove"]
    except:
        try:
            solution = solution.set_index("Product_SK")
            solution.columns = ["StepMove"]
        except:
            print("solution have to have 2 columns, Product_SK and clustering columns")

    volumes = volumes.set_index("Product_SK").join(solution).reset_index()[["Product_SK","StepMove","Volumes"]]
    volumes[["Stock"]] = 0   # bool -> 0 = nepřestěhované , 1 = přestěhované

    if len(volumes[volumes.StepMove.isna()]) > 0:
        print("There are missing product in solution!!!")
        print("loss function count time:",round(time.time()-start,2),"s" )
        #return 1e40-1, []

    if max(volumes[["StepMove","Volumes"]].groupby(["StepMove"]).sum().Volumes.values) > one_move_size*1.1:
        print("Warning - inadmissible solution - the volume of goods does not fit in the truck")


    LOG = list()
    loss = 0
    for i in range(1,len(time_sequence)):

        volumes.loc[volumes["StepMove"] == i, "Stock"] = 1  # přesune zboží ze skaldu 1 do skaldu 2 (zboží které se má přesouvat ve fázi 1)
        sales_subset = sales[(sales.SalesDate >= time_sequence[i-1]) &
                             (sales.SalesDate < time_sequence[i])][["Product_SK","EcommOrderNumber"]].copy() # subset objednávek na dané období

        loss_function_value = _loss_function_main(sales_subset, volumes)

        total_subset_orders_count = len(list(set(sales_subset.EcommOrderNumber)))
        LOG.append([time_sequence[i-1],loss_function_value,total_subset_orders_count]) # udává přírůstky loss function a total orders v jednotlivých dnech
        loss += loss_function_value

    print("loss function count time:",round(time.time()-start,2),"s" )
    return loss, np.array(LOG)

def _loss_function_main(sales_subset : pd.DataFrame, volumes : pd.DataFrame) -> int:
    """a function used only within a loss function"""

    # spojí tabulky sales_subset a volumes
    # podívá se na objednávky které se musely expedovat ze dvou míst
    # tedy pro jedno EcommOrderNumber byly 2 různé hodnoty Stock
    # Stock : bool -> 0 / 1
    table_lf = sales_subset.set_index("Product_SK").join(volumes.set_index("Product_SK")).reset_index()[["Product_SK",
                                                                                                     "EcommOrderNumber",
                                                                                                     "Stock"]]
    table_lf = table_lf[table_lf.Stock.notna()][["EcommOrderNumber",
                                                 "Stock"]].drop_duplicates().groupby(["EcommOrderNumber"]).count().reset_index()
    return len(table_lf[table_lf.Stock == 2])


def generate_random_solution(volumes_original, count_of_move):
    # vygeneruje náhodné řešení
    random_solution = np.random.randint(1,count_of_move + 1,len(volumes_original))
    result = volumes_original[["Product_SK"]].copy()
    result["result"] = random_solution
    print("Solution was successfully generated (random)")
    return result

def count_one_move_size(count_of_move, volumes_original = volumes_original):
    entire_stock_volumes = sum(volumes_original.Volumes)
    one_move_size = math.ceil( entire_stock_volumes / (count_of_move))
    return one_move_size

def generate_category_based_solution(volumes_original, count_of_move):
    # vygeneruje řešení na základě kategorii a brandu

    one_move_size = count_one_move_size(count_of_move)

    volumes_subset = volumes_original[["CategoryName","Brand","Volumes"]].groupby(by = ["CategoryName","Brand"]).sum(["Volumes"]).reset_index()
    volumes_subset["solution_j"] = 0
    conter_volumes = 0

    j = 1
    for i in range(len(volumes_subset)):
        volumes_subset.loc[i,"solution_j"] = j       # zapíše "j" ty cluster na řadek
        conter_volumes += volumes_subset.iloc[i].Volumes # přičte velikost (Volume) kategorie
        if conter_volumes >= one_move_size * 0.98:    # při naplnění clusteru zvedne index "j" a vynuluje counter
            j += 1
            conter_volumes = 0

    volumes_subset = volumes_subset.set_index(["CategoryName","Brand"])      # left join přes CategoryName a Brand
    help_volumes_solution = volumes_original.set_index(["CategoryName","Brand"]).join(volumes_subset,
                                                                                      rsuffix = '_').reset_index()[["Product_SK",
                                                                                                                    "solution_j"]]
    cat_based_solution = volumes_original.set_index("Product_SK").join(help_volumes_solution.set_index("Product_SK"))[["solution_j"]]
    cat_based_solution.columns = ["result"]
    print("Solution was successfully generated (category_based)")
    return cat_based_solution


def do_experiment_category_and_random(volumes_original,sales_original,start_date,count_of_move):
    # vygeneruje časové skoky kdy se sklad přesouvá
    # start_date = '01-10-2020'

    one_move_size = count_one_move_size(count_of_move)

    start_date = datetime.datetime.strptime(start_date,"%d-%m-%Y")
    time_sequence = [start_date + datetime.timedelta(days = 2*i) for i in range(count_of_move+1)]

    # Vytvoří řešení
    random_solution    = generate_random_solution(volumes_original)
    cat_based_solution = generate_category_based_solution(volumes_original)

    # Spočítá loss function
    loss_random,    LOG_random      = loss_function(volumes_original, sales_original, random_solution,    time_sequence, one_move_size)
    loss_cat_based, LOG_cat_based = loss_function(volumes_original, sales_original, cat_based_solution, time_sequence, one_move_size)

    plt.figure(figsize = [10,6])
    plt.plot(LOG_random.T[0],LOG_random.T[2],       color = 'gray',  label = 'total orders', alpha = 0.5)
    plt.plot(LOG_random.T[0],LOG_random.T[1],       color = 'red',   label = '{} - random'.format(loss_random))
    plt.plot(LOG_cat_based.T[0],LOG_cat_based.T[1], color = 'green', label = '{} - category based asc'.format(loss_cat_based))
    plt.grid(True)
    plt.legend()
    plt.title("Loss function")
    plt.show()


    vol = volumes_original[["Volumes"]].copy()
    vol["random_solution"] = random_solution
    vol["category_based_solution"] = cat_based_solution

    B = vol[["Volumes","random_solution"]].groupby("random_solution").sum("Volumes")
    C = vol[["Volumes","category_based_solution"]].groupby("category_based_solution").sum("Volumes")


    plt.figure(figsize = [10,3])
    plt.bar(B.index,B.Volumes.values, color = 'red',   alpha = 0.5, label = "random")
    plt.bar(C.index,C.Volumes.values, color = 'green', alpha = 0.5, label = "category based asc")
    plt.legend()
    plt.show()

    orders_count = sum(LOG_random.T[2])
    print("\n")
    print(tabulate([['Random', loss_random, str(round(loss_random/orders_count * 100,2))+"%"],
                    ['Category based', loss_cat_based, str(round(loss_cat_based/orders_count * 100,2))+"%"]]
                   , headers=['Method', 'Loss function', 'percentage'], tablefmt='orgtbl'))


def compare_two_models(solution_1,solution_2,start_date,count_of_move,
                       volumes_original = volumes_original,
                       sales_original = sales_original):
    # vygeneruje časové skoky kdy se sklad přesouvá
    # start_date = '01-10-2020'
    one_move_size = count_one_move_size(count_of_move)

    start_date = datetime.datetime.strptime(start_date,"%d-%m-%Y")
    time_sequence = [start_date + datetime.timedelta(days = 2*i) for i in range(count_of_move+1)]

    entire_stock_volumes = sum(volumes_original.Volumes)
    one_move_size = math.ceil( entire_stock_volumes / count_of_move )

    # Spočítá loss function
    loss_1,    LOG_1      = loss_function(volumes_original, sales_original, solution_1,    time_sequence, one_move_size)
    loss_2,    LOG_2      = loss_function(volumes_original, sales_original, solution_2,    time_sequence, one_move_size)

    plt.figure(figsize = [10,6])
    plt.plot(LOG_1.T[0],LOG_1.T[2], color = 'gray',  label = 'total orders', alpha = 0.5)
    plt.plot(LOG_1.T[0],LOG_1.T[1], color = 'red',   label = '{} - model 1'.format(loss_1))
    plt.plot(LOG_2.T[0],LOG_2.T[1], color = 'green', label = '{} - model 2'.format(loss_2))
    plt.grid(True)
    plt.legend()
    plt.title("Loss function")
    plt.show()


    vol = volumes_original[["Volumes"]].copy()
    vol["solution_1"] = solution_1
    vol["solution_2"] = solution_2

    B = vol[["Volumes","solution_1"]].groupby("solution_1").sum("Volumes")
    C = vol[["Volumes","solution_2"]].groupby("solution_2").sum("Volumes")


    plt.figure(figsize = [10,3])
    plt.bar(B.index,B.Volumes.values, color = 'red',   alpha = 0.5, label = "model 1")
    plt.bar(C.index,C.Volumes.values, color = 'green', alpha = 0.5, label = "model 2")
    plt.legend()
    plt.title("Cluster capacity utilization")
    plt.grid(True)
    plt.show()

    orders_count = sum(LOG_1.T[2])
    print("\n")
    print(tabulate([['Model 1', loss_1, str(round(loss_1/orders_count * 100,2))+"%"],
                    ['Model 2', loss_2, str(round(loss_2/orders_count * 100,2))+"%"],
                    [''       , 'improvement'    , str(round((1-loss_2/loss_1) * 100,2))+"%"]]
                   , headers=['Method', 'Loss function', 'percentage'], tablefmt='orgtbl'))






def compare_two_models_2(solution_1 : list, solution_2 : pd.DataFrame,
                       start_date,count_of_move,
                       volumes_original = volumes_original,
                       sales_original = sales_original):
    # vygeneruje časové skoky kdy se sklad přesouvá
    # start_date = '01-10-2020'
    one_move_size = count_one_move_size(count_of_move)

    start_date = datetime.datetime.strptime(start_date,"%d-%m-%Y")
    time_sequence = [start_date + datetime.timedelta(days = 2*i) for i in range(count_of_move+1)]

    entire_stock_volumes = sum(volumes_original.Volumes)
    one_move_size = math.ceil( entire_stock_volumes / count_of_move )

    # Spočítá loss function
    loss_1,    LOG_1      = loss_function_2(volumes_original, sales_original, solution_1,    time_sequence, one_move_size)
    loss_2,    LOG_2      = loss_function_2(volumes_original, sales_original, solution_2,    time_sequence, one_move_size)

    plt.figure(figsize = [10,6])
    plt.plot(LOG_1.T[0],LOG_1.T[2], color = 'gray',  label = 'total orders', alpha = 0.5)
    plt.plot(LOG_1.T[0],LOG_1.T[1], color = 'red',   label = '{} - model 1'.format(loss_1))
    plt.plot(LOG_2.T[0],LOG_2.T[1], color = 'green', label = '{} - model 2'.format(loss_2))
    plt.grid(True)
    plt.legend()
    plt.title("Loss function")
    plt.show()


    vol = volumes_original[["Product_SK","Volumes"]].copy()

    try:
        solution_1 = solution_1.set_index("Product_SK")
        solution_1.columns = ["solution_1"]
    except:
        solution_1.columns = ["solution_1"]
    try:
        solution_2 = solution_2.set_index("Product_SK")
        solution_2.columns = ["solution_2"]
    except:
        solution_2.columns = ["solution_2"]
    vol = vol.set_index("Product_SK").join(solution_1)
    vol = vol.join(solution_2)

    B = vol[["Volumes","solution_1"]].groupby("solution_1").sum("Volumes")
    C = vol[["Volumes","solution_2"]].groupby("solution_2").sum("Volumes")


    plt.figure(figsize = [10,3])
    plt.bar(B.index,B.Volumes.values, color = 'red',   alpha = 0.5, label = "model 1")
    plt.bar(C.index,C.Volumes.values, color = 'green', alpha = 0.5, label = "model 2")
    plt.legend()
    plt.title("Cluster capacity utilization")
    plt.grid(True)
    plt.show()

    orders_count = sum(LOG_1.T[2])
    print("\n")
    print(tabulate([['Model 1', loss_1, str(round(loss_1/orders_count * 100,2))+"%"],
                    ['Model 2', loss_2, str(round(loss_2/orders_count * 100,2))+"%"],
                    [''       , 'improvement'    , str(round((1-loss_2/loss_1) * 100,2))+"%"]]
                   , headers=['Method', 'Loss function', 'percentage'], tablefmt='orgtbl'))
