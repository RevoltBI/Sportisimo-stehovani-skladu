from math import ceil

import pandas as pd

IN_SALES_FILE_PATH = "in/tables/DynamicalClustering_sales_full.csv"
IN_VOLUMES_FILE_PATH = "in/tables/DynamicalClustering_volumes_past.csv"
OUT_SOLUTION_FILE_PATH = "out/tables/solution.csv"

IN_SALES_LEARNING_FROM_DATE = '2020-01-07'
IN_SALES_LEARNING_TO_DATE = '2020-10-25'

MAX_CLUSTER_SIZE = 330


def read_input(sales_file_path, volumes_file_path):
    sales_original = pd.read_csv(sales_file_path)
    sales_original.SalesDate = pd.to_datetime(sales_original.SalesDate)
    sales_original = sales_original.loc[
        (sales_original.SalesDate >= IN_SALES_LEARNING_FROM_DATE) & (
                    sales_original.SalesDate < IN_SALES_LEARNING_TO_DATE)]

    volumes_original = pd.read_csv(volumes_file_path)
    volumes_original['Cartons'] = volumes_original['Percent_fullness'] * 1.1  # Added 10% as reserve
    return sales_original, volumes_original


def generate_common_ordered_products(sales):
    orders = sales["EcommOrderNumber"].unique()
    common_ordered_products = {}

    for order_number in orders:
        order_products = list(filter(lambda product: product in product_volume.keys(),
                                     list(sales[sales["EcommOrderNumber"] == order_number]["Product_ID"].unique())))
        for order_product in order_products:
            if order_product not in common_ordered_products.keys():
                common_ordered_products[order_product] = set()
            for product in order_products:
                if product != order_product:
                    common_ordered_products[order_product].add(product)
    return common_ordered_products


def compute_volume_of_common_products(main_product, products):
    return product_volume[main_product] + sum(list(map(lambda product: product_volume[product], products)))


def pick_products_to_cluster(products, size):
    cluster_volume = 0
    picked_products = []
    picked_products_all = []

    while size - cluster_volume > 0 and products:
        picked_product = max(products, key=lambda product: product.get('Volume'))
        if picked_product.get('Volume') > size - cluster_volume:
            products_bin_volume = 0
            products_bin = []

            products_bin.append(picked_product.get('Product_ID'))
            products_bin_volume += product_volume[picked_product.get('Product_ID')]
            sorted_common_products_by_volume = sorted(list(picked_product.get('Common_products')),
                                                      key=lambda product: product_volume[product], reverse=True)

            while size - cluster_volume - products_bin_volume > 0:
                product = sorted_common_products_by_volume.pop(0)
                products_bin.append(product)
                products_bin_volume += product_volume[product]

            cluster_volume += products_bin_volume
            picked_products_all.extend(products_bin)

        if picked_product.get('Volume') <= size - cluster_volume:
            cluster_volume += picked_product.get('Volume')
            picked_products.append(picked_product)
            picked_products_all.append(picked_product.get('Product_ID'))
            for product in picked_product.get('Common_products'):
                picked_products_all.append(product)

            for x in products:
                if picked_product.get('Product_ID') in x.get('Common_products'):
                    x['Common_products'].remove(picked_product.get('Product_ID'))
                    x['Volume'] = compute_volume_of_common_products(x.get('Product_ID'), x.get('Common_products'))
                    if x['Volume'] == 0:
                        products.remove(x)
            products.remove(picked_product)
            for product in picked_product.get('Common_products'):
                if product in list(map(lambda x: x.get('Product_ID'), products)):
                    x = list(filter(lambda y: y.get('Product_ID') == product, products))[0]
                    products.remove(x)
            for x in products:
                for picked_common_product in picked_product.get('Common_products'):
                    if picked_common_product == x.get('Product_ID'):
                        products.remove(x)
                    if picked_common_product in x.get('Common_products'):
                        x['Common_products'].remove(picked_common_product)
                        x['Volume'] = compute_volume_of_common_products(x.get('Product_ID'), x.get('Common_products'))
                        if x['Volume'] == 0:
                            products.remove(x)

    return picked_products_all


def get_picked_all_products(products):
    products_all = []
    for product in products:
        products_all.append(product.get('Product_ID'))
        products_all.extend(product.get('Common_products'))
    return list(set(products_all))


def refresh_common_ordered_products(common_ordered_products, picked_all):
    for product in picked_all:
        if product in common_ordered_products.keys():
            del common_ordered_products[product]
    for key in common_ordered_products.keys():
        for product in picked_all:
            if product in common_ordered_products[key]:
                common_ordered_products[key].remove(product)

    return common_ordered_products


def put_products_from_orders_to_clusters(result, cluster_checksum, common_ordered_products):
    cluster_num = 1

    while True:
        common_order_products_with_volume = list(map(lambda product: {'Product_ID': product,
                                                                      'Common_products': common_ordered_products[
                                                                          product],
                                                                      'Volume': compute_volume_of_common_products(
                                                                          product, common_ordered_products[product])},
                                                     common_ordered_products.keys()))
        common_order_products_with_volume = list(
            filter(lambda product: product.get('Volume') > 0, common_order_products_with_volume))

        picked = pick_products_to_cluster(common_order_products_with_volume, MAX_CLUSTER_SIZE)

        check_sum = 0
        for product in picked:
            check_sum += product_volume[product]
        cluster_checksum[cluster_num] = check_sum

        picked_all = set(picked)

        result.extend(list(map(lambda product: {'Product_ID': product, 'Cluster': cluster_num}, picked_all)))
        cluster_num += 1

        common_ordered_products = refresh_common_ordered_products(common_ordered_products, picked_all)
        if not common_ordered_products:
            print('Terminating')
            break
    return result, cluster_checksum


def put_products_from_warehouse_to_clusters(result, cluster_checksum, volumes_original):
    clusters_with_space = list(map(lambda cluster: {'Cluster': cluster,
                                                    'Size': cluster_checksum[cluster]},
                                   filter(lambda cluster: cluster_checksum[cluster] < MAX_CLUSTER_SIZE,
                                          cluster_checksum.keys())))

    products_in_clusters = list(map(lambda product: product.get('Product_ID'), result))
    products_left_in_warehouse = volumes_original[~volumes_original["Product_ID"].isin(products_in_clusters)]

    for _, row in products_left_in_warehouse.iterrows():
        for clusters in clusters_with_space:
            if row['Cartons'] + clusters.get('Size') <= MAX_CLUSTER_SIZE:
                result.append({'Product_ID': row['Product_ID'], 'Cluster': clusters.get('Cluster')})
                clusters['Size'] += row['Cartons']
                break
    return result


def generate_solution(result, file_path):
    df_results = pd.DataFrame(result)
    df_results.to_csv(file_path, index=False)


if __name__ == '__main__':
    sales, volumes = read_input(IN_SALES_FILE_PATH, IN_VOLUMES_FILE_PATH)
    product_volume = dict(volumes[["Product_ID", "Cartons"]].values)

    clusters_total_number = ceil(sum(volumes.Cartons) / MAX_CLUSTER_SIZE)

    print(f'Total number of clusters: {clusters_total_number}')

    print('Start building common_ordered_products')
    common_ordered_products = generate_common_ordered_products(sales)

    result = []
    cluster_size = {}
    for cluster in range(1, clusters_total_number + 1):
        cluster_size[cluster] = 0

    print('Start filling clusters with products from orders')
    result, cluster_size = put_products_from_orders_to_clusters(result, cluster_size, common_ordered_products)
    print('Start filling clusters with left products in warehouse')
    result = put_products_from_warehouse_to_clusters(result, cluster_size, volumes)

    generate_solution(result, OUT_SOLUTION_FILE_PATH)
