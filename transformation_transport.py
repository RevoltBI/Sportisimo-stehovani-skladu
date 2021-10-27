import datetime

import pandas as pd
import numpy as np

OUT_TRANSPORT_FILE_PATH = "out/tables/solution_TRANSPORT.csv"

START_TRANSPORT_DATE = '26-10-2020'
TRANSPORT_DAYS_COUNT = 40
TRANSPORT_DAYS_BOOSTED_COUNT = 22
DAYS_OFF = [5]  # 5 = transport from Saturday to Sunday


def create_time_sequence(start_date, total_days_count, days_off):
    start_date = datetime.datetime.strptime(start_date, "%d-%m-%Y")

    time_sequence = []
    number_of_days = 0

    while len(time_sequence) < total_days_count + 1:
        new_day = start_date + datetime.timedelta(days=number_of_days)
        number_of_days += 1
        weekno = new_day.weekday()

        if weekno not in days_off:
            time_sequence.append(new_day)

    return time_sequence


def generate_trucks_transport_plan(start_date, total_days_count, boosted_days_count, days_off):
    transport_size = np.ones((total_days_count,), dtype=int)
    for day in range(0, boosted_days_count):
        transport_size[day] = 2
    time_sequence = create_time_sequence(start_date, total_days_count, days_off)

    truck_id = 0
    truck_output = []
    for day in range(0, total_days_count):
        number_of_trucks_in_day = transport_size[day]
        for _ in range(0, number_of_trucks_in_day):
            truck_output.append([truck_id + 1, time_sequence[day].strftime("%d.%m.%Y")])
            truck_id += 1

    df_truck_output = pd.DataFrame(truck_output, columns=['order_of_transport', 'date'])
    df_truck_output.to_csv(OUT_TRANSPORT_FILE_PATH, index=False)


if __name__ == '__main__':
    generate_trucks_transport_plan(START_TRANSPORT_DATE, TRANSPORT_DAYS_COUNT, TRANSPORT_DAYS_BOOSTED_COUNT, DAYS_OFF)
