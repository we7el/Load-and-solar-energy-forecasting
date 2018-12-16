# Load initial data and upsamle
# analyze data

from pandas import read_csv
from numpy import shape
import numpy as np
from matplotlib import pyplot


def down_sample_data_30min(str_csv_source, str_csv_target):
    data = read_csv(str_csv_source, infer_datetime_format=True, low_memory=False, parse_dates={'datetime': [0]}, index_col=['datetime'])
    half_hourly_data = data.resample('30T').mean()
    half_hourly_data.to_csv(str_csv_target)
    half_hourly_data = np.array(half_hourly_data)
    return half_hourly_data

def plot_analyze_data():
    dataset = down_sample_data_30min('agg_load_123_july.csv', '30_minute_data.csv')
    for i in range(31):
        for j in range(95):
            pyplot.plot(dataset[i * 48: (i+1) * 48, j])
            pyplot.title("24-hour consumption sampled at 30 minutes over the month of July")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(31):
        pyplot.plot(dataset[i * 48: (i+1) * 48, 0])
        pyplot.title("All days one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(95):
        pyplot.plot(dataset[:48, i])
        pyplot.title("All houses one day")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(10):
        pyplot.subplot(10, 1, i+1)
        pyplot.plot(dataset[:48, i], "r")
        pyplot.title("Same day different houses")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(10):
        pyplot.subplot(10, 1, i+1)
        pyplot.plot(dataset[i * 48: (i+1) * 48, 0], "r")
        pyplot.title("Same house different days")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()


    # Our data starts on a Wednesday at 12AM
    wednesdays = np.vstack((dataset[:48], dataset[336:384], dataset[672:720], dataset[1008:1056], dataset[1344:1392]))
    thursdays = np.vstack((dataset[48:96], dataset[384:432], dataset[720:768], dataset[1056:1104], dataset[1392:1440]))
    fridays = np.vstack((dataset[96:144], dataset[432:480], dataset[768:816], dataset[1104:1152], dataset[1440:1488]))
    saturdays = np.vstack((dataset[144:192], dataset[480:528], dataset[816:864], dataset[1152:1200]))
    sundays = np.vstack((dataset[192:240], dataset[528:576], dataset[864:912], dataset[1200:1248]))
    mondays = np.vstack((dataset[240:288], dataset[576:624], dataset[912:960], dataset[1248:1296]))
    tuesdays = np.vstack((dataset[288:336], dataset[624:672], dataset[960:1008], dataset[1296:1344]))

    for i in range(4):
        pyplot.plot(mondays[i * 48: (i+1) * 48, 0])
        pyplot.plot(tuesdays[i * 48: (i+1) * 48, 0])
        pyplot.plot(wednesdays[i * 48: (i+1) * 48, 0])
        pyplot.plot(thursdays[i * 48: (i+1) * 48, 0])
        pyplot.plot(fridays[i * 48: (i+1) * 48, 0])
        pyplot.title("Power consumption during weekdays")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(4):
        pyplot.plot(saturdays[i * 48: (i+1) * 48, 0])
        pyplot.plot(sundays[i * 48: (i+1) * 48, 0])
        pyplot.title("Power consumption during weekends")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(5):
        pyplot.plot(wednesdays[i * 48: (i+1) * 48, 0])
        pyplot.title("All wednesdays one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(5):
        pyplot.plot(thursdays[i * 48: (i+1) * 48, 0])
        pyplot.title("All thursdays one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(5):
        pyplot.plot(fridays[i * 48: (i+1) * 48, 0])
        pyplot.title("All fridays one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(4):
        pyplot.plot(saturdays[i * 48: (i+1) * 48, 0])
        pyplot.title("All saturdays one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()

    for i in range(4):
        pyplot.plot(sundays[i * 48: (i+1) * 48, 0])
        pyplot.title("All sundays one house")
    pyplot.ylabel("kWh")
    pyplot.xlabel("Time of the day")
    pyplot.show()
    print(np.shape(wednesdays), "shape wednesdays")


if __name__ == '__main__':
    plot_analyze_data()
