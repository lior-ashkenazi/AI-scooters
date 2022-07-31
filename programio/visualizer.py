import pandas as pd
import datetime
import numpy as np
import tkinter
from tkinter import *
from typing import List
from custom_gui import CustomTkinterMapView

TEL_AVIV_CENTER_COORDS = (32.0853, 34.7818)
TIME_PER_HOUR = 300
BLUE = "#3E69CB"
RED = "#CB3E69"
class Visualizer():
    def __init__(self, rides_list: List[pd.DataFrame], nests_list:List[pd.DataFrame],
                 revenue_list: List[int]):
        """
        rides_list: List of pd.DataFrames. Each index i corresponds to rides
        of day number i. Must have following columns-
        starttime, endtime, dest, origin, completed
        where "completed" is a boolean whether the ride was carried out.
        nests_list: List of pd.DataFrames. Each index i corresponds to nests
        of day number i.
        Must have following columns-
        location, num_scooters.
        revenue_list:
        """
        # init tkinter GUI
        self.root = self.init_tk()

        # lists of data per day
        self.revenue_list = revenue_list
        self.rides_list = rides_list
        self.nests_list = nests_list

        # updating data for display
        self.num_days = len(self.rides_list)
        self.cur_nests = list()
        self.cur_rides = list()

        # start the visualization
        self.day_loop(day_index=0)
        self.root.mainloop()

    def init_tk(self):
        root = Tk()
        main_frame = Frame(root)
        main_frame.grid(row=0, column=0, sticky="nswe")

        # Init left side: Stats
        self.stats = {k: 0 for k in ['Completed Rides',
                                     'Non-Completed Rides',
                                     'Daily Revenue']}
        left_frame = Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nswe")
        self.stats_label = Label(left_frame,
                                 text='\n'.join([f'{k}: {v}\n' for k, v in self.stats.items()]),
                                 font="Helvetica 16", justify=tkinter.CENTER)
        self.stats_label.grid(row=0, column=0, sticky=tkinter.NSEW)

        # Init right side: Map
        right_frame = Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nswe")
        self.map_widget = CustomTkinterMapView(right_frame, width=600, height=400, corner_radius=0)
        self.map_widget.set_position(*TEL_AVIV_CENTER_COORDS)
        self.map_widget.set_zoom(14)
        self.map_widget.pack()
        return root

    def day_loop(self, day_index):
        if day_index >= self.num_days:
            # todo what to do when all days are done
            self.root.quit()
        else:
            # todo generate procedurally
            rides, nests, revenue = self.rides_list[day_index], self.nests_list[day_index], self.revenue_list[day_index]
            self.update_nests(nests=nests)
            rides_by_hour = self.get_rides_by_hour(rides)
            self.hour_loop(rides=rides_by_hour, hour=0)
            self.update_stats(rides, revenue)
            self.root.after(TIME_PER_HOUR * 25, self.day_loop, day_index + 1)

    def hour_loop(self, rides, hour):
        if hour >= 24:
            return
        rides_for_hour = rides[hour][1]
        self.update_rides(rides_for_hour)
        self.root.after(TIME_PER_HOUR, self.hour_loop, rides, hour + 1)

    def get_rides_by_hour(self, rides):
        rides['hour'] = rides.starttime.apply(lambda x: x.hour)
        rides['color'] = np.where(rides['completed'], BLUE, RED)
        return list(rides.groupby('hour'))

    def update_nests(self, nests):
        for nest in self.cur_nests:
            self.map_widget.delete(nest)
        self.cur_nests = nests.apply(lambda x: self.map_widget.set_marker(*(x.location), text=x.num_scooters),axis=1)

    def update_rides(self, rides):
        for ride in self.cur_rides:
            self.map_widget.delete(ride)
        self.cur_rides = rides.apply(lambda x: self.map_widget.set_path([x.orig, x.dest], color=x.color),axis=1)

    def update_stats(self, rides, revenue):
        completed_rides = rides['completed'].sum()
        self.stats['Completed Rides'] = completed_rides
        self.stats['Non-Completed Rides'] = (len(rides) - completed_rides)
        self.stats['Daily Revenue'] = revenue
        self.stats_label.config(text='\n'.join([f'{k}: {v}\n' for k, v in self.stats.items()]))

"""
test methods to generate random rides/nests 
"""
def random_r():
    r = list()
    c = datetime.datetime.now()
    s = (32.0853, 34.7818)
    for i in range(100):
        c = c + datetime.timedelta(minutes=30)
        r.append([c, c ,
                  s + (np.random.random((2,)) - 0.5) / 50,
                  s + (np.random.random((2,)) - 0.5) / 50,
                 np.random.random() < 0.5])
    r = pd.DataFrame(r, columns =['starttime', 'endtime', 'orig', 'dest', 'completed'])
    return r

def random_n():
    s = (32.0853, 34.7818)
    n = list()
    for i in range(5):
        n.append([s + (np.random.random((2,)) - 0.5) / 50, 10])
    n = pd.DataFrame(n, columns =['location', 'num_scooters'])
    return n

if __name__ == '__main__':
    rides_list = [random_r() for r in range(10)]
    nest_list = [random_n() for n in range(10)]
    revenue_list = [np.random.randint(10, 100) for i in range(10)]
    a = Visualizer(rides_list, nest_list, revenue_list)
