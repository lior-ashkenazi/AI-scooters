import pandas as pd
import datetime
import numpy as np
import tkinter
from tkinter import *
from typing import List, Tuple
from programio.custom_gui import CustomTkinterMapView
import heapq
from data.trafficdatatypes import Ride, NestAllocation, Point

TEL_AVIV_CENTER_COORDS = (32.0853, 34.7818)
BLUE = "#3E69CB"
RED = "#CB3E69"
class Visualizer():
    def __init__(self, rides_list: List[List[Ride]], nests_list: List[List[NestAllocation]],
                 revenue_list: List[float],
                 frame_speed=200,
                 frames_per_day=24):
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

        # variables for GUI
        self.frame_speed = frame_speed
        self.frames_per_day = frames_per_day

        # lists of data per day
        self.revenue_list = revenue_list
        self.rides_list = rides_list
        self.nests_list = nests_list

        # updating data for display
        self.num_days = len(self.rides_list)
        self.cur_day = None
        self.cur_nests = list()
        self.cur_rides_in = list()
        self.cur_rides_out = list()

    def visualise(self):
        self.day_loop(day_index=0)
        self.root.mainloop()

    def init_tk(self):
        root = Tk()
        main_frame = Frame(root)
        main_frame.grid(row=0, column=0, sticky="nswe")

        # Init left side: Stats
        self.stats = {k: 0 for k in ['Completed Rides',
                                     'Non-Completed Rides',
                                     'Daily Revenue',
                                     'Day',
                                     'Hour',]}
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
            self.update_day_stats(day_index)
            self.update_day_rides(self.rides_list[day_index])
            self.update_day_nests(self.nests_list[day_index])
            # todo revenue

            self.frame_loop(frame_index=0)

            self.root.after(self.frame_speed * (self.frames_per_day + 1),
                            self.day_loop, day_index + 1)

    def frame_loop(self, frame_index):
        if frame_index >= self.frames_per_day:
            return
        cur_time = self.frame_to_cur_time(frame_index)
        self.update_frame_rides(cur_time)
        self._refresh_stats()
        self.stats['Hour'] += 1
        self.root.after(self.frame_speed,
                        self.frame_loop, frame_index + 1)

    def frame_to_cur_time(self, frame_index):
        return (datetime.datetime.min +
                datetime.timedelta(days=(float(frame_index + 1) / self.frames_per_day))).time()

    def update_day_nests(self, nests):
        # delete yesterday's nests
        for nest in self.cur_nests:
            self.map_widget.delete(nest)
        # create today's nests
        self.cur_nests = [self.map_widget.set_marker(nest.location.x, nest.location.y,
                                                     text=str(nest.scooters_num))
                          for nest in nests]

    def update_day_rides(self, rides):
        # delete remaining rides from yesterday
        for ride in self.cur_rides_out:
            self.map_widget.delete(ride)
        self.cur_rides_out = list()

        # order today's rides by start_time, as min heap
        self.cur_rides_in = [(ride.start_time, i, ride)
                             for i, ride in enumerate(rides)]
        heapq.heapify(self.cur_rides_in)

    def update_day_stats(self, day_index):
        self.stats['Day'] = day_index
        self.stats['Hour'] = 0
        self._refresh_stats()

    def update_frame_rides(self, cur_time):
        # add new rides
        while len(self.cur_rides_in) > 0 and self.cur_rides_in[0][0] <= cur_time:
            # todo add color?
            start_time, i, new_ride = heapq.heappop(self.cur_rides_in)
            ride_object = self.map_widget.set_path([new_ride.orig, new_ride.dest])
            heapq.heappush(self.cur_rides_out, (new_ride.end_time, i, ride_object))

        # remove old rides
        while len(self.cur_rides_out) > 0 and self.cur_rides_out[0][0] < cur_time:
            # todo add color?
            end_time, _, old_ride = heapq.heappop(self.cur_rides_out)
            self.map_widget.delete(old_ride)

    def update_stats(self, rides, revenue):
        completed_rides = rides['completed'].sum()
        self.stats['Completed Rides'] = completed_rides
        self.stats['Non-Completed Rides'] = (len(rides) - completed_rides)
        self.stats['Daily Revenue'] = revenue
        self._refresh_stats()

    def _refresh_stats(self):
        self.stats_label.config(text='\n'.join([f'{k}: {v}\n' for k, v in self.stats.items()]))


"""
test methods to generate random rides/nests 
"""
def random_r():
    r = list()
    for i in range(50):
        a = datetime.time(hour=np.random.randint(low=0, high=22),
                          minute=np.random.randint(low=0, high=59),
                          second=np.random.randint(low=0, high=59))
        b = datetime.time(hour=np.random.randint(low=0, high=22),
                          minute=np.random.randint(low=0, high=59),
                          second=np.random.randint(low=0, high=59))
        r.append(Ride(orig=TEL_AVIV_CENTER_COORDS + (np.random.random((2,)) - 0.5) / 50,
                      dest=TEL_AVIV_CENTER_COORDS + (np.random.random((2,)) - 0.5) / 50,
                      start_time=min(a, b),
                      end_time=max(a, b)))
    return r

def random_n():
    n = list()
    for i in range(5):
        location = TEL_AVIV_CENTER_COORDS + ((np.random.random((2,)) - 0.5) / 50)
        n.append(NestAllocation(location=Point(*location),
                                scooters_num=10))
    return n

if __name__ == '__main__':
    rides_list = [random_r() for r in range(10)]
    nest_list = [random_n() for n in range(10)]
    revenue_list = [np.random.randint(10, 100) for i in range(10)]
    a = Visualizer(rides_list, nest_list, revenue_list, frame_speed=200, frames_per_day=24)
    a.visualise()
