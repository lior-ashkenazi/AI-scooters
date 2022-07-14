import pandas as pd
from typing import List

END_TIME = "end_time"
START_TIME = "start_time"
DEST_Y = "dest_y"
ORIG_Y = "orig_y"
DEST_X = "dest_x"
ORIG_X = "orig_x"


class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return str(self)


class Ride:
    def __init__(self, orig: Point, dest: Point, start_time: int, end_time: int):
        self.orig: Point = orig
        self.dest: Point = dest
        self.start_time: int = start_time
        self.end_time: int = end_time

    def __str__(self):
        return f"\norig: {self.orig}, dest: {self.dest}\n" \
               f"start time: {self.start_time}, end time: {self.end_time}\n"

    def __repr__(self):
        return str(self)


def rides_list_to_pd(rides: List[Ride]) -> pd.DataFrame:
    orig_x, orig_y, dest_x, dest_y, start_time, end_time = [], [], [], [], [], []
    for ride in rides:
        orig_x.append(ride.orig.x)
        orig_y.append(ride.orig.y)
        dest_x.append(ride.dest.x)
        dest_y.append(ride.dest.y)
        start_time.append(ride.start_time)
        end_time.append(ride.end_time)
    data = {ORIG_X: orig_x,
            ORIG_Y: orig_y,
            DEST_X: dest_x,
            DEST_Y: dest_y,
            START_TIME: start_time,
            END_TIME: end_time}
    return pd.DataFrame(data)


def pd_to_rides_list(rides_df: pd.DataFrame) -> List[Ride]:
    rides_list = []
    for index, row in rides_df.iterrows():
        rides_list.append(Ride(Point(row[ORIG_X], row[ORIG_Y]),
                               Point(row[DEST_X], row[DEST_Y]),
                               row[START_TIME],
                               row[END_TIME]))
    return rides_list


class NestAllocation:
    def __init__(self, location: Point, scooters_num: int):
        self.location: Point = location
        self.scooters_num: int = scooters_num

    def __str__(self):
        return f"location: {self.location}\t scooters' number: {self.scooters_num}"

    def __repr__(self):
        return str(self)


class Map:
    pass


if __name__ == '__main__':
    a: Ride = Ride(Point(2.3,4.5), Point(2,4), 1, 4)
    b: Ride = Ride(Point(4,6), Point(4,6), 5, 3)
    c: List[Ride] = [b, a]
    print(c)
    d = rides_list_to_pd(c)
    print(d)
    print(pd_to_rides_list(d))