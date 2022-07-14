import pandas as pd
from typing import List, Optional
import numpy as np
from scipy import spatial


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

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])


def point_dist(a: Point, b: Point) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def point_from_numpy(arr: np.ndarray) -> Point:
    return Point(arr[0], arr[1])


def point_list_to_numpy(point_list: List[Point]) -> np.ndarray:
    arr = []
    for point in point_list:
        arr.append([point.x, point.y])
    return np.array(arr)


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

    def __lt__(self, other):
        return self.end_time < other.end_time


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
    def __init__(self, points: np.ndarray):
        self._points: np.ndarray = points

    def add_point(self, point: Point) -> None:
        """
        add point to the map
        :param point: new point to add
        :return: None
        """
        self._points = np.vstack([self._points, point.to_numpy()])

    def pop_nearest_point_in_radius(self, location: Point,
                                    radius: float) -> Optional[Point]:
        """
        given points search for the nearest neighbor, and if it is closer that
        the radius given, return the neighbor and remove it from the map
        :param location: location to search nearest point
        :param radius: distance to search nearest point
        :return: if there exists nearest point in the radius specified, return this
            point and pop it. Else, return None
        """
        if self._points.size == 0:
            return None

        distance, index = spatial.KDTree(self._points).query(location.to_numpy())
        if distance > radius:
            return None
        selected_point: Point = point_from_numpy(self._points[index])
        self._points = np.delete(self._points, index, axis=0)
        return selected_point


def optimal_transport(src: Map, dest: Map) -> float:
    # todo complete
    return 0

