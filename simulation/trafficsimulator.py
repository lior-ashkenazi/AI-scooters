from typing import Tuple
from data.trafficdatatypes import *
from queue import PriorityQueue
import datetime
from data.trafficgenerator import TrafficGenerator


class TrafficSimulator:

    def __init__(self, rides_per_day_part, search_radius: int, const_rides: bool, data_type):
        self._rides_per_day_part: int = rides_per_day_part
        self._traffic_generator: TrafficGenerator = TrafficGenerator(None)  # todo: check what to enter here
        self._search_radius: int = search_radius
        self._const_rides = const_rides
        self.data_type = data_type
        if self._const_rides:
            self._potential_rides = self._traffic_generator.get_custom_data(self._rides_per_day_part, mode=self.data_type)
        else:
            self._potential_rides = None

    def get_simulation_result(self, scooters_initial: Map, option_idx=0) -> \
            Tuple[List[Ride], Map]:
        """
        :param scooters_initial_locations: scooters' initial location
        :return:
            - list of rides that were completed
            - map of final locations of scooters
        """
        scooters_initial_locations = Map(scooters_initial.get_points().copy())
        # initialize datastructures
        if self._const_rides:
            potential_rides: List[Ride] = self._potential_rides
        else:
            potential_rides: List[Ride] = self._traffic_generator.get_custom_data(self._rides_per_day_part,
                                                                                  option_idx=option_idx,
                                                                                  search_radius=self._search_radius,
                                                                                  mode=self.data_type)
        potential_rides.sort(key=lambda r: r.start_time)
        available_scooters: Map = scooters_initial_locations
        starting_scooters_lst = []
        unavailable_scooters: PriorityQueue = PriorityQueue()
        rides_performed: List[Ride] = []
        used_scooters = []

        for ride in potential_rides:
            # return scooters that finish ride to available scooters
            cur_time: datetime.time = ride.start_time
            TrafficSimulator.finish_rides(available_scooters, unavailable_scooters,
                                          cur_time)

            # if ride can be performed, update data structures accordingly
            starting_scooters_lst.append(ride.orig.to_numpy())
            is_feasible, nearest_scooter = self.ride_is_feasible(ride, available_scooters)
            if is_feasible:
                unavailable_scooters.put(ride)
                rides_performed.append(ride)
                used_scooters.append(nearest_scooter)

        # at the end of the simulation, get all the scooters back to the map
        while unavailable_scooters.qsize() > 0:
            ride_done: Ride = unavailable_scooters.get()
            available_scooters.add_point(ride_done.dest)
        starting_scooters = Map(np.array(starting_scooters_lst))
        return rides_performed, available_scooters, starting_scooters, used_scooters

    def ride_is_feasible(self, ride: Ride, available_scooters: Map) -> bool:
        nearest_point: Optional[Point] = available_scooters.\
            pop_nearest_point_in_radius(ride.orig, self._search_radius)
        if nearest_point is None:
            return False, None
        return True, nearest_point

    @staticmethod
    def finish_rides(available_scooters: Map, unavailable_scooters: PriorityQueue,
                     cur_time: datetime.time) -> None:
        while unavailable_scooters.qsize() > 0:
            next_ride: Ride = unavailable_scooters.get()
            # if end time is smaller than current time, scooter is available again
            if next_ride.end_time <= cur_time:
                available_scooters.add_point(next_ride.dest)
            # else - scooter is not available yet. put it back and return.
            else:
                unavailable_scooters.put(next_ride)
                return

    @staticmethod
    def get_scooters_location_from_nests_spread(
            nests_spread: List[NestAllocation]) -> Map:
        """
        gets the location of the scooters given the list of the nest allocations
        """
        result = []
        for nest_allocation in nests_spread:
            for i in range(nest_allocation.scooters_num):
                result.append(nest_allocation.location.to_numpy())
        return Map(np.array(result))
