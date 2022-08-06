from data.trafficdatatypes import *
import data.config as config

import datetime as dt


class TimeGenerator:
    LATEST_HOUR: int = 23
    LATEST_MINUTE: int = 59
    LATEST_TIME: dt.time = dt.time(hour=LATEST_HOUR, minute=LATEST_MINUTE).replace(second=0,
                                                                                   microsecond=0)

    SCOOTERS_AVERAGE_SPEED: int = 20


    def generate_start_time(self, day_part):
        return dt.time(
            hour=self._sample_hour_normal_distribution(*config.DAY_PARTS_HOURS_PROB[day_part]),
            minute=np.random.randint(0, TimeGenerator.LATEST_MINUTE)).replace(second=0,
                                                                              microsecond=0)

    def generate_end_time(self, a: Point, b: Point, start_time: dt.time):
        dist: float = point_dist(a, b)
        time_in_hours: float = dist / TimeGenerator.SCOOTERS_AVERAGE_SPEED
        time_in_minutes: int = round(60 * time_in_hours)
        start_time: dt.datetime = dt.datetime(year=2022, month=1, day=1, hour=start_time.hour,
                                              minute=start_time.minute, second=start_time.second)
        end_time: dt.datetime = start_time + dt.timedelta(minutes=round(time_in_minutes))
        if end_time.hour <= TimeGenerator.LATEST_HOUR:
            return end_time.time()
        if end_time.hour > 23:
            return TimeGenerator.LATEST_TIME

    def _sample_hour_normal_distribution(self, hour_mean, hour_variance):
        while True:
            hour: int = round(np.random.normal(hour_mean, hour_variance))
            if hour < 24:
                return hour
