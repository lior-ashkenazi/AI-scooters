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
        return f"orig: {self.orig}, dest: {self.dest}\n" \
               f"start time: {self.start_time}, end time: {self.end_time}"

    def __repr__(self):
        return str(self)


class NestAllocation:
    def __init__(self, location: Point, scooters_num: int):
        self.location: Point = location
        self.scooters_num: int = scooters_num

    def __str__(self):
        return f"location: {self.location}\t scooters' number: {self.scooters_num}"

    def __repr__(self):
        return str(self)
