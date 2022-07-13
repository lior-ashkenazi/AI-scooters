class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y


class Ride:
    def __init__(self, orig: Point, dest: Point, start_time: int, end_time: int):
        self.orig: Point = orig
        self.dest: Point = dest
        self.start_time: int = start_time
        self.end_time: int = end_time

