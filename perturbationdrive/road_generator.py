import random


class RoadGenerator:
    """
    Instanciates the road generator class.

    Given the performance of the car, it will create a road with an appropriate difficulty level
    """

    def __init__(self):
        pass

    def generate_random_road(self, length):
        commands = ["S", "L", "R", "DX", "DY"]
        road = ["S 4"]

        while length > 0:
            command = random.choice(commands)

            if command in ["S", "L", "R"]:
                value = random.randint(1, 10)
                length -= value
            else:
                value = random.uniform(-25, 25)

            road_segment = f"{command} {round(value, 2)}"
            road.append(road_segment)

        return road
