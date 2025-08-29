# examples/carla/main.py
from perturbationdrive.imageperturbations import ImagePerturbation
from perturbationdrive.Simulator.Scenario import Scenario
from perturbationdrive.perturbationdrive import PerturbationDrive

from examples.carla.carla_simulator import CarlaSimulator
from examples.carla.example_ads import ExampleADS

if __name__ == "__main__":
    sim = CarlaSimulator(town="Town03", image_width=320, image_height=160, port=2000)
    ads = ExampleADS()

    # no perturbation → identity
    pert = ImagePerturbation()

    # one simple scenario (e.g., 20s free drive)
    scn = Scenario(
        name="carla_smoke_test",
        initial_pos=None,     # or (x, y, z, yaw) from CARLA map
        duration_s=20.0,
        metadata={"town": "Town03"},
    )

    bench = PerturbationDrive(simulator=sim, ads=ads)
    sim.connect()
    try:
        # run just this scenario with the perturbation controller bound inside benchmarker
        bench.simulate_scenarios([scn], perturbation_controller=pert)
    finally:
        sim.tear_down()