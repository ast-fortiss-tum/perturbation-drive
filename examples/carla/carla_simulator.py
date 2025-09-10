# examples/carla/carla_simulator.py
from __future__ import annotations
import time
from typing import Tuple, Optional
import numpy as np

try:
    import carla  # pip install carla==0.9.15 (match your CARLA build)
except ImportError as e:
    raise ImportError(
        "Missing CARLA Python API. Install a version matching your CARLA server, e.g. `pip install carla==0.9.15`."
    ) from e

from perturbationdrive.Simulator.Scenario import Scenario, ScenarioOutcome
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.Simulator.Simualtor import PerturbationSimulator  # note: repo spells 'Simualtor'
from perturbationdrive.imageperturbations import ImagePerturbation


class CarlaSimulator(PerturbationSimulator):
    """
    CARLA adapter for PerturbationDrive.
    Responsibilities:
      - connect(): attach to a running CARLA server and set up world, vehicle, and RGB camera
      - simulate_scanario(): run one scenario loop:
            * grab sensor frame
            * apply ImagePerturbation
            * feed to ADS to get control
            * step sim until done/failed/timeout
      - tear_down(): clean actors and restore settings
    """

    def __init__(
        self,
        max_xte: float = 2.0,
        simulator_exe_path: str = "",
        host: str = "127.0.0.1",
        port: int = 2000,
        initial_pos: Optional[Tuple[float, float, float, float]] = None,  # x, y, z, yaw (world coords)
        town: str = "Town03",
        image_width: int = 320,
        image_height: int = 160,
        fov: float = 90.0,
        fixed_delta_seconds: float = 0.05,  # 20 FPS sync mode
    ):
        super().__init__(max_xte, simulator_exe_path, host, port, initial_pos)
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._vehicle: Optional[carla.Actor] = None
        self._camera: Optional[carla.Sensor] = None
        self._actors = []
        self._image = None
        self._town = town
        self._w = image_width
        self._h = image_height
        self._fov = fov
        self._fixed_dt = fixed_delta_seconds
        self._orig_settings = None

    # --- lifecycle ---

    def connect(self):
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(10.0)
        self._world = self._client.load_world(self._town)

        # synchronous mode for deterministic stepping
        self._orig_settings = self._world.get_settings()
        settings = self._orig_settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_dt
        self._world.apply_settings(settings)

        blueprint_library = self._world.get_blueprint_library()
        veh_bp = blueprint_library.filter("model3")[0]  # any drivable vehicle
        spawn = self._pick_spawn_point(self._world)
        self._vehicle = self._world.spawn_actor(veh_bp, spawn)
        self._actors.append(self._vehicle)

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self._w))
        cam_bp.set_attribute("image_size_y", str(self._h))
        cam_bp.set_attribute("fov", str(self._fov))
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.4))  # hood cam
        self._camera = self._world.spawn_actor(cam_bp, cam_tf, attach_to=self._vehicle)
        self._actors.append(self._camera)

        # image callback
        self._image = None
        self._camera.listen(self._on_carla_image)

        # tick once to stabilize
        self._world.tick()

    def tear_down(self):
        if self._camera:
            self._camera.stop()
        for a in self._actors[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        self._actors.clear()
        if self._world and self._orig_settings:
            try:
                self._world.apply_settings(self._orig_settings)
            except Exception:
                pass
        self._image = None
        self._vehicle = None
        self._camera = None
        self._world = None
        self._client = None

    # --- core simulation ---

    def simulate_scanario(
        self,
        agent: ADS,
        scenario: Scenario,
        perturbation_controller: ImagePerturbation,
        perturb=True, 
        model_drive=True, 
        weather=None, 
        intensity=None,
    ) -> ScenarioOutcome:
        """
        Minimal loop:
          - drive for scenario.duration_s (or until failure)
          - failure condition example: cross-track error > max_xte or collision
        The framework’s Scenario/Outcome types come from the repo.
        """
        assert self._world and self._vehicle

        start_time = time.time()
        collided = False
        xte_violation = False

        # optional collision sensor if you want hard failure on contact
        col_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        col_sensor = self._world.spawn_actor(col_bp, carla.Transform(), attach_to=self._vehicle)
        self._actors.append(col_sensor)
        col_sensor.listen(lambda ev: self._set_collision_flag())

        # reset agent if supported
        if hasattr(agent, "reset"):
            agent.reset()

        # place vehicle if scenario has an initial pose
        if scenario.initial_pos is not None:
            x, y, z, yaw = scenario.initial_pos
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            self._vehicle.set_transform(tf)
            self._world.tick()

        # main loop: synchronous ticks
        while True:
            self._world.tick()

            # fetch latest camera frame
            frame = self._image
            if frame is None:
                continue  # wait for first image

            # apply perturbation (expects uint8 HxWx3)
            perturbed = perturbation_controller.apply(frame)

            # predict control using ADS (expects np.uint8 image)
            steer, throttle, brake = agent.predict(perturbed)

            # send to CARLA
            control = carla.VehicleControl(
                throttle=float(np.clip(throttle, 0.0, 1.0)),
                steer=float(np.clip(steer, -1.0, 1.0)),
                brake=float(np.clip(brake, 0.0, 1.0)),
            )
            self._vehicle.apply_control(control)

            # sample a naive cross-track error proxy (lane center not trivial in CARLA; demo uses deviation from forward vector + speed)
            xte_violation = self._heuristic_xte_violation(self._vehicle, threshold=self.max_xte)

            # end conditions
            now = time.time()
            if (scenario.duration_s is not None) and (now - start_time >= scenario.duration_s):
                break
            if self._collided:
                collided = True
                break
            if xte_violation:
                break

        outcome = ScenarioOutcome(
            success=not (collided or xte_violation),
            collided=collided,
            xte_violation=xte_violation,
            duration_s=float(time.time() - start_time),
        )
        # clean temp sensors
        try:
            col_sensor.stop()
            col_sensor.destroy()
            self._actors.remove(col_sensor)
        except Exception:
            pass
        self._collided = False
        return outcome

    # --- helpers ---

    _collided = False

    def _set_collision_flag(self):
        self._collided = True

    def _pick_spawn_point(self, world: "carla.World") -> carla.Transform:
        spawns = world.get_map().get_spawn_points()
        return spawns[0] if spawns else carla.Transform()

    def _on_carla_image(self, image: "carla.Image"):
        # Convert CARLA BGRA bytes → uint8 RGB numpy array (HxWx3)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self._image = array.copy()

    def _heuristic_xte_violation(self, vehicle: "carla.Actor", threshold: float) -> bool:
        v = vehicle.get_velocity()
        speed = np.hypot(v.x, v.y)
        # crude proxy: too much lateral slip relative to heading at speed
        ang_vel = vehicle.get_angular_velocity().z
        return (abs(ang_vel) * max(speed, 0.1)) > (threshold * 2.0)