# used modules from perturbation drive
from perturbationdrive import (
    PerturbationSimulator,
    ADS,
    Scenario,
    ScenarioOutcome,
    ImagePerturbation,
    GlobalLog,
)

# used libraries
from gym_donkeycar.core.sim_client import SimClient
from typing import Any, Union, List, Tuple, Dict
import time

# imports from this example
from examples.self_driving_sandbox_donkey.donkey_exec import DonkeyProcess
from examples.self_driving_sandbox_donkey.donkey_sim_msg_handler import (
    DonkeySimMsgHandler,
)


class SDSandboxSimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./sim/donkey-sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
        show_image_cb=True,
    ):
        super().__init__(
            max_xte=2.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.port = port
        self.client: Union[DonkeySimMsgHandler, None] = None
        self.process: DonkeyProcess = DonkeyProcess()
        self.logger = GlobalLog("SDSandBoxSimulator")
        self.show_image_cb = show_image_cb

    def connect(self):
        # launch the sim binary here
        self.process.start(self.simulator_exe_path, port=self.port)

        super().connect()
        address = (self.host, self.port)
        handler = DonkeySimMsgHandler(show_image_cb=self.show_image_cb)
        self.client = SimClient(address, handler)

        # wait for the first observation here
        while len(self.client.msg_handler.sim_data) == 0:
            self.logger.info("Waiting for inital obs")
            time.sleep(0.04)
        # the last value if the road width, which should be equivalent to max_xte * 2
        self.initial_pos = (
            self.client.msg_handler.sim_data["pos_x"],
            self.client.msg_handler.sim_data["pos_y"],
            self.client.msg_handler.sim_data["pos_z"],
            self.max_xte * 2,
        )

    def simulate_scanario(
        self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation
    ) -> ScenarioOutcome:
        try:
            waypoints = scenario.waypoints
            perturbation_function_string = scenario.perturbation_function
            perturbation_scale = scenario.perturbation_scale

            # set all params for init loop
            actions = [[0.0, 0.0]]
            perturbed_image = None

            # set up params for saving data
            pos_list = []
            xte_list = []
            actions_list = []
            speed_list = []
            original_image_list = []
            perturbed_image_list = []
            isSuccess = False
            timeout = False

            # reset the scene to match the scenario
            self.client.msg_handler.reset_scenario(waypoints)
            self.logger.info(f"Reset the scenario")
            time.sleep(2.0)
            start_time = time.time()

            # run the scenario
            while self._client_connected(self.client):
                try:
                    # TODO: Play around with this value
                    time.sleep(0.01)
                    # we provide the actions and perturbed image here
                    obs: Dict[str, Any] = self.client.msg_handler.update(
                        actions, perturbed_image, perturbation_function_string
                    )
                    # check if we are done
                    if obs["done"]:
                        isSuccess = True
                        self.logger.info("SDSandBox: Done")
                        break
                    elif abs(obs["xte"]) > self.max_xte:
                        break
                    elif time.time() - start_time > 300:
                        self.logger.info("SDSandBox: Timeout after 300s")
                        timeout = True
                        break

                    # perturb the image
                    perturbed_image = perturbation_controller.perturbation(
                        obs["image"],
                        perturbation_name=perturbation_function_string,
                        intensity=perturbation_scale,
                    )
                    # get ads actions
                    actions = agent.action(perturbed_image)

                    # save data for output
                    pos_list.append([obs["pos_x"], obs["pos_y"], obs["pos_z"]])
                    xte_list.append(obs["xte"])
                    speed_list.append(obs["speed"])
                    original_image_list.append(obs["image"])
                    perturbed_image_list.append(perturbed_image)
                    actions_list.append(actions)

                except KeyboardInterrupt:
                    self.logger.info(
                        f"{5 * '+'} SDSandBox Simulator Got Interrupted {5 * '+'}"
                    )
                    self.client.stop()
                    raise KeyboardInterrupt
            print("SDSandBox: Finished scenario")
            # break
            self.client.msg_handler.update([[0.0, 0.0]], None)
            # send reset to sim client
            self.client.msg_handler.reset_car()

            # return the resul of this simulation
            res = ScenarioOutcome(
                frames=[x for x in range(len(pos_list))],
                pos=pos_list,
                xte=xte_list,
                speeds=speed_list,
                actions=[
                    (f"{action[0][0]}", f"{action[0][1]}") for action in actions_list
                ],
                original_images=original_image_list,
                perturbed_images=perturbed_image_list,
                scenario=scenario,
                isSuccess=isSuccess
                and max([abs(xte) for xte in xte_list]) <= self.max_xte,
                timeout=timeout,
            )
            del (
                pos_list,
                xte_list,
                speed_list,
                actions_list,
                obs,
                perturbed_image,
                perturbed_image_list,
                original_image_list,
                actions,
            )
            return res
        except Exception as e:
            # close the simulator
            self.tear_down()
            del (
                pos_list,
                xte_list,
                speed_list,
                actions_list,
                obs,
                perturbed_image,
                perturbed_image_list,
                original_image_list,
                actions,
            )
            # throw the exception
            raise e

    def tear_down(self):
        self.client.msg_handler.on_disconnect()
        self.process.quit()

    def _client_connected(self, client: SimClient) -> bool:
        """
        Retruns true if the client is still connected
        """
        return client.is_connected()

    def name(self) -> str:
        """
        Returns the name of the simulator
        """
        return "SDSandBoxSimulatorAdapter"
