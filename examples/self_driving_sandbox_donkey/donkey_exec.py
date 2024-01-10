import glob
import subprocess
import os
import platform
import time
from perturbationdrive import GlobalLog as Gl


class DonkeyProcess(object):
    """
    Utility class to start donkey process if needed.
    """

    def __init__(self) -> None:
        self.process = None
        self.logger = Gl("DonkeyProcess")

    def start(
        self,
        sim_path: str = "./sim/donkey-sim.app",
        port: int = 9091,
        headless: bool = False,
    ):
        """
        :param sim_path: (str) Path to the executable
        :param headless: (bool)
        :param port: (int)
        """
        if not os.path.exists(sim_path):
            self.logger.error("{} does not exist".format(sim_path))
            return

        cwd = os.getcwd()
        file_name = (
            sim_path.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
        true_filename = os.path.basename(os.path.normpath(file_name))
        launch_string = None
        port_args = ["--port", str(port), "-logFile", "unitylog.txt"]
        platform_ = platform.system()

        if platform_.lower() == "linux" and sim_path:
            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform_.lower() == "darwin" and sim_path:
            candidates = glob.glob(
                os.path.join(
                    cwd, file_name + ".app", "Contents", "MacOS", true_filename
                )
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform_.lower() == "windows" and sim_path:
            candidates = glob.glob(os.path.join(cwd, file_name) + ".exe")
            if len(candidates) > 0:
                launch_string = candidates[0]

        if launch_string is None:
            self.logger.critical("Launch string is Null")
        else:
            self.logger.info("This is the launch string {}".format(launch_string))

            # Launch Unity environment
            if headless:
                self.process = subprocess.Popen(
                    [launch_string, "-batchmode"] + port_args
                )
            else:
                self.process = subprocess.Popen([launch_string] + port_args)

            if sim_path:
                # hack to wait for the simulator to start
                time.sleep(20)

        self.logger.info("donkey subprocess started")

    def quit(self):
        """
        Shutdown donkey environment
        """
        if self.process is not None:
            self.logger.info("Closing donkey sim subprocess")
            self.process.kill()
            self.process = None
