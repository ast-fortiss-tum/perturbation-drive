import argparse
from typing import List, Dict, Any
import traceback

from sdsandbox_simulator import SDSandboxSimulator
from examples.models.example_agent import ExampleAgent

from perturbationdrive import PerturbationDrive, RandomRoadGenerator


def go(
    host: str,
    port: int,
    pert_funcs: List[str] = [],
    attention: Dict[str, Any] = {},
):
    try:
        simulator = SDSandboxSimulator(host=host, port=port)
        ads = ExampleAgent()
        road_generator = RandomRoadGenerator(map_size=250)
        benchmarking_obj = PerturbationDrive(simulator, ads)

        # start the benchmarking
        benchmarking_obj.grid_seach(
            perturbation_functions=pert_funcs,
            attention_map=attention,
            road_generator=road_generator,
            log_dir="./examples/sdsandbox_perturbations/logs.json",
            overwrite_logs=True,
            image_size=(240, 320),  # images are resized to these values
        )
        print(f"{5 * '#'} Finished Running SDSandBox Sim {5 * '#'}")
    except Exception as e:
        print(
            f"{5 * '#'} SDSandBox Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDSandBox Example")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="server sim host")
    parser.add_argument("--port", type=int, default=9091, help="bind to port")
    parser.add_argument(
        "--perturbation",
        dest="perturbation",
        action="append",
        type=str,
        default=[],
        help="perturbations to use on the model. by default all are used",
    )
    parser.add_argument(
        "--attention_map", type=str, default="", help="which attention map to use"
    )
    parser.add_argument(
        "--attention_threshold",
        type=float,
        default=0.5,
        help="threshold for attention map perturbation",
    )
    parser.add_argument(
        "--attention_layer",
        type=str,
        default="conv2d_5",
        help="layer for attention map perturbation",
    )

    args = parser.parse_args()
    attention = (
        {}
        if args.attention_map == ""
        else {
            "map": args.attention_map,
            "threshold": args.attention_threshold,
            "layer": args.attention_layer,
        }
    )

    print(f"{5 * '#'} Started Running SDSandBox Sim {5 * '#'}")
    go(
        host=args.host,
        port=args.port,
        pert_funcs=args.perturbation,
        attention=attention,
    )
