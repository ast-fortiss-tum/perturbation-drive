from perturbationdrive import PerturbationDrive
from examples.models.example_agent import ExampleAgent

ads=ExampleAgent()
benchmarking_object = PerturbationDrive(None,ads)

# perform grid search as end to end test
benchmarking_object.offline_perturbation(
    dataset_path="./examples/dataset_dummy",
    perturbation_functions=["gaussian_noise", "impulse_noise"],
    attention_map={},
    log_dir='test_dataset_test.json',
    overwrite_logs=True,
    image_size=(240,240)
)
