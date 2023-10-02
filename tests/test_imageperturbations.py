import unittest
from perturbationdrive import ImagePerturbation


class ImagePerturbationTestCase(unittest.TestCase):
    def setUp(self):
        self.perturbation = ImagePerturbation()

    def test_empty(self):
        """Empty test placeholder"""

        result = self.perturbation.peturbate()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
