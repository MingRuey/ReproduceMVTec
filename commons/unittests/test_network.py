from commons.network import spawn_network

class TestNetwork:

    def test_network_shape_patch17(self):
        model = spawn_network(17)

        shape_checks = [
            (None, 17, 17, 3),
            (None, 13, 13, 128), "Leaky",
            (None, 9, 9, 256), "Leaky",
            (None, 5, 5, 256), "Leaky",
            (None, 1, 1, 128), "Leaky",
            (None, 1, 1, 512), "Leaky"
        ]
        assert len(model.layers) == len(shape_checks)
        for layer, shape in zip(model.layers, shape_checks):
            if layer.name.startswith("input"):
                assert layer.output_shape[0] == shape
            elif shape != "Leaky":
                assert layer.output_shape == shape

    def test_network_shape_patch33(self):
        model = spawn_network(33)
        model.summary()

        shape_checks = [
            (None, 33, 33, 3),
            (None, 31, 31, 128), "Leaky",
            (None, 15, 15, 128),  # maxpool
            (None, 11, 11, 256), "Leaky",
            (None, 5, 5, 256),  # maxpool
            (None, 4, 4, 256), "Leaky",
            (None, 1, 1, 128), "Leaky",
            (None, 1, 1, 512), "Leaky"
        ]
        assert len(model.layers) == len(shape_checks)
        for layer, shape in zip(model.layers, shape_checks):
            if layer.name.startswith("input"):
                assert layer.output_shape[0] == shape
            elif shape != "Leaky":
                assert layer.output_shape == shape

    def test_network_shape_patch65(self):
        model = spawn_network(65)
        model.summary()

        shape_checks = [
            (None, 65, 65, 3),
            (None, 61, 61, 128), "Leaky",
            (None, 30, 30, 128),  # maxpool
            (None, 26, 26, 128), "Leaky",
            (None, 13, 13, 128),  # maxpool
            (None, 9, 9, 128), "Leaky",
            (None, 4, 4, 128),  # maxpool
            (None, 1, 1, 256), "Leaky",
            (None, 1, 1, 128), "Leaky",
            (None, 1, 1, 512), "Leaky"
        ]
        assert len(model.layers) == len(shape_checks)
        for layer, shape in zip(model.layers, shape_checks):
            if layer.name.startswith("input"):
                assert layer.output_shape[0] == shape
            elif shape != "Leaky":
                assert layer.output_shape == shape