from src.data.data import generate_data


def test_generate_linear_data():
    data = generate_data("2 * x + 10", x=[1, 2, 3])
    correct = [12, 14, 16]
    for val, target in zip(data, correct):
        assert val == target


def test_generate_nonlinear_data():
    data = generate_data("x ** 2 + 10", x=[1, 2, 3])
    correct = [11, 14, 19]
    for val, target in zip(data, correct):
        assert val == target


def test_modded_data():
    data = generate_data("x % 2", x=[1, 2, 3])
    correct = [1, 0, 1]
    for val, target in zip(data, correct):
        assert val == target
