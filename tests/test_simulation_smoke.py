from diskd.preprocessing import fit_time_grid, transform_durations
from diskd.simulation import simulate_competing_risks


def test_simulation_and_time_grid_smoke():
    data = simulate_competing_risks(n=20, seed=7)
    assert {"duration", "event"}.issubset(data.columns)
    assert set(data["event"].unique()).issubset({0, 1, 2})
    grid = fit_time_grid(data["duration"], num_durations=5)
    idx = transform_durations(data["duration"], grid)
    assert idx.min() >= 0
    assert idx.max() < 5

