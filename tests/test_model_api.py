import numpy as np
import torch

from diskd import DiSKDStudent, DiscreteSurvivalModel, fit_time_grid, simulate_competing_risks

torch.set_num_threads(1)


class ConstantCompetingTeacher:
    def __init__(self, num_risks: int, num_durations: int):
        self.num_risks = num_risks
        self.num_durations = num_durations

    def predict_interval_probs(self, data):
        n = len(data)
        event = torch.full((n, self.num_risks, self.num_durations), 0.08)
        no_event = torch.full((n, 1, self.num_durations), 1.0 - 0.08 * self.num_risks)
        return torch.cat([event, no_event], dim=1)


class ConstantOverallTeacher:
    def __init__(self, num_durations: int):
        self.num_durations = num_durations

    def predict_interval_probs(self, data):
        n = len(data)
        hazard = torch.full((n, self.num_durations), 0.18)
        return torch.stack([hazard, 1.0 - hazard], dim=1)


def _tiny_data():
    return simulate_competing_risks(n=48, seed=3, censor_max=0.05)


def _feature_cols(data):
    return [c for c in data.columns if c.startswith("x")][:6]


def _student_kwargs(time_grid):
    return {
        "num_risks": 2,
        "num_durations": time_grid.num_durations,
        "backbone": "mlp",
        "hidden_dim": 8,
        "hidden_layers": 1,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 64,
        "device": "cpu",
        "time_grid": time_grid,
    }


def test_diskd_student_competing_teacher_fit_and_predict():
    torch.manual_seed(3)
    data = _tiny_data()
    time_grid = fit_time_grid(data["duration"], num_durations=6)
    student = DiSKDStudent(
        **_student_kwargs(time_grid),
        teacher_model=ConstantCompetingTeacher(num_risks=2, num_durations=6),
        teacher_type="competing",
        eta=0.5,
        temperature=2.0,
    )

    student.fit(data, feature_cols=_feature_cols(data))
    hazard = student.predict_hazard(data.head(4))
    cif = student.predict_cif(data.head(4))
    survival = student.predict_survival(data.head(4))

    assert hazard.shape == (4, 2, 6)
    assert cif.shape == (4, 2, 6)
    assert survival.shape == (4, 6)
    assert np.all(hazard >= 0.0)
    assert np.all(hazard.sum(axis=1) <= 1.0 + 1e-6)
    assert student.history is not None
    assert np.isfinite(student.history.losses[-1])


def test_single_risk_model_predict_hazard_shape_and_bounds():
    torch.manual_seed(6)
    data = _tiny_data()
    data = data.assign(event_any=(data["event"] > 0).astype(int))
    time_grid = fit_time_grid(data["duration"], num_durations=6)
    model = DiscreteSurvivalModel(
        num_risks=1,
        num_durations=time_grid.num_durations,
        backbone="mlp",
        hidden_dim=8,
        hidden_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=64,
        device="cpu",
        time_grid=time_grid,
    )

    model.fit(data, feature_cols=_feature_cols(data), event_col="event_any")
    hazard = model.predict_hazard(data.head(4))

    assert hazard.shape == (4, 6)
    assert np.all((hazard >= 0.0) & (hazard <= 1.0))


def test_diskd_student_overall_teacher_fit_and_predict():
    torch.manual_seed(4)
    data = _tiny_data()
    time_grid = fit_time_grid(data["duration"], num_durations=6)
    student = DiSKDStudent(
        **_student_kwargs(time_grid),
        teacher_model=ConstantOverallTeacher(num_durations=6),
        teacher_type="overall",
        eta=0.5,
    )

    student.fit(data, feature_cols=_feature_cols(data))
    hazard = student.predict_hazard(data.head(4))
    cif = student.predict_cif(data.head(4))

    assert hazard.shape == (4, 2, 6)
    assert cif.shape == (4, 2, 6)
    assert student.history is not None
    assert np.isfinite(student.history.losses[-1])


def test_diskd_student_binary_horizon_teacher_fit_and_predict():
    torch.manual_seed(5)
    data = _tiny_data()
    time_grid = fit_time_grid(data["duration"], num_durations=6)

    def teacher_risk_at_horizon(frame):
        return np.full(len(frame), 0.25)

    student = DiSKDStudent(
        **_student_kwargs(time_grid),
        teacher_model=teacher_risk_at_horizon,
        teacher_type="binary_horizon",
        binary_risk_index=0,
        binary_horizon_index=3,
        eta=0.5,
    )

    student.fit(data, feature_cols=_feature_cols(data))
    risk1_cif = student.predict_cif(data.head(4), risk=1)

    assert risk1_cif.shape == (4, 6)
    assert student.binary_horizon_index == 3
    assert student.history is not None
    assert np.isfinite(student.history.losses[-1])
