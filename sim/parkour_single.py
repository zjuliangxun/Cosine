from .humanoid_amp import HumanoidAMP


class ParkourSingle(HumanoidAMP):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless
        )
        return
