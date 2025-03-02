
# needed to import for allowing type-hinting: np.ndarray | torch.Tensor | None
import torch
from omni.isaac.lab.utils.configclass import configclass
from .visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from dataclasses import MISSING

@configclass
class VecAutoUpdateVisualizationMarkersCfg(VisualizationMarkersCfg):
    num_markers_per_type_env: dict|int = MISSING
    """The number of markers per type per env. If int, all types will have the same number."""

class VecAutoUpdateVisualizationMarkers(VisualizationMarkers):
    """Auto update visualization markers for the specified prims."""
    def __init__(self, 
                cfg: VisualizationMarkersCfg, 
                num_envs:int, 
                track_func: dict|None,
                device):
        super().__init__(cfg)
        self.device = device
        self.num_envs = num_envs
        self._setup_track_func(track_func)
        self._setup_num_per_type(cfg.num_markers_per_type_env)
        self._setup_marker_indices()
        self._setup_marker_pose_cache()

    def _setup_marker_pose_cache(self):
        """Setup marker pose cache."""
        self._marker_pose_cache = {}
        for name in self.cfg.markers.keys():
            num_idx = self.num_per_type_env[name]*self.num_envs 
            self._marker_pose_cache[name] = {
                "pos": torch.zeros((num_idx, 3), dtype=torch.float32, device=self.device),
                "quat": torch.zeros((num_idx, 4), dtype=torch.float32, device=self.device),
            }

    def _setup_track_func(self, track_func: dict):
        """Setup track function."""
        self._track_func = {}
        if track_func is None:
            return
        for name, func in track_func.items():
            if name not in self.cfg.markers.keys():
                raise ValueError(f"Marker {name} not in cfg.markers")
            if not callable(func):
                raise ValueError(f"Track function for {name} is not callable")
            self._track_func[name] = func

    def _setup_num_per_type(self, num_markers_per_type: dict):
        """Setup number of markers per type."""
        self.num_per_type_env = {name:1 for name in self.cfg.markers.keys()}
        if num_markers_per_type is None:
            for name, num in num_markers_per_type.items():
                if name not in self.cfg.markers.keys():
                    raise ValueError(f"Marker {name} not in cfg.markers")
                self.num_per_type_env[name] = num

    def _setup_marker_indices(self):
        # Build tensor
        self.marker_indices = torch.tensor([
            i for i, name in enumerate(self.cfg.markers.keys())
            for _ in range(self.num_envs)
            for _ in range(self.num_per_type_env[name])
        ])
    
    def _assert_update_shape(self, update_pos, update_quat, name):
        """Assert the shape of the update values."""
        exp_shape =self.num_per_type_env[name]*self.num_envs
        assert (update_pos.shape == (exp_shape, 3))
        assert (update_quat.shape == (exp_shape, 4))

    def step(self):
        """Update the visualization markers."""
        # Update values
        for name, func in self._track_func.items():
            update_pos, update_quat = func()
            self._assert_update_shape(update_pos, update_quat,name)
            self._set_pose(update_pos, update_quat, name)

        # We assume that everything that needs update externally has been updated      

        # Make the vectors for visualization
        marker_pos = torch.concatenate([m["pos"] for m in self._marker_pose_cache.values()], dim=0) 
        marker_quat = torch.concatenate([m["quat"] for m in self._marker_pose_cache.values()], dim=0)

        # Finally, visualize  
        self.visualize(marker_pos, marker_quat, None, self.marker_indices)

    def get_extern_update_func(self, name):
        def ext_update_func(marker_pos, marker_quat, _marker_name=name, _mgr=self):
            _mgr._assert_update_shape(marker_pos, marker_quat, _marker_name)
            _mgr._set_pose(marker_pos, marker_quat, _marker_name)
        return ext_update_func
    
    def _set_pose(self, pos, quat, name):
        self._marker_pose_cache[name]["pos"] = pos
        self._marker_pose_cache[name]["quat"] = quat