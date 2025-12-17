import hydra
import torch
import sys

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple, List

script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.random_forest import RandomForestWrapper

class ModelLoader:
    """Helper class to load and manage multiple models."""
    
    def __init__(self, project_root: Path = project_root):
        self.project_root = project_root
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.models = {}
        self.configs = {}
        
    def load_config(self, model_name: str) -> DictConfig:
        """Load configuration for a specific model."""
        base_cfg = OmegaConf.load(self.project_root / "configs" / "config.yaml")
        
        model_cfg = OmegaConf.load(
            self.project_root / "configs" / "model" / f"{model_name}.yaml"
        )
        
        cfg = OmegaConf.merge(base_cfg, {"model": model_cfg})
        
        self.configs[model_name] = cfg
        return cfg
    
    def load_cnn_model(self, config_name: str, display_name: str = None):
        """Load a CNN model (UNet, DeepLab, etc.)."""
        cfg = self.load_config(config_name)
        
        model = hydra.utils.instantiate(cfg.model.params)
        
        checkpoint_path = (
            self.project_root / cfg.checkpoint_path / f"{cfg.model.name}.pth"
        )
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            return None
        
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        model = model.to(self.device)
        model.eval()
        
        name = display_name or cfg.model.name
        self.models[name] = {
            "model": model,
            "config": cfg,
            "type": "cnn",
            "checkpoint": checkpoint_path
        }
        
        print(f"Loaded CNN: {name}")
        return model
    
    def load_rf_model(self, config_name: str, display_name: str = None):
        """Load a Random Forest model."""
        cfg = self.load_config(config_name)
        
        rf = RandomForestWrapper(cfg)
        
        model_path = (
            self.project_root / cfg.checkpoint_path / f"{cfg.model.name}.joblib"
        )
        
        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}")
            return None
        
        rf.load(model_path)
        
        name = display_name or cfg.model.name
        self.models[name] = {
            "model": rf,
            "config": cfg,
            "type": "rf",
            "checkpoint": model_path
        }
        
        print(f"Loaded RF: {name}")
        return rf
    
    def load_all_models(self, model_configs: Dict[str, Tuple[str, str]]):
        """
        Load multiple models at once.
        
        Args:
            model_configs: Dict mapping display names to (config_name, model_type)
                          e.g., {"UNet": ("unet_2", "cnn"), "RF": ("random_forest", "rf")}
        """
        for display_name, (config_name, model_type) in model_configs.items():
            if model_type == "cnn":
                self.load_cnn_model(config_name, display_name)
            elif model_type == "rf":
                self.load_rf_model(config_name, display_name)
            else:
                print(f"Unknown model type: {model_type}")
        
        print(f"Loaded {len(self.models)} models total")
    
    def get_model(self, name: str):
        """Get a loaded model by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not loaded. Available: {list(self.models.keys())}")
        return self.models[name]["model"]
    
    def get_config(self, name: str):
        """Get configuration for a loaded model."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not loaded. Available: {list(self.models.keys())}")
        return self.models[name]["config"]
    
    def list_models(self) -> List[str]:
        """Print all loaded models."""
        print("Loaded Models:")
        print("-" * 60)
        model_names = {}
        for name, info in self.models.items():
            print(f"{name:30s} [{info['type']:3s}] {info['checkpoint'].name}")
            model_names.update({name: info['type']})
        print("-" * 60)
        return model_names