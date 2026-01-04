from dataclasses import dataclass, fields, field
from pathlib import Path
import os
import yaml
from soilfauna.config import default
from typing import ClassVar, Mapping, List, get_args, get_origin
from types import UnionType

@dataclass(init=False)
class UserConfig:
    """
    Base class holding a configuration
    """
    CONFIG_NAMESPACE: ClassVar[str | None] = None
    
    verbose: bool = False
    root_dir: Path = default.ROOT_DIR
    
    @classmethod
    def from_file(cls, path: str | Path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
            
        base_config = cls._extract_base_config(raw)
        sub_config = cls._extract_sub_config(raw)
        
        obj = cls.from_dict(sub_config)
        obj._apply_base_config(base_config)
        
        return obj
            
    @classmethod
    def from_dict(cls, data: Mapping):
        fields_defs = {f.name: f for f in fields(cls)}
        obj = cls.__new__(cls)
        
        for name, field in fields_defs.items():
            if name in data:
                value = cls._coerce(field.type, data[name])
            else:
                value = field.default
            setattr(obj, name, value)

        obj.validate()
        return obj
        
    @classmethod
    def _extract_base_config(cls, raw: Mapping) -> dict:
        return {
            k: v for k, v in raw.items() if not isinstance(v, dict)
        }
        
    @classmethod
    def _extract_sub_config(cls, raw: Mapping) -> dict:
        if cls.CONFIG_NAMESPACE:
            return raw.get(cls.CONFIG_NAMESPACE, {})
        return {}
    
    def _apply_base_config(self, base_config: dict):
        base_fields = {f.name for f in fields(UserConfig)}
        
        for name in base_fields:
            if name in base_config:
                value = self._coerce(
                    UserConfig.__annotations__[name],
                    base_config[name],
                )
                
                setattr(self, name, value)
    
    @staticmethod 
    def _coerce(tp, value):
        if tp is Path and isinstance(value, str):
            return Path(value)
        if isinstance(tp, UnionType) and Path in get_args(tp):
            return Path(value)
        if get_origin(tp) is list and Path in get_args(tp):
            return [
                Path(val) for val in value if isinstance(val, str)
            ]
        return value
    
    def validate(self):
        """
        Config validation.
        
        Override in subclasses
        """
        pass
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        parts = []

        for f in fields(self.__class__):
            value = getattr(self, f.name, None)
            parts.append(f"{f.name}={value!r}")

        return f"{cls_name}({', '.join(parts)})"
    

@dataclass(init=False)
class SegmentationConfig(UserConfig):
    """
    Class holding a segmentation run configuration
    """
    CONFIG_NAMESPACE = 'segment'
    
    id: int | None = None
    name: str = default.DEFAULT_RUN_NAME
    base_dir: Path = default.DEFAULT_OUTPUT_DIR
    model: Path = default.DEFAULT_MODEL
    
    datasets: List[Path] = field(default_factory=[])
    
    def validate(self):
        if self.datasets is None:
            raise ValueError("Dataset is required")
        
        if self.id is None:
            self.id = self._generate_id()
            
    def _generate_id(self) -> int:
        run_dir = self.base_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        next_id = len(next(os.walk(run_dir))[1])
        
        return next_id + 1
    
    def create_run_folder(self):
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def base_output_dir(self) -> Path:
        return Path(os.path.join(self.base_dir, self.name, str(self.id)))
    
    