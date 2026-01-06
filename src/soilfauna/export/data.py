from typing import List, Dict, Any, TypeVar, Type
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

T = TypeVar("T", bound="Writable")

class Writable(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)

@dataclass
class CocoData(Writable):
    id: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class CocoImage(CocoData):
    width: int
    height: int
    file_name: str
    
@dataclass
class CocoAnnotation(CocoData):
    image_id: int
    category_id: int
    area: float
    iscrowd: int = 0
    bbox: List[float] = field(default_factory=list)
    segmentation: List[List[float]] = field(default_factory=list)

@dataclass
class CocoCategory(CocoData):
    name: str
    supercategory: str = ""
    

DEFAULT_CATEGORY = CocoCategory(1, 'Unclassified')