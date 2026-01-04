from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field

@dataclass
class CocoData:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CocoData":
        return cls(**data)

@dataclass
class CocoImage(CocoData):
    id: int
    width: int
    height: int
    file_name: str
    
@dataclass
class CocoAnnotation(CocoData):
    id: int
    image_id: int
    category_id: int
    area: float
    iscrowd: int = 0
    bbox: List[float] = field(default_factory=list)
    segmentation: List[List[float]] = field(default_factory=list)

@dataclass
class CocoCategory(CocoData):
    id: int
    name: str
    supercategory: str = ""
    

DEFAULT_CATEGORY = CocoCategory(1, 'Unclassified')