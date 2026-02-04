from typing import List, Dict, Any, TypeVar, Type
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

T = TypeVar("T", bound="Writable")

class Writable(ABC):
    """Asbtract class for writable objects.

    Writable objects are objects intended to be exported to a JSON file.
    """
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """

        Returns:
            Dict[str, Any]: Dict representation of the object
        """
        pass
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Creates an object from a dictionnary.

        Should be called from subclasses.

        Args:
            cls (Type[T]): Class of the returned object.
            data (Dict[str, Any]): Source dict to create an object from

        Returns:
            T: The instance of the created object
        """
        return cls(**data)

@dataclass
class CocoData(Writable):
    """Base class for COCO objects.

    Attributes:
        id (int): Id of the object
    """
    id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a Dict representation of the object

        Returns:
            Dict[str, Any]: Dict representation of the object
        """
        return asdict(self)

@dataclass
class CocoImage(CocoData):
    """Representation of a COCO image

    Attributes:
        id (int): 
        width (int): 
        height (int): 
        file_name (str): 
    """
    width: int
    height: int
    file_name: str
    
@dataclass
class CocoAnnotation(CocoData):
    """Representation of a COCO annotation

    Attributes:
        image_id (int): ID of the image of this annotation
        category_id (int): Category of the annotation
        area (float): Area of the annotation
        iscrowd (int):
        bbox (List[float]): Bounding-box of the annotation
        segmentation (List[List[float]]): Segmentation of the annotation
    """
    image_id: int
    category_id: int
    area: float
    iscrowd: int = 0
    bbox: List[float] = field(default_factory=list)
    segmentation: List[List[float]] = field(default_factory=list)

@dataclass
class CocoCategory(CocoData):
    """Representation of a COCO category

    Attributes:
        name (str): Name of the category
        supercategory (str): Name of the supercategory
    """
    name: str
    supercategory: str = ""


DEFAULT_CATEGORY = CocoCategory(1, 'Unclassified')
"""Global default category used for export
"""

