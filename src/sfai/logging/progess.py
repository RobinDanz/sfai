from tqdm.auto import tqdm
from typing import Optional

class PipelineProgess:
    """Handles a tqdm progress bar to display processing progress on an image.

        Attributes:
            _progess_bar (Optional[tqdm]): Internal variable representing a progress bar. Should not be manipulated directly.
    """
    def __init__(self):
        self._progess_bar: Optional[tqdm] = None

    def start(self, image_id: str, nb_tiles: int):
        """Starts a fresh progress bar

        Will stop any running PipelineProgress bar.

        Args:
            image_id (str): ID of the image that is displayed as description of the progress bar
            nb_tiles (int): Number of tiles that are processed for the image. Used to calculate progress
        """
        self.close()
        self._progess_bar = tqdm(
            total=nb_tiles,
            desc=f"Tiles ({image_id})",
            unit="tile",
            leave=False,
            position=1
        )
    
    def update(self, step: int = 1):
        """Updates the progress bar with the given step.

        Args:
            step (int, optional): Update step. Defaults to 1.
        """
        if self._progess_bar:
            self._progess_bar.update(step)

    def close(self):
        """Closes a progress bar. Reset internal variable for next use.
        """
        if self._progess_bar:
            self._progess_bar.close()
            self._progess_bar = None