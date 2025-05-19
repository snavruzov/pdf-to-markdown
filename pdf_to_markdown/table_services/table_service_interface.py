from abc import ABC, abstractmethod
from typing import Optional, Union
from PIL import Image


class TableDetectionError(Exception):
    """Custom exception for errors during table detection."""
    pass

class TableInterface(ABC):
    """
    Interface for services that can detect the presence of tables on a PDF page.
    """

    @abstractmethod
    def get_best_size(self) -> Optional[Union[tuple[int, int], int]]:
        """
        Returns the recommended or optimal image configuration for table detection.

        This can be:
        - A tuple (width, height) for a specific image size.
        - An integer representing a DPI value for rendering.
        - None, if no specific configuration is critical, in which case a default rendering setting will be used.

        Some table detection models may perform best with images of a specific resolution or
        aspect ratio. This method provides guidance on that.

        Returns:
            Optional[Union[tuple[int, int], int]]: The recommended (width, height), DPI,
                                                     or None if there's no specific recommendation.
        """
        pass

    @abstractmethod
    def detect_tables_on_page(self, image: Image.Image) -> bool:
        """
        Synchronously detects if the given image (rendered from a PDF page) contains one or more tables.

        Args:
            image: A PIL.Image.Image object rendered from a PDF page.

        Returns:
            True if tables are detected on the page, False otherwise.
        
        Raises:
            TableDetectionError: If an error occurs during the detection process.
        """
        pass

    @abstractmethod
    async def adetect_tables_on_page(self, image: Image.Image) -> bool:
        """
        Asynchronously detects if the given image (rendered from a PDF page) contains one or more tables.

        Args:
            image: A PIL.Image.Image object rendered from a PDF page.

        Returns:
            True if tables are detected on the page, False otherwise.
        
        Raises:
            TableDetectionError: If an error occurs during the detection process.
        """
        pass 