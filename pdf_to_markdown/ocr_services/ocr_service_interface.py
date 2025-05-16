from abc import ABC, abstractmethod

from PIL.Image import Image


class OCRProcessingError(Exception):
    pass


class OCRInterface(ABC):

    @abstractmethod
    def get_best_size(self) -> tuple[int, int]:
        """
        Returns the recommended or optimal image size (width, height) for OCR.

        Some OCR engines may perform best with images of a specific resolution or
        aspect ratio. This method provides guidance on that.
        If no specific size is critical, it might return None or a generally safe large size.

        Returns:
            tuple[int, int]: A tuple containing the recommended (width, height)
                             or None if there's no specific recommendation.
        """
        pass

    @abstractmethod
    def ocr_image(self, image: list[Image]) -> list[str]:
        """
        Performs Optical Character Recognition (OCR) on a list of images.

        This method takes a list of PIL Image objects and returns a list of strings,
        where each string is the extracted text from the corresponding image.
        This is expected to be a synchronous operation.

        Args:
            image (list[Image]): A list of PIL.Image.Image objects to process.

        Returns:
            list[str]: A list of strings, where each string is the OCR result
                       for the corresponding input image.
        """
        pass

    @abstractmethod
    async def aocr_image(self, image: list[Image]) -> list[str]:
        """
        Asynchronously performs Optical Character Recognition (OCR) on a list of images.

        This method takes a list of PIL Image objects and returns a list of strings,
        where each string is the extracted text from the corresponding image.
        This method is designed for asynchronous execution, suitable for I/O-bound tasks.

        Args:
            image (list[Image]): A list of PIL.Image.Image objects to process.

        Returns:
            list[str]: A list of strings, where each string is the OCR result
                       for the corresponding input image.
        """
        pass
