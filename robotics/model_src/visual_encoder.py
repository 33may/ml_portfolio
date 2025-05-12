from abc import ABC, abstractmethod
import torch
from transformers import CLIPModel, CLIPProcessor

class VisualEncoderBase(ABC):
    """
    Abstract base class for visual encoders.
    All custom encoders must inherit from this class and implement the required methods.
    """

    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def load_model(self):
        """
        Loads the pretrained model.
        """
        pass

    @abstractmethod
    def preprocess(self, image):
        """
        Preprocesses a single image or a batch of images for the encoder.
        Returns a tensor ready to be passed to the model.
        """
        pass

    @abstractmethod
    def encode(self, image_tensor):
        """
        Runs the model on the input tensor and returns the embeddings.
        """
        pass

    @abstractmethod
    def get_output_shape(self):
        """
        Runs the model on the input tensor and returns the embeddings.
        """
        pass

    def to(self, device):
        """
        Moves the model to the specified device.
        """
        self.device = device
        return self


class CLIPVisualEncoder(VisualEncoderBase):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)

    def preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def encode(self, image_tensor):
        inputs = self.preprocess(image_tensor)
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=inputs)

    def get_output_shape(self):
        return self.model.config.projection_dim

