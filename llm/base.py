# codec/llm/base.py

from abc import ABC, abstractmethod
from typing import List, Any, TYPE_CHECKING, Optional

# Import our new generic types
from .types import Message, LLMResponse, FileObject

# Use a forward reference for BaseTool to avoid a circular import dependency.
# The tool definition will eventually need the connector, and the connector needs
# the tool definition for type hinting. This import will only be evaluated by
# type checkers, not at runtime, preventing an import cycle.
if TYPE_CHECKING:
    from tools.base import BaseTool


class LLMConnector(ABC):
    """
    Abstract Base Class for LLM provider connectors.

    This class defines a common interface for interacting with different
    Large Language Model providers, such as Google Gemini or OpenAI. It ensures
    that the core agent logic can remain provider-agnostic by programming
    against this interface.
    """

    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the connector with the necessary credentials and model info.

        Args:
            api_key: The API key for the LLM provider.
            model_name: The specific model to be used (e.g., 'gemini-1.5-pro', 'gpt-4o').
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> Any:
        """
        Initializes and returns the provider-specific API client.
        This method is called by the constructor.
        """
        pass

    @abstractmethod
    def generate_content(
        self,
        history: List[Message],
        tools: List['BaseTool'],
        system_prompt: str,
        last_response_id: Optional[str]
    ) -> LLMResponse:
        """
        Sends a request to the LLM and gets a response.

        This method is responsible for translating the generic `Message` history
        and `BaseTool` list into the provider-specific format before making the API call.

        Args:
            history: A list of generic `Message` objects representing the conversation history.
            tools: A list of `BaseTool` instances available for the model to call.
            system_prompt: The system prompt to guide the model's behavior.
            last_response_id: The ID of the previous response, for use by stateful APIs.
                              Stateless APIs can ignore this.

        Returns:
            A generic `LLMResponse` object containing the model's reply and metadata.
        """
        pass

    @abstractmethod
    def upload_file(self, file_path: str, mime_type: str, display_name: str) -> FileObject:
        """
        Uploads a file to the provider's service for use in multimodal prompts.

        Args:
            file_path: The local path to the file to be uploaded.
            mime_type: The MIME type of the file (e.g., 'image/jpeg', 'audio/mpeg').
            display_name: A user-friendly name for the file.

        Returns:
            A generic `FileObject` containing the provider-specific ID and URI.
        """
        pass

    @abstractmethod
    def delete_file(self, file_id: str):
        """
        Deletes a previously uploaded file from the provider's service.

        Args:
            file_id: The provider-specific unique identifier of the file to delete.
        """
        pass