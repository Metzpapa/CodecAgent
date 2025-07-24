# codec/tools/base.py

from abc import ABC, abstractmethod
# --- MODIFIED: Add Optional for the new property's return type ---
from typing import Type, TYPE_CHECKING, Union, Tuple, List, Optional

from pydantic import BaseModel

# --- MODIFIED: Update forward references for the new return type ---
if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector
    from llm.types import ContentPart


# This class is a simple data structure and remains unchanged.
class NoOpArgs(BaseModel):
    """A default Pydantic model for tools that don't require any arguments."""
    pass


class BaseTool(ABC):
    """
    An abstract base class for creating tools that the agent can use.

    This class is now provider-agnostic. It defines the essential properties
    and the execution method for any tool in the system, without containing
    any logic specific to a particular LLM provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description of what the tool does, for the LLM to understand its purpose."""
        pass

    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """The Pydantic model that defines the arguments for the tool."""
        pass

    @property
    def supported_providers(self) -> Optional[List[str]]:
        """
        Specifies which LLM providers this tool is compatible with.

        - Returns a list of provider names in lowercase (e.g., ['gemini', 'openai']).
        - Returns None (the default) to indicate the tool is universal and works
          for all configured providers.

        This should be overridden in any tool class that relies on a
        provider-specific capability.
        """
        return None

    @abstractmethod
    def execute(self, state: 'State', args: BaseModel, connector: 'LLMConnector') -> Union[str, Tuple[str, List['ContentPart']]]:
        """
        Executes the tool's logic.

        Args:
            state: The current session state, providing context and memory.
            args: A Pydantic model instance containing the validated arguments for the tool.
            connector: The active LLMConnector instance, used for provider-specific
                       operations like file uploads.

        Returns:
            - A `str` for simple, text-based tool results.
            - A `Tuple[str, List[ContentPart]]` for complex, multimodal results.
              The first element of the tuple is a simple text confirmation string
              that will be placed in the 'tool' role message. The second element
              is a list of the actual multimodal `ContentPart`s (images, audio)
              that will be placed in a subsequent 'user' role message for the
              model to perceive.
        """
        pass