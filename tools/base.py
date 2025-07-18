# codec/tools/base.py

from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING

from pydantic import BaseModel

# --- MODIFIED: Use forward references to our new generic types ---
# These imports will only be processed by type checkers, not at runtime,
# preventing circular import errors.
if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector
    from llm.types import Message


# This class is a simple data structure and remains unchanged.
class NoOpArgs(BaseModel):
    """A default Pydantic model for tools that don't require any arguments."""
    pass


# REMOVED: The `_inline_schema_definitions` helper function has been removed.
# This logic is specific to preparing a schema for the Google GenAI API and
# will be moved into the `GeminiConnector` where it is actually needed. It does
# not belong in the generic `BaseTool` class.


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

    @abstractmethod
    def execute(self, state: 'State', args: BaseModel, connector: 'LLMConnector') -> str | 'Message':
        """
        Executes the tool's logic.

        Args:
            state: The current session state, providing context and memory.
            args: A Pydantic model instance containing the validated arguments for the tool.
            connector: The active LLMConnector instance, used for provider-specific
                       operations like file uploads.

        Returns:
            - A string for simple, text-based tool results.
            - A generic `Message` object for complex, multimodal results (e.g., from
              viewing a video or hearing audio), which will be added to the chat history.
        """
        pass

    # REMOVED: The `to_google_tool` method has been removed.
    # Each LLMConnector is now responsible for converting this BaseTool's
    # `args_schema` into the format required by its specific API. This is a
    # key part of the abstraction, ensuring that tools do not need to be
    # aware of the provider's implementation details.