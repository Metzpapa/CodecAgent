# codec/tools/base.py

from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING

# --- MODIFIED: Direct OpenAI import and simplified type hints ---
import openai
from pydantic import BaseModel

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State


class NoOpArgs(BaseModel):
    """A default Pydantic model for tools that don't require any arguments."""
    pass


class BaseTool(ABC):
    """
    An abstract base class for creating tools that the agent can use.

    This class has been simplified to work directly with the OpenAI client.
    It defines the essential properties and a simplified execution method for
    any tool in the system.
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

    # --- REMOVED: The supported_providers property is no longer needed. ---

    @abstractmethod
    def execute(self, state: 'State', args: BaseModel, client: openai.OpenAI, tmpdir: str) -> str:
        """
        Executes the tool's logic.

        This method now follows a simpler contract: it always returns a string.
        If a tool needs to produce multimodal output (like images or audio for the
        model to see), it should upload the file using the provided `client`,
        add the resulting file ID and its local path to a temporary list in the `state`
        object, and return a simple text confirmation. The agent is responsible for
        formatting the next API call with this multimodal content.

        Args:
            state: The current session state, providing context and memory.
            args: A Pydantic model instance containing the validated arguments for the tool.
            client: The active OpenAI client instance, used for any necessary API
                    operations like file uploads.
            tmpdir: The path to a temporary directory that is valid for the current
                    agent turn, for creating intermediate files.

        Returns:
            A `str` containing the text-based result or confirmation of the tool's execution.
        """
        pass