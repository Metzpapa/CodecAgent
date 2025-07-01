import inspect
from abc import ABC, abstractmethod
from typing import Type, Dict, Any, TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel, Field

# This import will only be processed by type checkers, not at runtime.
# This resolves the "State is not defined" error in your IDE.
if TYPE_CHECKING:
    from state import State


# A default Pydantic model for tools that don't require any arguments.
class NoOpArgs(BaseModel):
    pass

class BaseTool(ABC):
    """
    An abstract base class for creating tools that the agent can use.

    This class provides a standardized structure for defining a tool's name,
    description, and argument schema. It uses Pydantic for robust argument
    validation and automatically generates a `FunctionDeclaration` compatible
    with the Google GenAI API. This ensures a single, reliable source of truth
    for the tool's interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool, used by the LLM to call it."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description of what the tool does, for the LLM to understand its purpose."""
        pass

    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """The Pydantic model defining the arguments for the tool."""
        pass

    @abstractmethod
    def execute(self, state: 'State', args: BaseModel) -> str:
        """
        The core logic of the tool.

        Args:
            state: The current state of the agent, providing context like asset paths.
            args: A Pydantic model instance containing the validated arguments
                  provided by the LLM.

        Returns:
            A string observation or result to be sent back to the LLM.
        """
        pass

    def to_google_tool(self) -> types.FunctionDeclaration:
        """
        Converts the tool's Pydantic schema into a Google GenAI FunctionDeclaration.

        This method introspects the `args_schema` Pydantic model and transforms
        it into the format required by the Gemini API's `tools` parameter.
        """
        schema_dict = self.args_schema.model_json_schema()

        def uppercase_types(d: Dict[str, Any]) -> Dict[str, Any]:
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == 'type' and isinstance(value, str):
                        d[key] = value.upper()
                    else:
                        uppercase_types(value)
            elif isinstance(d, list):
                for item in d:
                    uppercase_types(item)
            return d

        uppercase_types(schema_dict)

        schema_dict.pop("title", None)
        schema_dict.pop("description", None)

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(**schema_dict) if schema_dict.get("properties") else None,
        )