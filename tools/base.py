# codec/tools/base.py

import inspect
from abc import ABC, abstractmethod
from typing import Type, Dict, Any, TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel, Field

# This import will only be processed by type checkers, not at runtime.
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

        This method introspects the `args_schema` Pydantic model, transforms
        it into the format required by the Gemini API's `tools` parameter,
        and prunes unsupported validation fields.
        """
        schema_dict = self.args_schema.model_json_schema()

        # --- START OF THE FIX ---
        # The Google API's Schema object doesn't support all JSON Schema validation keywords.
        # We need to recursively sanitize the schema to remove unsupported fields like
        # 'exclusiveMinimum', 'title', etc., which Pydantic adds automatically.
        
        # Define the set of keys that are allowed by the Google GenAI Schema.
        ALLOWED_KEYS = {'type', 'description', 'format', 'enum', 'properties', 'required', 'items'}

        def sanitize_and_uppercase(d: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(d, dict):
                return d
            
            # Create a new dictionary containing only the allowed keys.
            sanitized = {key: value for key, value in d.items() if key in ALLOWED_KEYS}

            # Uppercase the 'type' field if it exists.
            if 'type' in sanitized and isinstance(sanitized['type'], str):
                sanitized['type'] = sanitized['type'].upper()

            # Recurse into nested structures.
            if 'properties' in sanitized:
                sanitized['properties'] = {k: sanitize_and_uppercase(v) for k, v in sanitized['properties'].items()}
            if 'items' in sanitized:
                sanitized['items'] = sanitize_and_uppercase(sanitized['items'])
            
            return sanitized

        # Sanitize the entire schema dictionary.
        sanitized_schema = sanitize_and_uppercase(schema_dict)
        # --- END OF THE FIX ---

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(**sanitized_schema) if sanitized_schema.get("properties") else None,
        )