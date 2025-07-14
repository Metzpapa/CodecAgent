# codec/tools/base.py

import inspect
from abc import ABC, abstractmethod
from typing import Type, Dict, Any, TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# This import will only be processed by type checkers, not at runtime.
if TYPE_CHECKING:
    from state import State


# A default Pydantic model for tools that don't require any arguments.
class NoOpArgs(BaseModel):
    pass


def _inline_schema_definitions(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively inlines $ref definitions from a '$defs' block into a JSON schema.
    This is necessary because the google.genai.types.Schema object does not
    support top-level '$defs' or references.
    """
    if not isinstance(schema, dict):
        return schema

    # If there are no definitions, there's nothing to inline.
    defs = schema.get('$defs')
    if not defs:
        return schema

    # Create a deep copy to modify, leaving the original schema untouched.
    # We also remove the '$defs' key from the top level of the new schema.
    inlined_schema = {k: v for k, v in schema.items() if k != '$defs'}

    def _dereference(node: Any) -> Any:
        if isinstance(node, dict):
            if '$ref' in node and isinstance(node['$ref'], str):
                # Extract the definition name (e.g., '#/$defs/ClipToAdd' -> 'ClipToAdd')
                def_name = node['$ref'].split('/')[-1]
                if def_name in defs:
                    # Return a recursively dereferenced version of the definition
                    return _dereference(defs[def_name])
                else:
                    # Reference not found, return as is
                    return node
            else:
                # Recurse into dictionary values
                return {k: _dereference(v) for k, v in node.items()}
        elif isinstance(node, list):
            # Recurse into list items
            return [_dereference(item) for item in node]
        else:
            return node

    return _dereference(inlined_schema)


class BaseTool(ABC):
    """
    An abstract base class for creating tools that the agent can use.
    ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def execute(self, state: 'State', args: BaseModel, client: 'genai.Client') -> str | types.Content:
        pass

    def to_google_tool(self) -> types.FunctionDeclaration:
        """
        Converts the tool's Pydantic schema into a Google GenAI FunctionDeclaration.
        This now includes a crucial step to inline nested definitions ('$defs').
        """
        schema_dict = self.args_schema.model_json_schema()

        # --- NEW: Inline all definitions from '$defs' ---
        # This flattens the schema, making it compatible with the strict
        # google.genai.types.Schema object, which does not allow '$ref'.
        final_schema = _inline_schema_definitions(schema_dict)
        # --- END OF NEW BLOCK ---

        # The Google API's Schema object doesn't support all JSON Schema validation keywords.
        # We need to recursively sanitize the schema to remove unsupported fields.
        ALLOWED_KEYS = {'type', 'description', 'format', 'enum', 'properties', 'required', 'items'}

        def sanitize_and_uppercase(d: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(d, dict):
                return d
            
            sanitized = {key: value for key, value in d.items() if key in ALLOWED_KEYS}

            if 'type' in sanitized and isinstance(sanitized['type'], str):
                sanitized['type'] = sanitized['type'].upper()

            if 'properties' in sanitized:
                sanitized['properties'] = {k: sanitize_and_uppercase(v) for k, v in sanitized['properties'].items()}
            if 'items' in sanitized:
                sanitized['items'] = sanitize_and_uppercase(sanitized['items'])
            
            return sanitized

        sanitized_schema = sanitize_and_uppercase(final_schema)

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(**sanitized_schema) if sanitized_schema.get("properties") else None,
        )