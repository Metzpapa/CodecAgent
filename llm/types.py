# codec/llm/types.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class FileObject(BaseModel):
    """A generic, provider-agnostic representation of an uploaded file."""
    id: str = Field(
        ...,
        description="The provider-specific unique identifier for the file (e.g., 'files/abc-123' for Gemini, 'file-xyz-456' for OpenAI)."
    )
    display_name: str = Field(
        ...,
        description="A user-friendly name for the file."
    )
    uri: str = Field(
        ...,
        description="The provider-specific URI used to reference the file in an API call."
    )
    # This field will be crucial for the OpenAI implementation but is defined here for consistency.
    local_path: Optional[str] = Field(
        None,
        description="The absolute local path to the file, if applicable. Used by providers that require local access for requests."
    )

class ToolCall(BaseModel):
    """A generic, provider-agnostic representation of a tool call requested by the model."""
    id: str = Field(
        ...,
        description="A unique identifier for this specific tool call instance. Required for mapping results back to the call. For OpenAI Responses API, this is the 'call_id'."
    )
    name: str = Field(
        ...,
        description="The name of the tool to be called."
    )
    args: Dict[str, Any] = Field(
        ...,
        description="The arguments for the tool, provided as a dictionary."
    )
    internal_id: Optional[str] = Field(
        None,
        description="Provider-internal ID for the tool call object itself (e.g., OpenAI's 'fc_...' ID). This is separate from the ID used to link results."
    )

class ContentPart(BaseModel):
    """
    A generic, provider-agnostic representation of a single part of a message.
    A message can be composed of multiple parts (e.g., text and an image).
    """
    type: Literal['text', 'image', 'audio', 'tool_call', 'tool_result'] = Field(
        ...,
        description="The type of content this part holds."
    )
    text: Optional[str] = Field(
        None,
        description="The text content of the part. Used for 'text' and contains the output for 'tool_result'."
    )
    file: Optional[FileObject] = Field(
        None,
        description="A reference to an uploaded file. Used for 'image' and 'audio' types."
    )
    tool_call: Optional[ToolCall] = Field(
        None,
        description="A tool call requested by the model. Used for the 'tool_call' type."
    )
    # For tool results, we need to link back to the original call
    tool_call_id: Optional[str] = Field(
        None,
        description="The ID of the tool call that this part is a result for. Used for the 'tool_result' type."
    )
    tool_name: Optional[str] = Field(
        None,
        description="The name of the tool that this part is a result for. Used for the 'tool_result' type."
    )

class Message(BaseModel):
    """A generic, provider-agnostic representation of a single message in a conversation."""
    role: Literal['user', 'model', 'tool'] = Field(
        ...,
        description="The role of the entity that sent this message."
    )
    parts: List[ContentPart] = Field(
        ...,
        min_length=1,
        description="A list of content parts that make up this message."
    )

class LLMResponse(BaseModel):
    """A structured, generic response from an LLM call, abstracting away provider specifics."""
    message: Optional[Message] = Field(
        None,
        description="The message returned by the model. Can be None if the call was blocked or failed."
    )
    finish_reason: str = Field(
        ...,
        description="The reason the model stopped generating content (e.g., 'stop', 'tool_calls', 'max_tokens')."
    )
    is_blocked: bool = Field(
        ...,
        description="A flag indicating if the response was blocked due to safety filters or other reasons."
    )
    raw_response: Any = Field(
        ...,
        description="The original, raw response object from the provider's API, for debugging purposes."
    )
    # --- MODIFIED: Add a field to carry the response ID for stateful APIs ---
    id: Optional[str] = Field(
        None,
        description="The unique ID of this specific response, if provided by a stateful API."
    )