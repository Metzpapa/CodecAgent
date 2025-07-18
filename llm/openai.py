# codec/llm/openai.py

import base64
import json
from typing import List, Any, Dict

# --- OpenAI specific imports ---
import openai
from openai.types.chat import ChatCompletion

# --- Local, provider-agnostic imports ---
from .base import LLMConnector
from .types import Message, LLMResponse, FileObject, ContentPart, ToolCall

# --- Forward reference for type hinting ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tools.base import BaseTool


class OpenAIConnector(LLMConnector):
    """
    A connector for the OpenAI API (e.g., GPT-4o).

    This class implements the LLMConnector interface and handles all the specific
    details of communicating with the OpenAI API, including type translation,
    API calls, and response parsing.
    """

    def _initialize_client(self) -> openai.OpenAI:
        """Initializes and returns the OpenAI client."""
        return openai.OpenAI(api_key=self.api_key)

    def generate_content(
        self,
        history: List[Message],
        tools: List['BaseTool'],
        system_prompt: str
    ) -> LLMResponse:
        """
        Generates content using the OpenAI API.
        """
        try:
            # 1. Translate our generic tool definitions to the OpenAI tool format.
            openai_tools = [self._tool_to_openai_tool(t) for t in tools]

            # 2. Translate our generic message history to the OpenAI message format.
            openai_messages = self._messages_to_openai_messages(history, system_prompt)

            # 3. Make the API call.
            print("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            # 4. Translate the OpenAI response back to our generic LLMResponse.
            return self._openai_response_to_llm_response(response)

        except openai.APIError as e:
            # Handle API errors, such as content policy violations, by returning a blocked response.
            print(f"❌ OpenAI API Error: {e}")
            return LLMResponse(
                message=None,
                finish_reason="API_ERROR",
                is_blocked=True,
                raw_response={"error": str(e), "status_code": e.status_code}
            )
        except Exception as e:
            print(f"❌ Unexpected Error during OpenAI call: {e}")
            return LLMResponse(
                message=None,
                finish_reason="UNEXPECTED_ERROR",
                is_blocked=True,
                raw_response={"error": str(e)}
            )

    def upload_file(self, file_path: str, mime_type: str, display_name: str) -> FileObject:
        """
        "Uploads" a file for use with OpenAI's vision models.

        Unlike Gemini, OpenAI's chat completion API doesn't use a file store. Instead,
        it accepts images as base64-encoded data URIs directly in the prompt. This
        method adapts to that by encoding the file and storing the data URI in our
        generic FileObject, so the tool logic remains consistent.
        """
        print(f"Encoding '{display_name}' for OpenAI...")
        try:
            with open(file_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
            data_uri = f"data:{mime_type};base64,{encoded_string}"
            
            # We use the data URI as both the ID and the URI for consistency.
            return FileObject(
                id=data_uri,
                display_name=display_name,
                uri=data_uri,
                local_path=file_path
            )
        except Exception as e:
            # This is not a real API call, so we re-raise to signal a local file error.
            raise IOError(f"Failed to read and encode file '{file_path}': {e}") from e

    def delete_file(self, file_id: str):
        """
        Deletes a file. For OpenAI, this is a no-op since files are sent with each
        request and not stored remotely. This method exists to satisfy the interface.
        """
        # No remote file to delete for OpenAI's chat completion vision model.
        # The file_id is the data URI, which is ephemeral.
        print(f"  - No remote deletion needed for OpenAI file '{file_id[:50]}...'")
        pass

    # ==============================================================================
    # == PRIVATE TRANSLATION METHODS ===============================================
    # ==============================================================================

    def _tool_to_openai_tool(self, tool: 'BaseTool') -> Dict[str, Any]:
        """Converts a generic BaseTool into the OpenAI function tool format."""
        # OpenAI's tool format is very close to standard JSON Schema.
        # We can directly use the Pydantic model's schema.
        schema = tool.args_schema.model_json_schema()
        # Pydantic 2.x may include a 'title' in the schema, which OpenAI doesn't use.
        schema.pop('title', None)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            }
        }

    def _messages_to_openai_messages(self, messages: List[Message], system_prompt: str) -> List[Dict[str, Any]]:
        """Translates a list of generic Messages into the OpenAI message format."""
        openai_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            if msg.role == 'model':
                # Model messages can contain text content and/or tool calls.
                model_msg = {"role": "assistant", "content": None}
                tool_calls = []
                text_content = []
                for part in msg.parts:
                    if part.type == 'text' and part.text:
                        text_content.append(part.text)
                    elif part.type == 'tool_call':
                        tool_calls.append({
                            "id": part.tool_call.id,
                            "type": "function",
                            "function": {
                                "name": part.tool_call.name,
                                "arguments": json.dumps(part.tool_call.args)
                            }
                        })
                
                if text_content:
                    model_msg["content"] = "\n".join(text_content)
                if tool_calls:
                    model_msg["tool_calls"] = tool_calls
                
                # Only add the message if it has content or tool calls
                if model_msg["content"] or model_msg.get("tool_calls"):
                    openai_messages.append(model_msg)

            elif msg.role == 'user':
                # User messages can be simple text or multimodal (text + image/audio).
                content_parts = []
                for part in msg.parts:
                    if part.type == 'text':
                        content_parts.append({"type": "text", "text": part.text})
                    elif part.type == 'image':
                        # The FileObject's URI is the base64 data URI we created in upload_file.
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": part.file.uri}
                        })
                    # Note: Audio would be handled similarly if needed.
                
                openai_messages.append({"role": "user", "content": content_parts})

            elif msg.role == 'tool':
                # Tool messages are responses from our tools back to the model.
                for part in msg.parts:
                    if part.type == 'tool_result':
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": part.text
                        })
        
        return openai_messages

    def _openai_response_to_llm_response(self, openai_response: ChatCompletion) -> LLMResponse:
        """Parses a raw OpenAI API response into our generic LLMResponse."""
        choice = openai_response.choices[0]
        finish_reason = choice.finish_reason
        response_message = choice.message

        response_parts: List[ContentPart] = []

        # 1. Check for text content
        if response_message.content:
            response_parts.append(ContentPart(type='text', text=response_message.content))

        # 2. Check for tool calls
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # Handle cases where the model generates invalid JSON for arguments
                    args = {"error": "invalid JSON arguments", "raw": tc.function.arguments}

                tool_call = ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args
                )
                response_parts.append(ContentPart(type='tool_call', tool_call=tool_call))

        # Ensure there's always at least one part, even if empty.
        if not response_parts:
            response_parts.append(ContentPart(type='text', text=""))

        message = Message(role="model", parts=response_parts)

        return LLMResponse(
            message=message,
            finish_reason=finish_reason,
            is_blocked=False, # OpenAI uses API errors for blocking, not finish_reason
            raw_response=openai_response
        )