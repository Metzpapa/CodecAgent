# codec/llm/openai.py

import os
import uuid
import json
from pathlib import Path
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
    API calls, and response parsing. It now uploads files directly to OpenAI
    and references them by file_id to improve performance and avoid timeouts.
    """

    def _initialize_client(self) -> openai.OpenAI:
        """Initializes the OpenAI client."""
        print("🤖 Using OpenAI provider with direct file uploads.")
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
        Uploads a file directly to OpenAI's service for use in multimodal prompts.
        """
        print(f"Uploading '{display_name}' to OpenAI...")
        try:
            with open(file_path, "rb") as f:
                # Use purpose="user_data" as recommended for chat completions.
                uploaded_file = self.client.files.create(
                    file=f,
                    purpose="user_data"
                )
            print(f"Upload complete. File ID: {uploaded_file.id}")

            # The file ID is the unique identifier and also serves as the URI for our purposes.
            return FileObject(
                id=uploaded_file.id,
                display_name=display_name,
                uri=uploaded_file.id, # Using the ID as the URI for internal consistency
                local_path=file_path
            )
        except Exception as e:
            # This is a real API call, so we should handle the error gracefully.
            # Re-raising as IOError to signal a file-related failure upstream.
            raise IOError(f"Failed to upload file '{file_path}' to OpenAI: {e}") from e

    def delete_file(self, file_id: str):
        """
        Deletes a previously uploaded file from OpenAI's service.
        """
        print(f"  - Deleting OpenAI file: {file_id}")
        try:
            # The file_id is the ID returned by the upload_file method.
            self.client.files.delete(file_id=file_id)
            print(f"  - Successfully deleted OpenAI file {file_id}")
        except Exception as e:
            # Fail gracefully so that cleanup can continue with other files.
            print(f"  - Failed to delete OpenAI file {file_id}: {e}")

    # ==============================================================================
    # == PRIVATE TRANSLATION METHODS ===============================================
    # ==============================================================================

    def _tool_to_openai_tool(self, tool: 'BaseTool') -> Dict[str, Any]:
        """Converts a generic BaseTool into the OpenAI function tool format."""
        schema = tool.args_schema.model_json_schema()
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
                
                if model_msg["content"] or model_msg.get("tool_calls"):
                    openai_messages.append(model_msg)

            elif msg.role == 'user':
                content_parts = []
                for part in msg.parts:
                    if part.type == 'text':
                        content_parts.append({"type": "text", "text": part.text})
                    elif part.type == 'image':
                        # --- FIX: Use 'file' type instead of 'image_file' ---
                        # The API error indicates 'file' is the correct type for
                        # referencing uploaded content by ID.
                        content_parts.append({
                            "type": "file",
                            "file": {
                                "file_id": part.file.id
                            }
                        })
                
                openai_messages.append({"role": "user", "content": content_parts})

            elif msg.role == 'tool':
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

        if response_message.content:
            response_parts.append(ContentPart(type='text', text=response_message.content))

        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"error": "invalid JSON arguments", "raw": tc.function.arguments}

                tool_call = ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args
                )
                response_parts.append(ContentPart(type='tool_call', tool_call=tool_call))

        if not response_parts:
            response_parts.append(ContentPart(type='text', text=""))

        message = Message(role="model", parts=response_parts)

        return LLMResponse(
            message=message,
            finish_reason=finish_reason,
            is_blocked=False,
            raw_response=openai_response
        )