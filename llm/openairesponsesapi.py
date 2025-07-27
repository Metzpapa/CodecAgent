# codec/llm/openairesponsesapi.py

import os
import json
from pathlib import Path
from typing import List, Any, Dict, Optional

# --- OpenAI specific imports ---
import openai

# --- Local, provider-agnostic imports ---
from .base import LLMConnector
from .types import Message, LLMResponse, FileObject, ContentPart, ToolCall

# --- Forward reference for type hinting ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tools.base import BaseTool


class OpenAIResponsesAPIConnector(LLMConnector):
    """
    A connector for the new OpenAI Responses API (e.g., for GPT-4.1, o4-mini).

    This class implements the LLMConnector interface and handles all the specific
    details of communicating with the OpenAI Responses API, including type translation,
    API calls, and response parsing. It uses the Files API for multimodal inputs,
    avoiding URL length limits and timeouts.
    """

    def _initialize_client(self) -> openai.OpenAI:
        """Initializes the OpenAI client."""
        print("🤖 Using OpenAI Responses API (Stateful). File uploads will use the OpenAI Files API.")
        return openai.OpenAI(api_key=self.api_key)

    def generate_content(
        self,
        history: List[Message],
        tools: List['BaseTool'],
        system_prompt: str,
        last_response_id: Optional[str]
    ) -> LLMResponse:
        """
        Generates content using the OpenAI Responses API, handling both the initial
        and subsequent turns of a stateful conversation.
        """
        try:
            openai_tools = [self._tool_to_openai_tool(t) for t in tools]
            
            if last_response_id is None:
                # FIRST turn: Send full history and instructions.
                print("Sending initial request to OpenAI Responses API...")
                openai_input = self._history_to_responses_input(history)
                
                response = self.client.responses.create(
                    model=self.model_name,
                    instructions=system_prompt,
                    input=openai_input,
                    tools=openai_tools,
                )
            else:
                # SUBSEQUENT turns: Send only new messages and the previous response ID.
                print(f"Sending continuation request to OpenAI Responses API (ID: {last_response_id})...")
                
                # --- MODIFIED: Find all messages since the last model response ---
                # Find the index of the last message from the model.
                last_model_message_index = -1
                for i in range(len(history) - 1, -1, -1):
                    if history[i].role == 'model':
                        last_model_message_index = i
                        break
                
                # All messages after that are new.
                new_messages = history[last_model_message_index + 1:]
                
                if not new_messages:
                    # This is a safeguard, but this case shouldn't be hit in normal flow.
                    return LLMResponse(message=None, finish_reason="NO_NEW_CONTENT", is_blocked=True, raw_response={"error": "No new messages to send."})

                openai_input = self._history_to_responses_input(new_messages)

                response = self.client.responses.create(
                    model=self.model_name,
                    input=openai_input,
                    tools=openai_tools,
                    previous_response_id=last_response_id,
                )

            return self._openai_response_to_llm_response(response)

        except openai.APIError as e:
            print(f"❌ OpenAI API Error: {e}")
            raw_error = {"error": str(e)}
            if hasattr(e, 'status_code'):
                raw_error["status_code"] = e.status_code
            return LLMResponse(
                message=None,
                finish_reason="API_ERROR",
                is_blocked=True,
                raw_response=raw_error
            )
        except Exception as e:
            print(f"❌ Unexpected Error during OpenAI call: {e}")
            import traceback
            traceback.print_exc()
            return LLMResponse(
                message=None,
                finish_reason="UNEXPECTED_ERROR",
                is_blocked=True,
                raw_response={"error": str(e)}
            )

    def upload_file(self, file_path: str, mime_type: str, display_name: str) -> FileObject:
        """
        Uploads a file to the OpenAI Files API for use in vision prompts.
        """
        print(f"Uploading '{display_name}' to OpenAI Files API...")
        try:
            with open(file_path, "rb") as f:
                uploaded_file = self.client.files.create(file=f, purpose="vision")
            print(f"Upload complete. File ID: {uploaded_file.id}")

            return FileObject(
                id=uploaded_file.id,
                display_name=display_name,
                uri=f"file_id://{uploaded_file.id}",
                local_path=file_path
            )
        except Exception as e:
            raise IOError(f"Failed to upload file '{file_path}' to OpenAI: {e}") from e

    def delete_file(self, file_id: str):
        """
        Deletes a file from the OpenAI service using its unique file ID.
        """
        try:
            self.client.files.delete(file_id=file_id)
            print(f"  - Deleted OpenAI file {file_id}")
        except Exception as e:
            print(f"  - Failed to delete OpenAI file {file_id}: {e}")

    # ==============================================================================
    # == PRIVATE TRANSLATION METHODS (Unchanged from previous step) ================
    # ==============================================================================

    def _tool_to_openai_tool(self, tool: 'BaseTool') -> Dict[str, Any]:
        """Converts a generic BaseTool into the OpenAI Responses API function tool format."""
        schema = tool.args_schema.model_json_schema()
        schema.pop('title', None)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        }

    def _history_to_responses_input(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Translates a list of generic Messages into the OpenAI Responses API 'input' format."""
        responses_input = []
        for msg in messages:
            if msg.role == 'user':
                content_list = []
                for part in msg.parts:
                    if part.type == 'text' and part.text:
                        content_list.append({"type": "input_text", "text": part.text})
                    elif part.type == 'image':
                        content_list.append({"type": "input_image", "file_id": part.file.id})
                if content_list:
                    responses_input.append({"role": "user", "content": content_list})

            elif msg.role == 'model':
                text_parts = []
                for part in msg.parts:
                    if part.type == 'text' and part.text:
                        text_parts.append(part.text)
                    elif part.type == 'tool_call' and part.tool_call:
                        tool_call_obj = {
                            "type": "function_call",
                            "id": part.tool_call.internal_id or part.tool_call.id,
                            "call_id": part.tool_call.id,
                            "name": part.tool_call.name,
                            "arguments": json.dumps(part.tool_call.args)
                        }
                        responses_input.append(tool_call_obj)
                if text_parts:
                    full_text = "\n".join(text_parts)
                    responses_input.append({"role": "assistant", "content": full_text})

            elif msg.role == 'tool':
                for part in msg.parts:
                    if part.type == 'tool_result':
                        tool_output_obj = {
                            "type": "function_call_output",
                            "call_id": part.tool_call_id,
                            "output": part.text
                        }
                        responses_input.append(tool_output_obj)
        return responses_input

    def _openai_response_to_llm_response(self, openai_response: Any) -> LLMResponse:
        """Parses a raw OpenAI Responses API response into our generic LLMResponse."""
        response_parts: List[ContentPart] = []
        if not openai_response.output:
            response_parts.append(ContentPart(type='text', text=""))
        else:
            for item in openai_response.output:
                if item.type == 'message' and hasattr(item, 'content'):
                    text_content = "".join([c.text for c in item.content if hasattr(c, 'text')])
                    if text_content:
                        response_parts.append(ContentPart(type='text', text=text_content))
                elif item.type == 'function_call':
                    try:
                        args = json.loads(item.arguments)
                    except json.JSONDecodeError:
                        args = {"error": "invalid JSON arguments", "raw": item.arguments}
                    tool_call = ToolCall(
                        id=item.call_id,
                        internal_id=item.id,
                        name=item.name,
                        args=args
                    )
                    response_parts.append(ContentPart(type='tool_call', tool_call=tool_call))

        if not response_parts:
            response_parts.append(ContentPart(type='text', text=""))

        message = Message(role="model", parts=response_parts)
        finish_reason = getattr(openai_response, 'status', 'unknown')

        return LLMResponse(
            id=openai_response.id,
            message=message,
            finish_reason=finish_reason.upper(),
            is_blocked=False,
            raw_response=openai_response.to_dict() if hasattr(openai_response, 'to_dict') else str(openai_response)
        )