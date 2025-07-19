# codec/llm/openai.py

import os
import uuid
import base64
import json
from pathlib import Path
from typing import List, Any, Dict

# --- OpenAI specific imports ---
import openai
from openai.types.chat import ChatCompletion

# --- Local, provider-agnostic imports ---
from .base import LLMConnector
from .types import Message, LLMResponse, FileObject, ContentPart, ToolCall
from s3utils import S3Uploader # <-- IMPORT THE S3 UTILITY

# --- Forward reference for type hinting ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tools.base import BaseTool


class OpenAIConnector(LLMConnector):
    """
    A connector for the OpenAI API (e.g., GPT-4o).

    This class implements the LLMConnector interface and handles all the specific
    details of communicating with the OpenAI API, including type translation,
    API calls, and response parsing. It now supports uploading images to an
    S3-compatible service instead of sending them as base64.
    """

    def _initialize_client(self) -> openai.OpenAI:
        """Initializes the OpenAI client and the S3 uploader if configured."""
        # Initialize S3 Uploader if environment variables are set
        self.s3_uploader = None
        # Check for all required S3 variables, including the new public URL base
        if os.getenv("S3_ENDPOINT_URL") and os.getenv("S3_PUBLIC_URL_BASE"):
            print("🤖 S3 Uploader configured for OpenAI.")
            self.s3_uploader = S3Uploader(
                endpoint_url=os.environ["S3_ENDPOINT_URL"],
                access_key=os.environ["S3_ACCESS_KEY_ID"],
                secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                bucket_name=os.environ["S3_BUCKET_NAME"],
                # Pass the new, explicit public URL base to the uploader
                public_url_base=os.environ["S3_PUBLIC_URL_BASE"]
            )
        else:
            # This maintains the old base64 behavior if S3 is not configured
            print("⚠️  S3 not configured for OpenAI. Falling back to base64 encoding for images.")

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
        Uploads a file for use with OpenAI.
        If S3 is configured, it uploads to a public bucket and returns the URL.
        Otherwise, it falls back to base64 encoding.
        """
        # --- S3 LOGIC ---
        if self.s3_uploader:
            file_extension = Path(file_path).suffix
            # Create a unique name to avoid collisions in the bucket
            object_name = f"frames/{uuid.uuid4().hex}{file_extension}"
            
            public_url = self.s3_uploader.upload(file_path, object_name)
            
            # The ID is the S3 object name (for deletion), and the URI is the public URL
            return FileObject(
                id=object_name,
                display_name=display_name,
                uri=public_url,
                local_path=file_path
            )

        # --- FALLBACK BASE64 LOGIC ---
        print(f"Encoding '{display_name}' for OpenAI (S3 not configured)...")
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
        Deletes a file. If S3 is used, it deletes the object from the bucket.
        Otherwise, it's a no-op for base64.
        """
        # --- S3 LOGIC ---
        if self.s3_uploader:
            # The file_id is the S3 object_name.
            # We check for a prefix to ensure we only try to delete S3 objects,
            # not a base64 data URI if the fallback was used during the session.
            if file_id.startswith("frames/"):
                self.s3_uploader.delete(object_name=file_id)
            else:
                 print(f"  - Skipping non-S3 file deletion: '{file_id[:50]}...'")
        else:
            # --- FALLBACK NO-OP ---
            print(f"  - No remote deletion needed for base64 file '{file_id[:50]}...'")
        pass

    # ==============================================================================
    # == PRIVATE TRANSLATION METHODS (UNCHANGED) ===================================
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
                        # The FileObject's URI is now either a public S3 URL
                        # or the base64 data URI. This logic works for both.
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": part.file.uri}
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