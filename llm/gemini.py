# codec/llm/gemini.py

import os
import uuid
from typing import List, Any, Dict

# --- Google GenAI specific imports ---
from google import genai
from google.genai import types

# --- Local, provider-agnostic imports ---
from .base import LLMConnector
from .types import Message, LLMResponse, FileObject, ContentPart, ToolCall

# --- Forward reference for type hinting ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tools.base import BaseTool


# ==============================================================================
# == CONSTANTS AND HELPERS REPLICATED FROM THE ORIGINAL agent.py ================
# ==============================================================================

# Alias for brevity, improving readability.
FINISH = types.FinishReason

# Define the set of finish reasons that indicate a blocked or incomplete response.
# This is a robust way to handle various failure modes from the API.
BLOCKED_FINISH_REASONS = {
    FINISH.SAFETY,
    FINISH.RECITATION,
    FINISH.BLOCKLIST,
    FINISH.PROHIBITED_CONTENT,
    FINISH.SPII,
    FINISH.IMAGE_SAFETY,
    FINISH.MALFORMED_FUNCTION_CALL,
    FINISH.OTHER,
    FINISH.LANGUAGE,
}
# MAX_TOKENS is intentionally *not* included, as we often want to process partial output.


class GeminiConnector(LLMConnector):
    """
    A connector for the Google Gemini API.

    This class implements the LLMConnector interface and handles all the specific
    details of communicating with the Gemini API, including type translation,
    API calls, and response parsing.
    """

    def _initialize_client(self) -> genai.Client:
        """Initializes and returns the Google GenAI client."""
        return genai.Client(api_key=self.api_key)

    def generate_content(
        self,
        history: List[Message],
        tools: List['BaseTool'],
        system_prompt: str
    ) -> LLMResponse:
        """
        Generates content using the Gemini API, replicating the logic from the original agent.
        """
        # --- 1. Translate generic inputs to Gemini-specific formats ---
        gemini_history = self._messages_to_gemini_content(history)
        function_declarations = [tool.to_google_tool() for tool in tools]
        gemini_tool_set = types.Tool(function_declarations=function_declarations)

        config = types.GenerateContentConfig(
            tools=[gemini_tool_set],
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )

        # --- 2. Make the API call ---
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_history,
            config=config,
        )

        # --- 3. Translate Gemini-specific response to our generic LLMResponse ---
        return self._gemini_response_to_llm_response(response)

    def upload_file(self, file_path: str, mime_type: str, display_name: str) -> FileObject:
        """Uploads a file to the Gemini API and returns a generic FileObject."""
        print(f"Uploading '{display_name}' to Google Gemini...")
        with open(file_path, "rb") as f:
            uploaded_file = self.client.files.upload(
                file=f,
                config={"mimeType": mime_type, "displayName": display_name}
            )
        print(f"Upload complete. Name: {uploaded_file.name}")

        # Translate the provider-specific response to our generic FileObject
        return FileObject(
            id=uploaded_file.name,
            display_name=uploaded_file.display_name,
            uri=uploaded_file.uri,
            local_path=file_path # Store the local path for potential future use
        )

    def delete_file(self, file_id: str):
        """Deletes a file from the Gemini service using its unique ID."""
        try:
            self.client.files.delete(name=file_id)
            print(f"  - Deleted Gemini file {file_id}")
        except Exception as e:
            print(f"  - Failed to delete Gemini file {file_id}: {e}")

    # ==============================================================================
    # == PRIVATE TRANSLATION METHODS ===============================================
    # ==============================================================================

    def _messages_to_gemini_content(self, messages: List[Message]) -> List[types.Content]:
        """Translates a list of generic Messages into a list of Gemini Content objects."""
        gemini_content_list = []
        for msg in messages:
            gemini_parts = []
            for part in msg.parts:
                if part.type == 'text':
                    gemini_parts.append(types.Part.from_text(part.text))
                elif part.type in ['image', 'audio']:
                    # Determine mime_type for the Part object. This mirrors the original
                    # tool's hardcoded values.
                    mime_type = 'image/jpeg' if part.type == 'image' else 'audio/mpeg'
                    gemini_parts.append(types.Part.from_uri(
                        file_uri=part.file.uri,
                        mime_type=mime_type
                    ))
                elif part.type == 'tool_result':
                    # The original code wrapped results in a dict. We replicate that here.
                    response_dict = {"result": part.text}
                    gemini_parts.append(types.Part.from_function_response(
                        name=part.tool_name,
                        response=response_dict
                    ))
                # 'tool_call' is a model output, so it's not part of the input history in this direction.

            gemini_content_list.append(types.Content(role=msg.role, parts=gemini_parts))
        return gemini_content_list

    def _gemini_response_to_llm_response(self, gemini_response: types.GenerateContentResponse) -> LLMResponse:
        """
        Parses a raw Gemini API response into our generic LLMResponse,
        including all the original validation and error handling logic.
        """
        # --- BULLET-PROOF RESPONSE VALIDATION (from agent.py) ---
        if gemini_response.prompt_feedback and gemini_response.prompt_feedback.block_reason:
            print(f"❌ The prompt was blocked. Reason: {gemini_response.prompt_feedback.block_reason.name}")
            return LLMResponse(
                message=None,
                finish_reason=gemini_response.prompt_feedback.block_reason.name,
                is_blocked=True,
                raw_response=gemini_response
            )

        if not gemini_response.candidates:
            print("🤖 Agent did not return a candidate. This might be due to a safety filter or other issue.")
            return LLMResponse(
                message=None,
                finish_reason="NO_CANDIDATES",
                is_blocked=True,
                raw_response=gemini_response
            )

        candidate = gemini_response.candidates[0]
        finish_reason = candidate.finish_reason or FINISH.FINISH_REASON_UNSPECIFIED

        if finish_reason in BLOCKED_FINISH_REASONS:
            print(f"❌ Response was blocked by the model. Reason: {finish_reason.name}")
            return LLMResponse(
                message=None,
                finish_reason=finish_reason.name,
                is_blocked=True,
                raw_response=gemini_response
            )

        # --- Translate the valid response into generic types ---
        model_content = candidate.content
        response_parts: List[ContentPart] = []

        # The original agent printed thoughts and text separately. We'll combine them
        # into text parts, as the 'thought' is just a special kind of text output.
        if model_content and model_content.parts:
            for part in model_content.parts:
                if part.text:
                    response_parts.append(ContentPart(type='text', text=part.text))

        # Gemini returns tool calls at the top level of the response, not inside the content parts.
        if gemini_response.function_calls:
            for fc in gemini_response.function_calls:
                tool_call = ToolCall(
                    # Gemini doesn't provide a call ID, so we generate one for consistency.
                    id=f"call_{uuid.uuid4().hex}",
                    name=fc.name,
                    args=dict(fc.args)
                )
                response_parts.append(ContentPart(type='tool_call', tool_call=tool_call))

        # If there are no parts (e.g., only a finish reason), we still need a valid message.
        if not response_parts:
             # This can happen if the model finishes without saying anything or calling a tool.
             # We create an empty text part to ensure the message is valid.
             response_parts.append(ContentPart(type='text', text=""))


        message = Message(role="model", parts=response_parts)

        return LLMResponse(
            message=message,
            finish_reason=finish_reason.name,
            is_blocked=False,
            raw_response=gemini_response
        )