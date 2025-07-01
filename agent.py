import os
import inspect
import pkgutil
import importlib
from typing import Dict

from google import genai
from google.genai import types
from pydantic import BaseModel

# Local imports
import tools
from state import State
from tools.base import BaseTool

class Agent:
    """
    The core AI agent responsible for orchestrating LLM calls and tool execution.
    """

    def __init__(self, state: State):
        """
        Initializes the agent.

        Args:
            state: The state object that holds the session's context.
        """
        self.state = state

        # As per the documentation, initialize the client.
        # It will automatically use the GOOGLE_API_KEY environment variable.
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # We will use a model that supports function calling well.
        self.model_name = "gemini-2.0-flash-001"

        print("Loading tools...")
        self.tools = self._load_tools()
        function_declarations = [tool.to_google_tool() for tool in self.tools.values()]
        self.google_tools = [types.Tool(function_declarations=function_declarations)]
        print(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory.
        """
        loaded_tools = {}
        # Use pkgutil to find all modules in the 'tools' package
        for _, module_name, _ in pkgutil.iter_modules(tools.__path__, tools.__name__ + "."):
            # Dynamically import all modules except the base class
            if not module_name.endswith(".base"):
                module = importlib.import_module(module_name)
                # Find all classes in the module that are subclasses of BaseTool
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if issubclass(cls, BaseTool) and cls is not BaseTool:
                        tool_instance = cls()
                        loaded_tools[tool_instance.name] = tool_instance
        return loaded_tools

    def run(self, prompt: str):
        """
        Starts the agent's execution loop for a single turn of conversation.

        Args:
            prompt: The user's request to the agent for this turn.
        """
        print("\n--- User Prompt ---")
        print(prompt)
        print("-------------------\n")

        # Add the initial user prompt to the conversation history.
        self.state.history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        while True:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.state.history,
                config=types.GenerateContentConfig(tools=self.google_tools),
            )

            # The full response from the model is always added to history for context.
            model_response_content = response.candidates[0].content
            self.state.history.append(model_response_content)

            # Separate the parts into text for display and function calls for execution.
            text_parts = [part.text for part in model_response_content.parts if part.text]
            function_calls = [
                part.function_call
                for part in model_response_content.parts
                if part.function_call
            ]

            # 1. Display the agent's reasoning (text parts) if any exist.
            if text_parts:
                print(f"🤖 Agent says: {''.join(text_parts)}")

            # 2. Decide if we need to execute tools.
            if not function_calls:
                # If there are no function calls, the agent's turn is over.
                print("\n✅ Agent has finished its turn.")
                break

            # --- Tool Execution Step ---
            # If we are here, it means there is at least one function call to execute.
            tool_results = []
            for func_call in function_calls:
                tool_name = func_call.name
                tool_args = dict(func_call.args)

                print(f"🤖 Agent wants to call tool: {tool_name}({tool_args})")

                tool_to_execute = self.tools.get(tool_name)
                if not tool_to_execute:
                    result = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        validated_args = tool_to_execute.args_schema(**tool_args)
                        result = tool_to_execute.execute(self.state, validated_args)
                    except Exception as e:
                        result = f"Error executing tool '{tool_name}': {e}"

                print(f"🛠️ Tool Result:\n{result}\n")

                # Collect the result to send back to the model.
                tool_results.append(types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                ))

            # Send all tool results back to the model in a single turn.
            self.state.history.append(types.Content(role="tool", parts=tool_results))
            # The loop continues, sending the tool results back to the model for the next step.