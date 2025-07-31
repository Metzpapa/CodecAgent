# codec/tools/list_assets.py

import os
# --- MODIFIED: Update TYPE_CHECKING imports for the new interface ---
from typing import TYPE_CHECKING
import openai

from tools.base import BaseTool, NoOpArgs

if TYPE_CHECKING:
    from state import State


class ListAssetsTool(BaseTool):
    """A tool to list all available files in the assets directory."""

    @property
    def name(self) -> str:
        return "list_assets"

    @property
    def description(self) -> str:
        return "Lists all available asset files (videos, images, audio) in the provided assets directory and its subdirectories. Use this to see what files you can work with. If you are ever confused on where to start, or don't know how complete a task, start here and coninue as necessary. "

    @property
    def args_schema(self):
        return NoOpArgs
    def execute(self, state: 'State', args: NoOpArgs, client: openai.OpenAI) -> str:
        """
        Scans the assets directory and all its subdirectories, returning a
        list of all found files, ignoring hidden system files.

        Note: The `connector` argument is unused in this specific tool, but it is
        a required part of the BaseTool interface for consistency.
        """
        assets_dir = state.assets_directory
        found_files = []

        if not os.path.isdir(assets_dir):
            return f"Error: The assets directory '{assets_dir}' was not found."

        try:
            # The core logic of this tool does not depend on the LLM provider,
            # so it remains completely unchanged.
            for root, dirs, files in os.walk(assets_dir):
                for filename in files:
                    # Ignore hidden files (like .DS_Store)
                    if filename.startswith('.'):
                        continue

                    # Get the full path of the file
                    full_path = os.path.join(root, filename)
                    # Get the path relative to the main assets directory for a cleaner output
                    relative_path = os.path.relpath(full_path, assets_dir)
                    found_files.append(relative_path)

            if not found_files:
                return "No asset files found in the directory or its subdirectories."

            return "Here are the available assets:\n" + "\n".join(sorted(found_files))

        except Exception as e:
            return f"An error occurred while trying to list assets: {e}"