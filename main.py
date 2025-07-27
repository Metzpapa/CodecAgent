# codec/main.py

import os
import sys
import warnings
from dotenv import load_dotenv

# --- CONFIGURE WARNINGS FIRST ---
# This ensures the filter is active before any other modules that might
# trigger the warning are imported.
warnings.filterwarnings(
    "ignore",
    message="there are non-text parts in the response",
    category=UserWarning
)
# --------------------------------

from agent import Agent
from state import State


def check_api_key():
    """
    Checks if the necessary API key for the configured LLM provider is set.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            print("❌ Error: LLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not set.")
            print("Please add it to your .env file.")
            sys.exit(1)
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set.")
            print("Please add it to your .env file.")
            sys.exit(1)
        
        # --- MODIFIED: Removed S3 validation logic ---
        # The new OpenAIResponsesAPIConnector uses the OpenAI Files API, so S3 is no longer needed.
        # The S3-related environment variable checks have been removed.

    else:
        print(f"❌ Error: Unsupported LLM_PROVIDER '{provider}'. Please use 'gemini' or 'openai'.")
        sys.exit(1)


def print_startup_screen():
    """Displays a visually distinct welcome message for the CLI tool."""
    print("=" * 60)
    print("🎬 Welcome to Codec - The AI Video Editing Agent 🎬")
    print("=" * 60)
    print("Type 'exit' or 'quit' at any time to end the session.\n")


def get_assets_directory() -> str:
    """
    Gets the assets directory path. It first checks for the CODEC_ASSETS_DIR
    environment variable. If not found or invalid, it falls back to prompting the user.
    """
    env_dir = os.getenv("CODEC_ASSETS_DIR")
    if env_dir:
        if os.path.isdir(env_dir):
            print(f"✅ Using default assets directory from environment: {env_dir}\n")
            return env_dir
        else:
            print(f"⚠️ Warning: CODEC_ASSETS_DIR is set to '{env_dir}', but this is not a valid directory.")
            print("Please provide a valid path below.")

    # Fallback to interactive prompt
    while True:
        assets_dir = input("➡️  Enter the path to your assets directory: ").strip()
        if os.path.isdir(assets_dir):
            print(f"✅ Assets directory found: {assets_dir}\n")
            return assets_dir
        else:
            print(f"❌ Error: Directory not found at '{assets_dir}'. Please try again.")


def get_initial_prompt() -> str:
    """
    Prompts the user for the initial multi-line prompt.
    """
    print("➡️  Enter your initial editing instructions below.")
    print("   (You can write multiple lines. Press Enter on an empty line to send.)")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:  # Handle Ctrl+D as a way to end input
            break
    return "\n".join(lines).strip()


def main():
    """The main entry point and orchestration logic for the application."""
    # --- One-Time Setup ---
    load_dotenv()
    check_api_key() # This function is now provider-aware
    print_startup_screen()

    assets_directory_input = get_assets_directory()

    absolute_assets_directory = os.path.abspath(assets_directory_input)
    print(f"✅ Using absolute path for assets: {absolute_assets_directory}\n")

    # The Agent's __init__ now handles selecting the correct connector.
    session_state = State(assets_directory=absolute_assets_directory)
    video_agent = Agent(state=session_state)

    # --- Main Conversation Loop ---
    try:
        prompt = get_initial_prompt()
        session_state.initial_prompt = prompt
        while True:
            if not prompt:
                prompt = input("➡️  You: ").strip()
                continue

            if prompt.lower().strip() in ["exit", "quit"]:
                break

            video_agent.run(prompt=prompt)
            prompt = input("\n➡️  You: ").strip()

    finally:
        # The cleanup block now uses the generic connector, which works for all providers.
        print("\nCleaning up session resources...")
        if session_state.uploaded_files:
            print(f"Deleting {len(session_state.uploaded_files)} uploaded files...")
            # `f` is our generic FileObject from llm.types
            for f in session_state.uploaded_files:
                try:
                    # We call the delete_file method on the agent's connector,
                    # passing the provider-specific ID from our generic FileObject.
                    video_agent.connector.delete_file(file_id=f.id)
                    # The connector's implementation handles the specifics (OpenAI Files API, Gemini API, or no-op).
                except Exception as e:
                    # Log if a specific file fails to delete, but continue trying others
                    print(f"  - Failed to delete {f.id}: {e}")
        else:
            print("No uploaded files to clean up.")
        
        print("\n👋 Goodbye! Session ended.")


if __name__ == "__main__":
    main()