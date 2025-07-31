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
    Checks if the necessary OpenAI API key is set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY is not set.")
        print("Please add it to your .env file.")
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
        print("\nCleaning up session resources...")
        if session_state.uploaded_files:
            print(f"Deleting {len(session_state.uploaded_files)} uploaded files...")
            # session_state.uploaded_files is now a simple list of file ID strings.
            for file_id in session_state.uploaded_files:
                try:
                    # We make a direct, simple call to the OpenAI client.
                    print(f"  - Deleting OpenAI file: {file_id}")
                    video_agent.client.files.delete(file_id=file_id)
                except Exception as e:
                    # Log if a specific file fails to delete, but continue trying others
                    print(f"  - Failed to delete {file_id}: {e}")
        else:
            print("No uploaded files to clean up.")
        
        print("\n👋 Goodbye! Session ended.")


if __name__ == "__main__":
    main()