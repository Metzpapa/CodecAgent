# cli.py

import os
import uuid
import shutil
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Use a rich console for beautiful, color-coded output
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

# Import the Brain components
from codec.agent import Agent
from codec.state import State
from codec.agent_logging import AgentContextLogger

# --- Configuration ---
load_dotenv()
SAMPLE_PROJECT_PATH = os.environ.get("SAMPLE_PROJECT_PATH")
JOBS_BASE_DIR = Path("codec_jobs")

# --- Setup Logging to Console ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=False,
        show_level=False
    )]
)
log = logging.getLogger("rich")
console = Console()

# --- Main CLI Function ---
def run_cli():
    """
    Sets up a temporary environment and runs an interactive chat session
    with the Codec Agent.
    """
    if not SAMPLE_PROJECT_PATH or not Path(SAMPLE_PROJECT_PATH).is_dir():
        console.print(f"[bold red]Error:[/bold red] SAMPLE_PROJECT_PATH is not set in your .env file or the directory does not exist.")
        console.print(f"Please add `SAMPLE_PROJECT_PATH=/path/to/your/sample/assets` to your .env file.")
        return

    # 1. Setup the temporary job environment
    job_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_BASE_DIR / job_id
    assets_dir = job_dir / "assets"
    output_dir = job_dir / "output"
    
    session_cleanup_path = job_dir

    try:
        console.print(f"[cyan]Setting up new session: {job_id}[/cyan]")
        shutil.copytree(SAMPLE_PROJECT_PATH, assets_dir)
        output_dir.mkdir(exist_ok=True)
        console.print(f"[green]Copied sample assets from '{SAMPLE_PROJECT_PATH}' to '{assets_dir}'[/green]")

        # 2. Initialize the Brain (This happens only ONCE per session)
        state = State(assets_directory=str(assets_dir))
        context_logger = AgentContextLogger(job_id=job_id, stream_logger=log)
        agent = Agent(state=state, context_logger=context_logger)

        # 3. The Interaction Loop
        console.print("\n[bold magenta]Welcome to the Codec Agent CLI (Interactive Mode).[/bold magenta]")
        console.print("This is a continuous conversation. Type 'exit' or 'quit' to end the session.")
        
        while True:
            prompt = Prompt.ask("\n[bold yellow]You[/bold yellow]")
            if prompt.lower() in ['exit', 'quit']:
                break

            try:
                agent_response = agent.process_turn(prompt)

                # --- FIX: Only process the response if the agent actually said something ---
                if agent_response:
                    # Print the agent's text response to the console
                    console.print(f"\n[bold cyan]Agent[/bold cyan]: {agent_response}")

                    # Parse the response for file citations
                    referenced_files = re.findall(r'\[([\w\.\-\_]+)\]', agent_response)
                    if referenced_files:
                        console.print("\n[bold green]Referenced Files:[/bold green]")
                        for filename in referenced_files:
                            # Construct the full, absolute path for the developer
                            full_path = (output_dir / filename).resolve()
                            # Use rich's link markup to make it clickable in compatible terminals
                            console.print(f"  - [link=file://{full_path}]{full_path}[/link]")
                        console.print("-" * 50)
                # If agent_response is None, we simply do nothing and wait for the next user prompt.

            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred in the agent run:[/bold red]")
                log.exception(e) # Print full, rich-formatted traceback

    finally:
        # 4. Cleanup
        if 'session_cleanup_path' in locals() and session_cleanup_path.exists():
            console.print(f"\n[cyan]Cleaning up session directory: {session_cleanup_path}[/cyan]")
            shutil.rmtree(session_cleanup_path)
        console.print("[bold magenta]Session ended.[/bold magenta]")


if __name__ == "__main__":
    run_cli()