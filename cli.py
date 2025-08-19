# cli.py

import os
import uuid
import shutil
import logging
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
from codec.tools.finish_job import JobFinishedException

# --- Configuration ---
load_dotenv()
SAMPLE_PROJECT_PATH = os.environ.get("SAMPLE_PROJECT_PATH")
JOBS_BASE_DIR = Path("codec_jobs")

# --- Setup Logging to Console ---
# This will make the AgentContextLogger print beautifully to the terminal
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
        console.print("\n[bold magenta]Welcome to the Codec Agent CLI (Batch Mode).[/bold magenta]")
        console.print("Each prompt will run to completion. Type 'exit' or 'quit' to end the session.")
        
        while True:
            prompt = Prompt.ask("[bold yellow]You[/bold yellow]")
            if prompt.lower() in ['exit', 'quit']:
                break

            try:
                # Call run_to_completion, which will run the agent in a loop until it calls finish_job
                agent.run_to_completion(prompt)

            except JobFinishedException as e:
                # Handle the exception as the end of a SINGLE JOB.
                # Instead of breaking the loop, we print the result and continue,
                # allowing for the next prompt.
                console.print("\n[bold green]JOB FINISHED[/bold green]")
                console.print(f"Final Message: {e.result.get('message')}")
                
                # --- MODIFICATION: Handle a list of output paths (attachments) ---
                output_paths = e.result.get('output_paths')
                if output_paths:
                    console.print("Attachments:")
                    for path in output_paths:
                        console.print(f"  - {path}")
                # --- END MODIFICATION ---

                console.print("-" * 50)
                # The loop continues, waiting for the next prompt...

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