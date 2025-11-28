import os
import json
import argparse
import yaml
import base64
from datetime import datetime

import asyncio
import os
import traceback
from datetime import datetime

from oagi import AsyncScreenshotMaker
from oagi.agent.observer import AsyncAgentObserver
from oagi.agent.tasker import TaskerAgent
from oagi.handler import AsyncPyautoguiActionHandler


async def main():
    parser = argparse.ArgumentParser(description='Run Todo Agent v2 on a single JSON task')
    # cvs personal information
    parser.add_argument('--first_name', type=str, default='John', help='First name')
    parser.add_argument('--last_name', type=str, default='Doe', help='Last name')
    parser.add_argument('--email', type=str, default='john.doe@gmail.com', help='Email')
    parser.add_argument('--birthday', type=str, default='01-01-1990', help='Birthday')
    parser.add_argument('--zip_code', type=str, default='94103', help='Zip code')
    
    # run configurations
    parser.add_argument('--exp_name', type=str, default='cvs', help='Experiment name; all saves under results/exp_name')
    parser.add_argument('--save_dir', type=str, default='results/', help='Directory to save the results')

    # tasker configurations
    parser.add_argument('--model_name', type=str, default='sft-bigs1-1027-s2-1113-mixoc-1107', help='Model name')
    parser.add_argument('--max_steps', type=int, default=24, help='Max steps per todo')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')

    args = parser.parse_args()

    # save directory
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # initialize the tasker and environment
    observer = AsyncAgentObserver()
    image_provider = AsyncScreenshotMaker()
    action_handler = AsyncPyautoguiActionHandler()

    tasker = TaskerAgent(
        api_key=os.getenv("OAGI_API_KEY"),
        base_url=os.getenv("OAGI_BASE_URL", "https://api.agiopen.org:8081"),
        model=args.model_name,
        max_steps=args.max_steps,  # Max steps per todo
        temperature=args.temperature,
        step_observer=observer,
    )

    month, day, year = args.birthday.split('-')

    # -------- Define the Workflow --------
    instruction = f"Schedule an appointment at CVS for {args.first_name} {args.last_name} with email {args.email} and birthday {args.birthday}"
    todos = [
        f"Open a new tab, go to www.cvs.com, type 'flu shot' in the search bar and press enter, wait for the page to load, then click on the button of Schedule vaccinations on the top of the page",
        f"Enter the first name '{args.first_name}', last name '{args.last_name}', and email '{args.email}' in the form. Do not use any suggested autofills. Make sure the mobile phone number is empty.",
        f"Slightly scroll down to see the date of birth, enter Month '{month}', Day '{day}', and Year '{year}' in the form",
        f"Click on 'Continue as guest' button, wait for the page to load with wait, click on 'Select vaccines' button, select 'Flu', scroll down and click on 'Add vaccines'",
        f"Click on 'next' to enter the page with recommendation vaccines, then click on 'next' again, until on the page of entering zip code, enter '{args.zip_code}', select the first option from the dropdown menu, and click on 'Search'",
    ]

    # -------- Run the Tasker --------
    tasker.set_task(
        task=instruction,
        todos=todos,
    )

    print(f"Starting task execution at {datetime.now()}")
    print(f"Task: {instruction}")
    print(f"Number of todos: {len(todos)}")
    print("=" * 60)

    try:
        # Execute the task
        success = await tasker.execute(
            instruction="",
            action_handler=action_handler,
            image_provider=image_provider,
        )

        # Get final memory state
        memory = tasker.get_memory()

        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Overall success: {success}")
        print(f"\nTask execution summary:\n{memory.task_execution_summary}")

        # Print todo statuses
        print("\nTodo Status:")
        for i, todo in enumerate(memory.todos):
            status_icon = {
                "completed": "✅",
                "pending": "⏳",
                "in_progress": "🔄",
                "skipped": "⏭️",
            }.get(todo.status.value, "❓")
            print(f"  {status_icon} [{i + 1}] {todo.description} - {todo.status.value}")

        # Print execution statistics
        status_summary = memory.get_todo_status_summary()
        print("\nExecution Statistics:")
        print(f"  Completed: {status_summary.get('completed', 0)}")
        print(f"  Pending: {status_summary.get('pending', 0)}")
        print(f"  In Progress: {status_summary.get('in_progress', 0)}")
        print(f"  Skipped: {status_summary.get('skipped', 0)}")

        # Print execution history summary
        if memory.history:
            print(f"\nExecution History ({len(memory.history)} entries):")
            for hist in memory.history:
                print(f"  - Todo {hist.todo_index}: {hist.todo}")
                print(f"    Actions: {len(hist.actions)}, Completed: {hist.completed}")
                if hist.summary:
                    print(f"    Summary: {hist.summary[:100]}...")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        traceback.print_exc()

    # ---------- Export the Execution History ----------
    output_file = os.path.join(save_dir, f"cvs_execution_history.html")
    observer.export("html", output_file)
    print(f"\n📄 Execution history exported to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
