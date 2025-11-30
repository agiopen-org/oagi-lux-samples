import argparse
import asyncio
import os
import traceback
from datetime import datetime

from oagi import AsyncScreenshotMaker
from oagi.agent.observer import AsyncAgentObserver
from oagi.agent.tasker import TaskerAgent
from oagi.handler import AsyncPyautoguiActionHandler


async def main():
    parser = argparse.ArgumentParser(
        description="Run TaskerAgent to schedule a CVS appointment"
    )

    # CVS contact info
    parser.add_argument("--first_name", default="Joe")
    parser.add_argument("--last_name", default="Doe")
    parser.add_argument("--email", default="user@example.com")
    parser.add_argument("--birthday", default="01-01-1990")
    parser.add_argument("--zip_code", default="94404")

    # run config
    parser.add_argument("--exp_name", default="cvs")
    parser.add_argument("--save_dir", default="results")

    # agent config
    parser.add_argument("--model_name", default="lux-actor-1")
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    observer = AsyncAgentObserver()
    image_provider = AsyncScreenshotMaker()
    action_handler = AsyncPyautoguiActionHandler()

    tasker = TaskerAgent(
        api_key=os.getenv("OAGI_API_KEY"),
        base_url=os.getenv("OAGI_BASE_URL", "https://api.agiopen.org"),
        model=args.model_name,
        max_steps=args.max_steps,
        temperature=args.temperature,
        step_observer=observer,
    )

    month, day, year = args.birthday.split("-")
    instruction = (
        f"Schedule an appointment at CVS for {args.first_name} {args.last_name} "
        f"with email {args.email} and birthday {args.birthday}"
    )
    todos = [
        "Open a new tab, go to www.cvs.com, type 'flu shot' in the search bar and press enter, wait for the page to load, then click on the button of Schedule vaccinations on the top of the page",
        f"Enter the first name '{args.first_name}', last name '{args.last_name}', and email '{args.email}' in the form. Do not use any suggested autofills. Make sure the mobile phone number is empty.",
        f"Slightly scroll down to see the date of birth, enter Month '{month}', Day '{day}', and Year '{year}' in the form",
        "Scroll down and click on 'Continue as guest' button, wait for the page to load with wait, click on 'Add vaccines' button, select 'Flu' and click on 'Add vaccines'",
        f"Click on 'next' to enter the page with recommendation vaccines, then click on 'next' again, until on the page of entering zip code, enter '{args.zip_code}', select the first option from the dropdown menu, and click on 'Search'",
    ]

    tasker.set_task(instruction, todos)

    print(f"Starting task execution at {datetime.now()}")
    print("=" * 60)

    try:
        success = await tasker.execute(
            instruction="",
            action_handler=action_handler,
            image_provider=image_provider,
        )

        memory = tasker.get_memory()
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print(f"Overall success: {success}")
        print(memory.task_execution_summary)

        status_summary = memory.get_todo_status_summary()
        print(f"Completed: {status_summary.get('completed', 0)}")
    except Exception as exc:
        print(f"Error during execution: {exc}")
        traceback.print_exc()

    output_file = os.path.join(save_dir, "cvs_execution_history.html")
    observer.export("html", output_file)
    print(f"Exported execution history to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
