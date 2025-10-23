import subprocess
import sys
from pathlib import Path


def build_game_catalog(base_dir: Path):
    return [
        {
            "label": "Falling Dodge (hand tracking)",
            "path": base_dir / "game.py",
            "description": "Move your hand to dodge falling squares. Uses MediaPipe Hands.",
        },
        {
            "label": "Bubble Popper (index finger)",
            "path": base_dir / "bubble_pop_game.py",
            "description": "Pop rising bubbles by tapping them with your index fingertip.",
        },
        {
            "label": "Colour Chase (green object)",
            "path": base_dir / "color_chase_game.py",
            "description": "Guide a green prop to hit glowing orbs for points.",
        },
        {
            "label": "Hand Overlay (attach image)",
            "path": base_dir / "hand_overlay_tracker.py",
            "description": "Stick a chosen image to your detected hand; customize with --image.",
        },
    ]


def print_menu(games):
    print("\n=== Webcam Game Hub ===")
    for idx, game in enumerate(games, start=1):
        print(f"{idx}. {game['label']}")
        print(f"   -> {game['description']}")
    print("Q. Quit")


def resolve_interpreter():
    return sys.executable or "python"


def run_game(game):
    interpreter = resolve_interpreter()
    print(f"\nLaunching {game['label']}...\n")
    try:
        subprocess.run([interpreter, str(game["path"])], check=True)
    except subprocess.CalledProcessError as err:
        print(f"Game terminated with a non-zero exit status: {err.returncode}")
    except FileNotFoundError:
        print("Unable to locate Python interpreter or game script.")
    finally:
        print("\nGame finished. Returning to hub.")


def main():
    base_dir = Path(__file__).resolve().parent
    games = build_game_catalog(base_dir)

    missing = [game for game in games if not game["path"].exists()]
    if missing:
        print("Warning: some game scripts are missing:")
        for game in missing:
            print(f" - {game['label']} (expected at {game['path']})")
        print("Please ensure all scripts exist before launching games.")

    while True:
        print_menu(games)
        choice = input("Select a game number or Q to quit: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        if not choice.isdigit():
            print("Invalid selection. Please enter a number.")
            continue

        index = int(choice) - 1
        if 0 <= index < len(games):
            run_game(games[index])
        else:
            print("Number out of range. Try again.")


if __name__ == "__main__":
    main()
