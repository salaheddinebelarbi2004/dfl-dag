"""
QUICK START: Run Complete System with Real Data
===============================================

This is the easiest way to run the entire system:

1. Have 3 terminals open
2. Run the commands below in each terminal
3. Watch the federated learning happen!
"""

import sys
import subprocess
import time
import platform


def main():
    """Guide user through system startup"""

    print("\n" + "="*70)
    print("🚀 FEDERATED LEARNING SYSTEM - QUICK START")
    print("="*70)

    print("""
This script will start your complete P2P federated learning system.

You need 4 terminals total:
  • Terminal 1: Node 1 (port 5000)
  • Terminal 2: Node 2 (port 5001, connects to node 1)
  • Terminal 3: Node 3 (port 5002, connects to node 1)
  • Terminal 4: Main training orchestrator
    """)

    start_mode = input("\nChoose mode:\n1. Manual (I'll show you the commands)\n2. Auto (Start everything now)\n\nEnter 1 or 2: ").strip()

    if start_mode == "1":
        print_manual_commands()
    elif start_mode == "2":
        start_auto()
    else:
        print("Invalid choice!")
        sys.exit(1)


def print_manual_commands():
    """Print manual commands for user to run"""

    print("\n" + "="*70)
    print("MANUAL MODE - Copy & Paste These Commands")
    print("="*70)

    print("\n[TERMINAL 1] Start Node 1:")
    print("  python p2p_node.py 127.0.0.1 5000")

    print("\n[TERMINAL 2] Start Node 2 (wait 2 seconds after terminal 1 starts):")
    print("  python p2p_node.py 127.0.0.1 5001")
    print("  Then type: connect 127.0.0.1 5000")

    print("\n[TERMINAL 3] Start Node 3 (wait 2 seconds after terminal 2 starts):")
    print("  python p2p_node.py 127.0.0.1 5002")
    print("  Then type: connect 127.0.0.1 5000")

    print("\n[TERMINAL 4] Start main training (wait 5 seconds after all nodes start):")
    print("  python run_full_system.py")

    print("\n" + "-"*70)
    print("The system will:")
    print("  ✓ Load real federated data from real_federated_data/")
    print("  ✓ Initialize 12 clients across 3 shards")
    print("  ✓ Train lightweight classifier locally on each client")
    print("  ✓ Aggregate models using FedAvg")
    print("  ✓ Run 5 federated learning rounds")
    print("  ✓ Save results to full_system_results.json")
    print("\nEstimated time: 2-5 minutes\n")


def start_auto():
    """Automatically start P2P nodes"""

    print("\n" + "="*70)
    print("AUTO MODE - Starting Everything")
    print("="*70)

    if platform.system() == "Windows":
        start_auto_windows()
    else:
        start_auto_unix()


def start_auto_windows():
    """Start nodes on Windows with PowerShell"""

    print("\nStarting P2P nodes on Windows...")

    nodes = [
        ("Node-1", "5000"),
        ("Node-2", "5001"),
        ("Node-3", "5002"),
    ]

    for name, port in nodes:
        cmd = f'Start-Process pwsh -ArgumentList "-NoExit", "-Command", "python p2p_node.py 127.0.0.1 {port}"'
        print(f"  Starting {name} on port {port}...")

        try:
            subprocess.Popen(["powershell", "-Command", cmd])
            time.sleep(1)
        except Exception as e:
            print(f"    ⚠️  Could not start {name}: {e}")

    time.sleep(5)
    print("✅ Nodes started!\n")

    # Now run training
    print("Starting federated training...")
    print("-" * 70)

    try:
        subprocess.run([sys.executable, "run_full_system.py"], check=False)
    except Exception as e:
        print(f"❌ Error running training: {e}")


def start_auto_unix():
    """Start nodes on Unix/Linux/Mac"""

    print("\nStarting P2P nodes on Unix...")

    nodes = [
        ("Node-1", "5000"),
        ("Node-2", "5001"),
        ("Node-3", "5002"),
    ]

    for name, port in nodes:
        print(f"  Starting {name} on port {port}...")

        try:
            subprocess.Popen([
                "gnome-terminal", "--",
                sys.executable, "p2p_node.py", "127.0.0.1", port
            ])
            time.sleep(1)
        except:
            print(f"    ⚠️  Could not start {name}")

    time.sleep(5)
    print("✅ Nodes started!\n")

    print("Starting federated training...")
    print("-" * 70)

    try:
        subprocess.run([sys.executable, "run_full_system.py"], check=False)
    except Exception as e:
        print(f"❌ Error running training: {e}")


if __name__ == "__main__":
    main()
