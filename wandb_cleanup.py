import os
import wandb

# 🔹 Initialize WandB (if not already logged in)
wandb.login()

# 🔹 List all runs in the project
api = wandb.Api()
runs = api.runs("your_username/your_project_name")  # Replace with actual WandB project name

# ✅ Automatically delete old runs
for run in runs:
    if run.state == "finished":  # Only delete completed runs
        print(f"Deleting run {run.id} - {run.name}")
        run.delete()

print("🚀 Cleanup completed: Old finished runs deleted.")
