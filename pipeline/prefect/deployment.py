from datetime import timedelta
from pipeline import model_training_flow

if __name__ == "__main__":
    # Manually trigger an immediate run
    model_training_flow()

    # Schedule subsequent runs every 7 days
    model_training_flow.serve(
        name="model_training_deployment",
        interval=timedelta(days=7),  # Runs every 7 days
        tags=["ml", "production"],
    )
