from prefect import serve
from datetime import timedelta
from pipeline import model_training_flow

if __name__ == "__main__":
    model_training_flow.serve(
        name="model_training_deployment",
        interval=timedelta(minutes=5),
        tags=["ml", "production"]
    )
