import os
from locust import HttpUser, task, between


# Path to a sample image used for every prediction request.
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")


class DermAIUser(HttpUser):
    """Simulates a user hitting the three main endpoints."""

    # Each simulated user waits 1–3 seconds between tasks.
    wait_time = between(1, 3)

    #Health / uptime 
    @task(1)
    def check_uptime(self):
        """Lightweight health check — low weight, called occasionally."""
        self.client.get("/uptime")

    # Prediction 
    @task(5)
    def predict(self):
        """POST an image to /predict — the heaviest endpoint, highest weight."""
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            # Skip gracefully if no test image is present.
            return

        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("test_image.jpg", f, "image/jpeg")},
                name="/predict",
            )

    # Root 
    @task(1)
    def root(self):
        self.client.get("/")
