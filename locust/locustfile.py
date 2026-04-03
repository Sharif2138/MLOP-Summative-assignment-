"""
Locust load-test for the DermAI FastAPI backend.

Usage
-----
1. Install Locust:
       pip install locust

2. Put one test image in this folder named test_image.jpg
   (any skin image from your dataset works).

3. Run the web UI:
       locust -f locustfile.py --host http://localhost:8000

4. Open http://localhost:8089 in your browser.
   Set number of users and spawn rate, then click Start.

5. To run headlessly (no browser) and save a CSV report:
       locust -f locustfile.py --host http://localhost:8000 \
              --headless -u 50 -r 5 --run-time 60s \
              --csv results/locust_report

Docker container scaling experiment
------------------------------------
Run these commands in separate terminals and record the Locust
stats (avg/p95 response time, RPS, failure rate) for each:

  # 1 container
  docker compose up --scale ml-api=1

  # 2 containers  (requires a load balancer — see README)
  docker compose up --scale ml-api=2

  # 3 containers
  docker compose up --scale ml-api=3

Then paste the CSV results into your README under "Flood Simulation Results".
"""

import os
from locust import HttpUser, task, between


# Path to a sample image used for every prediction request.
# Keep it small (< 500 KB) so the test measures API latency,
# not file-upload time.
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")


class DermAIUser(HttpUser):
    """Simulates a user hitting the three main endpoints."""

    # Each simulated user waits 1–3 seconds between tasks.
    wait_time = between(1, 3)

    # ── Health / uptime ──────────────────────────────────────────────────
    @task(1)
    def check_uptime(self):
        """Lightweight health check — low weight, called occasionally."""
        self.client.get("/uptime")

    # ── Prediction ───────────────────────────────────────────────────────
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

    # ── Root ─────────────────────────────────────────────────────────────
    @task(1)
    def root(self):
        self.client.get("/")
