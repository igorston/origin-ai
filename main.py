"""Entry point (the product name comes from the brand file, see origin/branding.py).

Run:
    python main.py
    # or
    uvicorn main:app --reload

The app itself is built by origin.app.create_app, so a project that depends on origin-ai
builds the same one (see origin.plugins).
"""

from origin.app import create_app, run

app = create_app()


if __name__ == "__main__":
    run()
