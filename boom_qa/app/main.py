"""Runner for FastAPI app"""
from uvicorn import Config, Server

from boom_qa.app.setup import application

app = application()

if __name__ == "__main__":
    server = Server(
        Config(
            "boom_qa.app.main:app",
            host="0.0.0.0",
            port=8000,
            use_colors=False,
            workers=16,
        )
    )
    server.run()
