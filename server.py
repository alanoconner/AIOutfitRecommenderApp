"""
Flask application exposing the outfit generation and evaluation API surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

import evaluator
from config import get_settings
from repo import generate_outfit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    """Factory to build the Flask application with configured extensions."""

    settings = get_settings()
    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = str(settings.uploads_dir)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    register_routes(app, settings.uploads_dir)
    return app


def register_routes(app: Flask, uploads_dir: Path) -> None:
    """Attach HTTP routes to the Flask application."""

    @app.route("/health", methods=["GET"])
    def healthcheck() -> Response:
        return jsonify({"status": "ok"})

    @app.route("/outfits", methods=["POST"])
    def create_outfit() -> Response:
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify({"error": "No image file supplied"}), 400

        gender = request.form.get("gender")
        height_cm = request.form.get("height")
        if not gender or not height_cm:
            return jsonify({"error": "Both gender and height fields are required"}), 400

        filename = secure_filename(file.filename)
        upload_path = uploads_dir / filename
        file.save(upload_path)
        logger.info("Uploaded file saved to %s", upload_path)

        try:
            result = generate_outfit(upload_path, gender, height_cm)
        finally:
            try:
                upload_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to remove upload %s: %s", upload_path, exc)

        return (
            jsonify(
                {
                    "message": "Outfit generated successfully",
                    "prompt": result.prompt,
                    "link": result.image_link,
                    "garment": result.garment,
                    "colour": result.colour,
                    "stylingAdvice": result.styling_advice,
                }
            ),
            200,
        )

    @app.route("/post", methods=["POST"])
    def legacy_create_outfit() -> Response:
        """Backwards-compatible endpoint preserved for the existing frontend."""

        return create_outfit()

    @app.route("/save-evaluations", methods=["POST"])
    def save_evaluations() -> Response:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Payload must be valid JSON"}), 400

        try:
            link = payload["generatedImageLink"]
            downloaded_path = evaluator.download_generated_image(link, str(uploads_dir))
            evaluator.save_eval_results(payload, downloaded_path)
        except KeyError as exc:
            return jsonify({"error": f"Missing required field: {exc}"}), 400
        except Exception as exc:
            logger.exception("Failed to persist evaluation")
            return jsonify({"error": str(exc)}), 500

        return jsonify({"message": "Evaluation saved"}), 200


app = create_app()


if __name__ == "__main__":
    app.run(port=8000, debug=True)
