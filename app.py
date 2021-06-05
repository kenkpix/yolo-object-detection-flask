from flask import Flask, render_template, after_this_request, redirect, flash, request
from werkzeug.utils import secure_filename
from model import *
import os
import random
import re

USER_RESULTS_FOLDER = os.path.join("static", "user-results")
USER_UPLOADS_FOLDER = "user-files"
ALLOWED_EXTENSIONS = ("jpg", "png", "jpeg", "avi", "mp4", "mov")
CFG_FILE_PATH = "cfg/yolov4-tiny.cfg"
WEIGHTS_FILE_PATH = "weights/yolov4-tiny.weights"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = USER_UPLOADS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 mb


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def identify_file_type(filename):
    return (
        "image"
        if re.search("\S{0}\w+$", filename.lower())[0] in ALLOWED_EXTENSIONS[:3]
        else "video"
    )


@app.route("/", methods=["GET", "POST"])
def main_page():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("The supported extensions for files is jpg, png, jpeg, avi, mp4")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            if identify_file_type(filename) == "image":
                # Create a result of photo detection
                res_filename = f"res_img_{random.randrange(100000)}.jpg"
                res_path = os.path.join("static/user-results", res_filename)
                detect_cv2(
                    CFG_FILE_PATH,
                    WEIGHTS_FILE_PATH,
                    file_path,
                    res_path,
                )

            else:
                # Create a result of video detection
                res_filename = f"res_vid_{random.randrange(100000)}.webm"
                res_path = os.path.join("static/user-results", res_filename)
                detect_cv2_camera(CFG_FILE_PATH, WEIGHTS_FILE_PATH, file_path, res_path)

            @after_this_request
            def delete_user_upload(response):
                try:
                    os.remove(file_path)
                except Exception as error:
                    app.logger.error("Error removing user uploaded image.", error)
                return response

            return redirect(f"/result/{res_filename}")

    return render_template("main_page.html")


@app.route("/result/<result>")
def show_result(result):
    file_type = identify_file_type(result)
    return render_template("show_result.html", result=result, file_type=file_type)


if __name__ == "__main__":
    app.run()