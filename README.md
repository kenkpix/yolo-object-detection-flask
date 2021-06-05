# Simple flask app for testing yolo model on your own computer
 
## Installation

1. Create virtual environment and install required modules:
    
    ```
    python -m venv venv
    pip install -r requirements.txt
    ```

2. Run Flask app

## About app

This app provide your opportunity to test YOLO model (object detection) on your own videos and photos.

* App will automatically use your GPU if possible. Without GPU, video and photos processing take significantly more time.
* Maximum file size restricted to 10 mb. You can change it in app.py file: 
```python
app.config["MAX_CONTENT_LENGTH"] = your_max_file_size
```
* By default, app will use tiny yolo model for increasing performance. If you want to use vanilla model, you should change path to weights and cfg file in app.py.
```python
CFG_FILE_PATH = "path to cfg"
WEIGHTS_FILE_PATH = "path to weights"
```