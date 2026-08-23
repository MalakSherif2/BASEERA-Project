from roboflow import Roboflow
rf = Roboflow(api_key="6ZU6DLXJ3PlFwT1ksXE1")
project = rf.workspace("shahinaz-salah-s-workspace").project("weapon-yolo8-7ktme")
version = project.version(1)
dataset = version.download("yolov8")