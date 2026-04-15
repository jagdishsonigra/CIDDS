import sys
import time
import numpy as np
import cv2
import pytesseract

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSlider, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class DamageDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚢 Container Damage Detector")
        self.setGeometry(100, 100, 1200, 700)

        self.model = None
        self.image_path = None

        self.init_ui()
        self.load_model()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()

        self.image_label = QLabel("Drop Image Here or Click Open")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; font-size: 16px;")
        self.image_label.setMinimumWidth(600)

        right_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.open_btn = QPushButton("📂 Open Image")
        self.open_btn.clicked.connect(self.open_image)

        self.detect_btn = QPushButton("🚀 Detect")
        self.detect_btn.clicked.connect(self.run_detection)

        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.detect_btn)

        conf_box = QGroupBox("Confidence Threshold")
        conf_layout = QVBoxLayout()

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(40)

        self.conf_label = QLabel("0.40")
        self.conf_slider.valueChanged.connect(self.update_conf)

        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        conf_box.setLayout(conf_layout)

        stats_box = QGroupBox("📊 Stats")
        stats_layout = QGridLayout()

        self.det_count = QLabel("0")
        self.avg_conf = QLabel("0.0")
        self.time_taken = QLabel("0 ms")

        stats_layout.addWidget(QLabel("Detections:"), 0, 0)
        stats_layout.addWidget(self.det_count, 0, 1)
        stats_layout.addWidget(QLabel("Avg Confidence:"), 1, 0)
        stats_layout.addWidget(self.avg_conf, 1, 1)
        stats_layout.addWidget(QLabel("Time:"), 2, 0)
        stats_layout.addWidget(self.time_taken, 2, 1)

        stats_box.setLayout(stats_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background:black; color:lime;")

        right_layout.addLayout(btn_layout)
        right_layout.addWidget(conf_box)
        right_layout.addWidget(stats_box)
        right_layout.addWidget(QLabel("Logs"))
        right_layout.addWidget(self.log_box)

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout)

        central.setLayout(main_layout)

    def load_model(self):
        try:
            self.model = YOLO("best.pt")
            self.log("✅ Model loaded successfully")
        except Exception as e:
            self.log(f"❌ Error loading model: {e}")

    def open_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file:
            self.image_path = file
            self.display_image(file)
            self.log(f"📂 Loaded image: {file}")

    def display_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def detect_text(self, image):
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        results_text = []

        versions = []
        versions.append(gray)
        versions.append(cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1])
        versions.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2))
        versions.append(cv2.Canny(gray, 50, 150))

        kernel = np.ones((2, 2), np.uint8)
        versions.append(cv2.dilate(gray, kernel, iterations=1))
        versions.append(cv2.erode(gray, kernel, iterations=1))

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        self.log("🔍 OCR Attempts:")

        for i, img in enumerate(versions):
            text = pytesseract.image_to_string(img, config=custom_config)
            cleaned = text.strip()

            if cleaned:
                self.log(f"--- Version {i+1} ---")
                lines = cleaned.split("\n")
                for line in lines:
                    line = line.strip()
                    if line:
                        self.log(f"➡️ {line}")
                        results_text.append(line)

        if not results_text:
            self.log("❌ No text detected")

        return results_text

    def run_detection(self):
        if not self.image_path:
            self.log("⚠️ No image selected")
            return

        conf = self.conf_slider.value() / 100.0
        self.log(f"🚀 Running detection (conf={conf:.2f})...")

        start = time.time()
        results = self.model(self.image_path, conf=conf)[0]
        end = time.time()

        original_bgr = cv2.imread(self.image_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        plotted_bgr = results.plot()
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        boxes = results.boxes
        count = len(boxes)

        confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        avg_conf = np.mean(confs) if len(confs) > 0 else 0

        self.det_count.setText(str(count))
        self.avg_conf.setText(f"{avg_conf:.2f}")
        self.time_taken.setText(f"{(end - start) * 1000:.0f} ms")

        if len(boxes) > 0:
            for i, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = original_rgb[y1:y2, x1:x2]
                self.log(f"🔎 OCR on detection {i}")
                if cropped.size > 0:
                    self.detect_text(cropped)
                else:
                    self.log("❌ Empty crop, skipped")
        else:
            self.log("🔎 No YOLO box found, running OCR on full image")
            self.detect_text(original_rgb)

        h, w, ch = plotted_rgb.shape
        qt_img = QImage(plotted_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

        self.log(f"✅ Done: {count} detections")

    def update_conf(self):
        val = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{val:.2f}")

    def log(self, msg):
        self.log_box.append(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DamageDetectorGUI()
    window.show()
    sys.exit(app.exec())