from PyQt5.QtGui import QPixmap, QFont, QMovie, QIcon
import os
import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog,
    QLineEdit, QHBoxLayout, QComboBox, QFormLayout, QFrame
)
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from Models.models import process_WBC, process_Lukemia
from PdfGenerator import make_leukemia_pdf , make_pdf
import subprocess
from PyQt5.QtWidgets import QMessageBox

class AnalysisWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, folder_path, test_type,patient_data, parent=None):
        super().__init__()
        self.folder_path = folder_path
        self.test_type = test_type
        self.patient_data = patient_data
    def run(self):
        if self.test_type.startswith("white blood cell"):
            result = process_WBC(self.folder_path,self.patient_data["Name"])
            test_type = "WBC"
        else:
            result = process_Lukemia(self.folder_path,self.patient_data["Name"])
            test_type = "Leukemia"

        # patient_data = {
        #     "name": self.parent().name_input.text(),
        #     "age": self.parent().age_input.text(),
        #     "gender": self.parent().gender_combo.currentText()
        # }

        self.finished.emit((result, test_type, self.patient_data))


class WelcomeWindow(QWidget):
    def __init__(self, switch_to_main):
        super().__init__()
        self.switch_to_main = switch_to_main
        self.setWindowTitle("Smart Blood - System for Blood test")
        self.showMaximized()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        logo = QLabel()
        logo_path = os.path.abspath("assets/depo.jpg")
        pixmap = QPixmap(logo_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)

        title = QLabel("SmartBlood")
        title.setFont(QFont("Cairo", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #D9D8D7;")

        subtitle = QLabel("A smart system for analyzing white blood cells and detecting leukemia in its early stages")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Cairo", 12))
        subtitle.setStyleSheet("color: #0E1700;")

        start_button = QPushButton("Start")
        start_button.setFont(QFont("Cairo", 14, QFont.Bold))
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.setFixedWidth(400)
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #F25252;
                color: white;
                border-radius: 20px;
                padding: 12px 24px;
                border: none;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
        """)
        start_button.clicked.connect(self.switch_to_main)

        layout.addWidget(logo)
        layout.addSpacing(30)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(30)
        layout.addWidget(start_button)

        self.setLayout(layout)


class AnalyzerWindow(QWidget):
    def __init__(self, switch_to_loading):
        super().__init__()
        self.setWindowTitle("white blood cell analysis")
        self.switch_to_loading = switch_to_loading
        self.showMaximized()
        self.folder_path = None
        self.setup_ui()

    def setup_ui(self):
        # --- Global Style ---
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
                font-family: 'Cairo';
            }
            QLineEdit, QComboBox {
                border: 2px solid #42a5f5;
                border-radius: 8px;
                padding: 8px;
                background-color: white;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #1e88e5;
            }
            QLabel {
                color: #0D47A1;
            }
        """)

        # --- Title ---
        title_label = QLabel("🔬 SmartBlood Analyzer")
        title_label.setFont(QFont("Cairo", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            background-color: #E3F2FD;
            color: #0D47A1;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        """)

        # --- Subtitle ---
        subtitle_label = QLabel("Patient Information & Test Selection")
        subtitle_label.setFont(QFont("Cairo", 13))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            background-color: #458C5E;
            color: #D9D8D7;
            padding: 8px;
            border-radius: 6px;
            margin-bottom: 20px;
        """)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        main_layout.setContentsMargins(50, 20, 50, 20)

        # --- Form Frame ---
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #AFA2FF;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #90CAF9;
            }
        """)
        form_layout = QFormLayout()
        input_font = QFont("Cairo", 12)

        # Inputs
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Name")
        self.name_input.setFont(input_font)
        form_layout.addRow(self.name_input)

        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age")
        self.age_input.setMaxLength(3)
        self.age_input.setValidator(QIntValidator(0, 120))
        self.age_input.setFont(input_font)
        form_layout.addRow(self.age_input)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"])
        self.gender_combo.setFont(input_font)
        form_layout.addRow(self.gender_combo)

        self.uhid_input = QLineEdit()
        self.uhid_input.setPlaceholderText("UHID")
        self.uhid_input.setFont(input_font)
        form_layout.addRow(self.uhid_input)

        self.address_title_input = QLineEdit()
        self.address_title_input.setPlaceholderText("Address Title")
        self.address_title_input.setFont(input_font)
        form_layout.addRow(self.address_title_input)

        self.address_text_input = QLineEdit()
        self.address_text_input.setPlaceholderText("Address")
        self.address_text_input.setFont(input_font)
        form_layout.addRow(self.address_text_input)

        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems([
            "white blood cell count (WBC)",
            "blood cancer screening (Leukemia)"
        ])
        self.test_type_combo.setFont(input_font)
        form_layout.addRow(self.test_type_combo)

        form_frame.setLayout(form_layout)
        main_layout.addWidget(form_frame)
        # main_layout.addSpacing(10)

        self.image_label = QLabel("The microscopic image will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(250, 250)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #42A5F5;
                border-radius: 15px;
                color: #90CAF9;
                font-size: 15px;
                background-color: #F1F8FF;
            }
        """)

        button_style = """
            QPushButton {{
                background-color: {bg};
                color: white;
                border-radius: 20px;
                padding: 12px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

        upload_btn = QPushButton("Input folder")
        upload_btn.setFixedWidth(220)
        upload_btn.setFont(QFont("Cairo", 14, QFont.Bold))
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.setStyleSheet(button_style.format(bg="#7A89C2", hover="#afa2ff"))
        upload_btn.clicked.connect(self.upload_image)

        analyze_btn = QPushButton("Analyze Images")
        analyze_btn.setFixedWidth(220)
        analyze_btn.setFont(QFont("Cairo", 12, QFont.Bold))
        analyze_btn.setCursor(Qt.PointingHandCursor)
        analyze_btn.setStyleSheet(button_style.format(bg="#1B5E20", hover="#388E3C"))
        analyze_btn.clicked.connect(self.run_analysis)

        # --- Layouting ---
        img_row = QHBoxLayout()
        img_row.addStretch()
        img_row.addWidget(self.image_label)
        img_row.addStretch()
        main_layout.addLayout(img_row)
        # main_layout.addSpacing(20)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(upload_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # --- Result label ---
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Cairo", 13))
        self.result_label.setStyleSheet("""
            color: #1565C0;
            font-size: 14px;
            margin-top: 10px;
        """)

        # main_layout.addWidget(self.result_label)
        main_layout.addWidget(analyze_btn, alignment=Qt.AlignRight)

        # Apply layout
        self.setLayout(main_layout)

    def upload_image(self):
        if not self.name_input.text().strip():
            self.result_label.setText("Please enter a name.")
            return
        if not self.age_input.text().strip():
            self.result_label.setText("Please enter an age.")
            return

        self.folder_path = QFileDialog.getExistingDirectory(self, "Choose the images folder")
        if self.folder_path:
            image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if image_files:
                first_image = os.path.join(self.folder_path, image_files[0])
                pixmap = QPixmap(first_image).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.result_label.setText(f"Selected {len(image_files)} images. Ready to analyze.")
            else:
                self.result_label.setText("⚠️ The folder has no images.")

    def run_analysis(self):
        if not self.folder_path:
            self.result_label.setText("⚠️ Please choose a folder first.")
            return

        test_type = self.test_type_combo.currentText()

        patient_data = {
            "Name": self.name_input.text().strip(),
            "Age": self.age_input.text().strip(),
            "Gender": self.gender_combo.currentText(),
            "Uhid": getattr(self, "uhid_input", None).text().strip() if hasattr(self, "uhid_input") else "N/A",
            "Address_Title": self.address_title_input.text().strip() or "Address",
            "Address_text": self.address_text_input.text().strip() or "N/A",
            "ref_by": "Ref By: AI"
        }

        # Pass patient_data to the worker; make the analyzer the parent (optional)
        self.worker = AnalysisWorker(self.folder_path, test_type, patient_data, parent=self)
        self.worker.finished.connect(self.analysis_done)

        self.switch_to_loading()  # move to loading screen immediately
        self.worker.start()

    def analysis_done(self, data):
        result_df, test_type, patient_data = data
        self.parent().goto_result(result_df, test_type, patient_data)


class LoadingScreen(QWidget):
    def __init__(self, switch_to_result):
        super().__init__()
        self.switch_to_result = switch_to_result
        self.setWindowTitle("Test is Running ...")
        self.setWindowIcon(QIcon("assets/depo.jpg"))
        self.showMaximized()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.spinner = QLabel()
        gif_path = os.path.abspath("assets/original.gif")
        movie = QMovie(gif_path)
        self.spinner.setMovie(movie)
        movie.start()

        layout.addWidget(self.spinner)
        self.setLayout(layout)



class ResultScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Analysis Result")
        self.showMaximized()
        self.setWindowIcon(QIcon("assets/depo.jpg"))
        self.result_df = None
        self.test_type = None
        self.patient_data = {}
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
                font-family: 'Cairo';
            }
            QLabel {
                color: #212121;
            }
            QTableWidget {
                border: 1px solid #90caf9;
                border-radius: 8px;
                gridline-color: #bbdefb;
                background-color: white;
                selection-background-color: #64b5f6;
                selection-color: black;
            }
            QHeaderView::section {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
        """)

        # --- Main Layout ---
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        # --- Logo + Report ID ---
        # header_layout = QHBoxLayout()
        # logo = QLabel()
        # logo.setPixmap(QPixmap("assets/hospital_logo.png").scaled(
        #     80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation
        # ))
        # header_layout.addWidget(logo)

        # report_id = QLabel("Report ID: #123456")
        # report_id.setFont(QFont("Cairo", 12))
        # report_id.setAlignment(Qt.AlignRight)
        # header_layout.addWidget(report_id)

        # self.layout.addLayout(header_layout) 

        # --- Header / Title ---
        self.title_label = QLabel("🔬 SmartBlood Analyzer")
        self.title_label.setFont(QFont("Cairo", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            background-color: #E3F2FD;
            color: #0D47A1;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        """)
        # self.title_label = QLabel("SmartBlood Diagnostic Report")
        # self.title_label.setAlignment(Qt.AlignCenter)
        # self.title_label.setFont(QFont("Cairo", 20, QFont.Bold))
        # self.title_label.setStyleSheet("color: #000001;background-color:#730202")
        self.layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("White Blood Cell / Leukemia Screening")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setFont(QFont("Cairo", 14))
        self.subtitle_label.setStyleSheet("color: #D9D8D7;background-color:#458C5E")
        self.layout.addWidget(self.subtitle_label)
        # self.layout.addSpacing(20)


                # --- Back Button ---
        back_btn = QPushButton("⬅ Back to Analyzer")
        back_btn.setFont(QFont("Cairo", 8))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #388E3C;
                color: white;
                border-radius: 10px;
                padding: 7.5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        back_btn.clicked.connect(self.go_back)
        self.layout.addWidget(back_btn, alignment=Qt.AlignLeft)

        # --- Patient Info ---
        self.patient_frame = QFrame()
        self.patient_layout = QFormLayout()
        self.patient_frame.setLayout(self.patient_layout)
        self.patient_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 12px;
                background-color: #E3F2FD;
            }
            QLabel {
                font-size: 13px;
                color: #212121;
            }
            QLabel[bold="true"] {
                font-weight: bold;
                color: #0D47A1;
            }
        """)
        self.layout.addWidget(self.patient_frame)

        # # --- Results Section Title ---
        # results_title = QLabel("📊 Test Results")
        # results_title.setAlignment(Qt.AlignCenter)
        # results_title.setFont(QFont("Cairo", 14, QFont.Bold))
        # results_title.setStyleSheet("""
        #     QLabel {
        #         background-color: #1976D2;
        #         color: white;
        #         padding: 6px;
        #         border-radius: 6px;
        #     }
        # """)
        # self.layout.addWidget(results_title)

        # --- Results Table ---
        self.result_table = QTableWidget()
        self.result_table.setMinimumSize(600, 300)
        self.result_table.horizontalHeader().setDefaultSectionSize(180)
        self.result_table.verticalHeader().setDefaultSectionSize(45)

        self.result_table.setStyleSheet("""
            QTableWidget {
                font-size: 13px;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                gridline-color: #ccc;
                background-color: #ffffff;
                alternate-background-color: #f7f7f7;
                selection-background-color: #1976D2;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #1565C0;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
        self.result_table.setAlternatingRowColors(True)
        self.layout.addWidget(self.result_table,alignment=Qt.AlignCenter)

        # # --- Medical Impression ---
        # self.impression_label = QLabel("")
        # self.impression_label.setWordWrap(True)
        # self.impression_label.setAlignment(Qt.AlignLeft)
        # self.impression_label.setStyleSheet("""
        #     QLabel {
        #         background-color: #FFF3CD;
        #         border: 1px solid #FFEEBA;
        #         border-radius: 6px;
        #         padding: 10px;
        #         color: #856404;
        #         font-size: 13px;
        #     }
        # """)
        # self.layout.addWidget(self.impression_label)

        # --- Save PDF Button ---
        save_btn = QPushButton("💾 Save as PDF")
        save_btn.setFont(QFont("Cairo", 14, QFont.Bold))
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #A9674F;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #92514F;
            }
        """)
        save_btn.clicked.connect(self.save_pdf)
        self.layout.addWidget(save_btn, alignment=Qt.AlignCenter)
        # --- Show Images Folder Button ---
        show_img_btn = QPushButton("📂 Open Images Folder")
        show_img_btn.setFont(QFont("Cairo", 14, QFont.Bold))
        show_img_btn.setCursor(Qt.PointingHandCursor)
        show_img_btn.setStyleSheet("""
            QPushButton {
                background-color: #4F7DA9;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3A5F80;
            }
        """)
        show_img_btn.clicked.connect(self.open_images_folder)
        self.layout.addWidget(show_img_btn, alignment=Qt.AlignCenter)

    def open_images_folder(self):
        """Open the folder where the images are saved based on test type."""
        if not self.test_type or not self.patient_data.get("Name"):
            QMessageBox.warning(self, "No Data", "⚠️ Please run an analysis first.")
            return

        patient_name = self.patient_data.get("Name", "Unknown")
        if self.test_type == "WBC":
            folder = os.path.abspath(f'./Output_WBC/{patient_name}')
        elif self.test_type == "Leukemia":
            folder = os.path.abspath(f'./output_Lukemia/{patient_name}')
        else:
            QMessageBox.warning(self, "Error", "Unknown test type.")
            return

        if not os.path.exists(folder):
            QMessageBox.warning(self, "Not Found", f"No images found in:\n{folder}")
            return

        # Open in OS file explorer
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(["open", folder])
        else:  # Linux
            subprocess.Popen(["xdg-open", folder])

    def display_result(self, df, test_type, patient_data):
        """Update screen when analysis is done."""
        self.result_df = df
        self.test_type = test_type
        self.patient_data = patient_data

        # --- Update patient info ---
        while self.patient_layout.count():
            item = self.patient_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # --- Add new patient info ---
        for k, v in patient_data.items():
            self.patient_layout.addRow(QLabel(f"<b>{k}</b>:"), QLabel(str(v)))

        # --- Update results table ---
        if df is not None and not df.empty:
            self.result_table.setRowCount(len(df))
            self.result_table.setColumnCount(len(df.columns))
            self.result_table.setHorizontalHeaderLabels(df.columns)

            for i in range(len(df)):
                for j, col in enumerate(df.columns):
                    self.result_table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        else:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            self.result_table.setHorizontalHeaderLabels([])

        # # --- Medical Impression ---
        # if hasattr(self, "impression_label"):
        #     self.layout.removeWidget(self.impression_label)
        #     self.impression_label.deleteLater()

        # if self.test_type == "WBC":
        #     impression_text = "WBC Differential test completed.\nNormal ranges appear stable." \
        #         if "Neutrophils" in df.columns and df["Neutrophils"].mean() < 70 else \
        #         "Abnormal WBC count detected. Further investigation recommended."
        # elif self.test_type == "Leukemia":
        #     impression_text = "Leukemia screening completed.\nNo malignant patterns detected." \
        #         if not df.empty and df["Count"].sum() == 0 else \
        #         "Suspicious cells detected. Hematologist consultation advised."
        # else:
        #     impression_text = "Analysis completed."

        # self.impression_label = QLabel(f"<b>Medical Impression:</b><br>{impression_text}")
        # self.impression_label.setWordWrap(True)
        # self.impression_label.setStyleSheet("color: #F25252;background-color:#D99152; font-size: 14px; margin: 10px;")
        # self.layout.addWidget(self.impression_label)
    
    def go_back(self):
        """Return to AnalyzerWindow"""
        main_app = self.parent()  # QStackedWidget
        if isinstance(main_app, QStackedWidget):
            main_app.setCurrentIndex(1)  # AnalyzerWindow is index 1

    def save_pdf(self):
        """Export report to PDF using PdfGenerator."""
        if self.result_df is None or self.test_type is None:
            self.result_label.setText("⚠️ No results to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf)")
        if not save_path:
            return

        if self.test_type == "WBC":
            make_pdf(filename=save_path, data={"patient": self.patient_data}, differential_df=self.result_df)
        elif self.test_type == "Leukemia":
            # print(f"this {self.result_df}")
            make_leukemia_pdf(filename=save_path, data={"patient": self.patient_data}, leukemia_df=self.result_df)

class MainApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartBlood")
        self.setWindowIcon(QIcon("assets/depo.jpg"))
        self.showMaximized()
        self.welcome_screen = WelcomeWindow(self.goto_analyzer)
        self.analyzer_screen = AnalyzerWindow(self.goto_loading)
        self.loading_screen = LoadingScreen(self.goto_result)
        self.result_screen = ResultScreen()

        self.addWidget(self.welcome_screen)
        self.addWidget(self.analyzer_screen)
        self.addWidget(self.loading_screen)
        self.addWidget(self.result_screen)

        self.setMinimumSize(1000, 700)
        self.setCurrentIndex(0)

    def goto_analyzer(self):
        self.setCurrentIndex(1)

    def goto_loading(self):
        self.setCurrentIndex(2)

    def goto_result(self, result_df, test_type, patient_data):
        self.result_screen.display_result(result_df, test_type, patient_data)
        self.setCurrentIndex(3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #8E938D;
        }
    """)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
