# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable, Image
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import mm
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# -------------------- Loading the font --------------------
try:
    font_path = r"E:\Fonts\Fonts\Hacen Liner Broadcast HD_0.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('Cairo', font_path))
        BASE_FONT = 'Cairo'
    else:
        # print("Font Cairo not found, falling back to Helvetica")
        BASE_FONT = 'Helvetica'
except Exception as e:
    # print("Error loading font:", e)
    BASE_FONT = 'Helvetica'

# -------------------- Custom horizontal rule --------------------
class HRule(Flowable):
    """Custom horizontal line."""
    def __init__(self, width=460, thickness=0.8, color=colors.black, space_before=4, space_after=6):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color
        self.space_before = space_before
        self.space_after = space_after
        self.height = thickness + space_before + space_after

    def draw(self):
        self.canv.saveState()
        self.canv.setFillColor(self.color)
        self.canv.setStrokeColor(self.color)
        y = self.space_after
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, y, self.width, y)
        self.canv.restoreState()

# -------------------- PDF generation function --------------------
def make_pdf(filename="dlc_report.pdf", data=None, differential_df=None):
    if data is None:
        data = {}

    # Lab information
    lab_name = data.get("lab_name", "AI HematoLab")
    lab_tagline = data.get("lab_tagline", "Accurate  |  Caring  |  Instant")
    lab_addr = data.get("lab_addr", "105 -108, SMART VISION COMPLEX, HEALTHCARE ROAD, OPPOSITE HEALTHCARE COMPLEX. Damas - 689578")
    lab_contacts = data.get("lab_contacts", ["0123456789", "0912345678"])
    lab_email = data.get("lab_email", "drlogypathlab@AI.com")
    lab_site = data.get("lab_site", "www.AI.com")
    lab_logo = data.get("lab_logo", "lab_logo.png")

    patient = data.get("patient", {
        "Name": "Yashvi M. Patel",
        "Age": "21 Years",
        "Gender": "Female",
        "Uhid": "556",
        "Address_Title": "Sample Collected At:",
        "Address_text": "123 , Damascuse, Bramke",
        "ref_by": "Ref By: AI"
    })

    regs = data.get("regs", {
        "registered": "02:31 PM 02 Dec, 2X",
        "collected":  "03:11 PM 02 Dec, 2X",
        "reported":   "04:35 PM 02 Dec, 2X",
    })

    # Use differential_df for DLC results if provided, otherwise use default
    if differential_df is not None and isinstance(differential_df, pd.DataFrame):
        dlc_results = {
            row['Cell Type'].replace('Normal_', ''): row['Percentage (%)']
            for _, row in differential_df.iterrows()
        }
    else:
        dlc_results = data.get("dlc_results", {
            "Neutrophils": 50,
            "Lymphocytes": 45,
            "Eosinophils": 1,
            "Monocytes": 3,
            "Basophils": 1
        })

    # Reference ranges for each cell type
    reference_ranges = {
        "Neutrophils": (50, 62),
        "Lymphocytes": (20, 40),
        "Eosinophils": (0, 6),
        "Monocytes": (0, 10),
        "Basophils": (0, 2)
    }
  
 
    # Convert results to table rows with High/Low flags
    dlc_rows = []
    for cell_type, value in dlc_results.items():
        low, high = reference_ranges.get(cell_type, (0, 100))
        flag = ""
    
        if cell_type.lower() == "basophils":  
         if value <= low:
            flag = "Low"
         elif value > high:
            flag = "High"
        else:
         if value < low:
            flag = "Low"
         elif value > high:
            flag = "High"
    
        dlc_rows.append([cell_type, f"{value:.2f}", flag, f"{low} - {high}", "%"])


    comments_intro = data.get("comments_intro", "WBC stands for white blood cell, responsible for fighting infections.")
    low_causes = data.get("low_wbc_causes", [
        "Viral infections",
        "Chemotherapy or radiation therapy",
        "Autoimmune disorders",
        "Bone marrow disorders",
        "HIV/AIDS"
    ])
    high_causes = data.get("high_wbc_causes", [
        "Infection - Bacterial, viral, or parasitic",
        "Leukemia",
        "Stress",
        "Smoking",
        "Allergies",
        "Trauma"
    ])

    # Setup document and styles
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="DLC Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="h_lab", fontName=BASE_FONT, fontSize=16, leading=18, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="h_tag", fontName=BASE_FONT, fontSize=10, leading=12, textColor=colors.HexColor("#444"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="small_gray", fontName=BASE_FONT, fontSize=8.5, textColor=colors.HexColor("#444")))
    styles.add(ParagraphStyle(name="small", fontName=BASE_FONT, fontSize=9))
    styles.add(ParagraphStyle(name="label", fontName=BASE_FONT, fontSize=9, textColor=colors.HexColor("#333")))
    styles.add(ParagraphStyle(name="value", fontName=BASE_FONT, fontSize=10))
    styles.add(ParagraphStyle(name="section", fontName=BASE_FONT, fontSize=11, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="table_head", fontName=BASE_FONT, fontSize=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="table_cell", fontName=BASE_FONT, fontSize=10))
    styles.add(ParagraphStyle(name="bold", fontName=BASE_FONT, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="foot", fontName=BASE_FONT, fontSize=8, alignment=TA_RIGHT))

    flow = []

    # -------------------- Lab logo --------------------
    if os.path.exists(lab_logo):
        img = Image(lab_logo, width=40*mm, height=20*mm)
        img.hAlign = 'CENTER'
        flow.append(img)
        flow.append(Spacer(1, 4))

    # Header
    flow.append(Paragraph(f"<b>{lab_name}</b>", styles["h_lab"]))
    flow.append(Paragraph(lab_tagline, styles["h_tag"]))
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(lab_addr, styles["small_gray"]))
    flow.append(Paragraph(f"{'  |  '.join(lab_contacts)}", styles["small_gray"]))
    flow.append(Paragraph(lab_email, styles["small_gray"]))
    flow.append(Paragraph(lab_site, styles["small_gray"]))
    flow.append(Spacer(1, 6))
    flow.append(HRule())

    # Patient information
    pt_rows = [
        [
            Paragraph("<b>Name</b>: " + patient["Name"], styles["label"]),
            Paragraph("<b>Age</b>: " + patient["Age"], styles["label"]),
            Paragraph("<b>Gender</b>: " + patient["Gender"], styles["label"]),
            Paragraph("<b>UHID</b>: " + patient["Uhid"], styles["label"]),
        ],
        [
            Paragraph(f"<b>{patient['Address_Title']}</b> {patient['Address_text']}", styles["label"]),
            "",
            "",
            "",
        ],
        [
            Paragraph(patient["ref_by"], styles["label"]),
            "",
            "",
            "",
        ]
    ]
    t_pt = Table(pt_rows, colWidths=[60*mm, 30*mm, 30*mm, 40*mm])
    t_pt.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
    ]))
    flow.append(t_pt)
    flow.append(Spacer(1, 6))

    # Timestamps
    reg_rows = [
        [Paragraph("<b>Registered on:</b> " + regs["registered"], styles["small"]),
         Paragraph("<b>Collected on:</b> " + regs["collected"], styles["small"]),
         Paragraph("<b>Reported on:</b> " + regs["reported"], styles["small"])],
    ]
    t_regs = Table(reg_rows, colWidths=[65*mm, 65*mm, 65*mm])
    t_regs.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    flow.append(t_regs)
    flow.append(HRule())

    # Results table
    table_data = [["DIFFERENTIAL LEUCOCYTE COUNT", "", "", "", ""]]
    for item, result, flag, ref, unit in dlc_rows:
        flag_text = f"⚠️ <font color='#b00000'><b>{flag}</b></font>" if flag else ""
        table_data.append([
            Paragraph(item, styles["table_cell"]),
            Paragraph(result + flag_text, styles["table_cell"]),
            Paragraph(flag_text, styles["table_cell"]),
            Paragraph(ref, styles["table_cell"]),
            Paragraph(unit, styles["table_cell"]),
        ])

    col_widths = [80*mm, 25*mm, 25*mm, 45*mm, 15*mm]
    t_main = Table(table_data, colWidths=col_widths, hAlign="LEFT")
    t_main.setStyle(TableStyle([
        ("SPAN", (0,0), (-1,0)),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e0e0e0")),
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("GRID", (0,1), (-1,-1), 0.5, colors.HexColor("#ccc")),
        ("FONTNAME", (0,0), (-1,-1), BASE_FONT),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ALIGN", (1,1), (1,-1), "CENTER"),
        ("ALIGN", (3,1), (4,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            t_main.setStyle(TableStyle([("BACKGROUND", (0,i), (-1,i), colors.HexColor("#f9f9f9"))]))
    flow.append(t_main)
    flow.append(Spacer(1, 6))

    # Comments
    flow.append(Paragraph("<b>Comments :</b>", styles["bold"]))
    flow.append(Paragraph(comments_intro, styles["small"]))
    flow.append(Spacer(1, 4))
    
     #General Recommendation
    general_recs = [
    "Results should be interpreted in the clinical context of the patient.",
    "Consult a specialist if any counts are outside the reference range.",
    "Consider repeating the test or additional investigations if abnormalities persist."
    ]

    flow.append(Paragraph("<b>Recommendations :</b>", styles["bold"]))
    for rec in general_recs:
     flow.append(Paragraph("• " + rec, styles["small"]))
    flow.append(Spacer(1, 4))
     
    # Specific Recommendations for abnormal cells
    cell_recommendations = {
    "Neutrophils": {
        "High": "May indicate bacterial infection; consult your doctor.",
        "Low": "May indicate viral infection or immune suppression."
    },
    "Lymphocytes": {
        "High": "May indicate chronic infection or inflammation.",
        "Low": "May indicate immune deficiency or viral infection."
    },
    "Eosinophils": {
        "High": "May indicate allergy or parasitic infection.",
        "Low": ""
    },
    "Monocytes": {
        "High": "May indicate infection or inflammation.",
        "Low": ""
    },
    "Basophils": {
        "High": "",
        "Low": "Usually clinically insignificant if slight."
    }
    }
    flow.append(Paragraph("<b>Result :</b>", styles["bold"]))
    for cell_type, value, flag, ref, unit in dlc_rows:
     if flag:  # High or Low
        rec_text = cell_recommendations.get(cell_type, {}).get(flag, "")
        if rec_text:
            flow.append(Paragraph(f"• {cell_type} ({flag}): {rec_text}", styles["small"]))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("<b>Low WBC Count Causes :</b>", styles["bold"]))
    for li in low_causes:
        flow.append(Paragraph("• " + li, styles["small"]))
    flow.append(Spacer(1, 4))

    flow.append(Paragraph("<b>High WBC Count Causes :</b>", styles["bold"]))
    for li in high_causes:
        flow.append(Paragraph("• " + li, styles["small"]))
    flow.append(Spacer(1, 8))

    # Footer
    footer_txt = (
        "To Check Report Authenticity by Scanning QR Code on Top"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Page 1 of 1"
    )
    flow.append(Paragraph(footer_txt, styles["foot"]))

    doc.build(flow)
    # print(f"PDF generated: {filename}")

def make_leukemia_pdf(filename="leukemia_report.pdf", data=None, leukemia_df=None):
    if data is None:
        data = {}

    # Lab information
    lab_name = data.get("lab_name", "AI HematoLab")
    lab_tagline = data.get("lab_tagline", "AI-Powered Leukemia Diagnostics")
    lab_logo = data.get("lab_logo", "lab_logo.png")

    patient = data.get("patient", {
        "Name": "Sample Patient",
        "Age": "45 Years",
        "Gender": "Male",
        "Uhid": "12345",
        "ref_by": "Ref By: AI System"
    })

    # Process leukemia_df
    if leukemia_df is not None and isinstance(leukemia_df, pd.DataFrame):
        results = {
            row['Leukemia Type']: row['Percentage (%)']
            for _, row in leukemia_df.iterrows()
        }
    else:
        results = {"ALL": 0, "AML": 0, "CLL": 0, "CML": 0}

    # Risk recommendations
    risk_recs = {
        "ALL": "Acute Lymphoblastic Leukemia is fast-growing. Immediate evaluation and chemotherapy are often required.",
        "AML": "Acute Myeloid Leukemia progresses rapidly. Urgent hematology referral is advised.",
        "CLL": "Chronic Lymphocytic Leukemia may progress slowly. Monitoring or targeted therapy may be needed.",
        "CML": "Chronic Myeloid Leukemia responds well to tyrosine kinase inhibitors (e.g., imatinib)."
    }

    # Stage approximation (example logic)
    stages = {
        "ALL": "Acute Phase",
        "AML": "Acute Phase",
        "CLL": "Chronic Phase",
        "CML": "Chronic Phase"
    }

    # -------------------- Document setup --------------------
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Leukemia Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="h_lab", fontName=BASE_FONT, fontSize=16, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="h_tag", fontName=BASE_FONT, fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#555")))
    styles.add(ParagraphStyle(name="small", fontName=BASE_FONT, fontSize=9))
    styles.add(ParagraphStyle(name="table_cell", fontName=BASE_FONT, fontSize=10))
    styles.add(ParagraphStyle(name="bold", fontName=BASE_FONT, fontSize=10, leading=12))

    flow = []

    # -------------------- Header --------------------
    if os.path.exists(lab_logo):
        img = Image(lab_logo, width=40*mm, height=20*mm)
        img.hAlign = 'CENTER'
        flow.append(img)
        flow.append(Spacer(1, 4))

    flow.append(Paragraph(f"<b>{lab_name}</b>", styles["h_lab"]))
    flow.append(Paragraph(lab_tagline, styles["h_tag"]))
    flow.append(Spacer(1, 6))
    flow.append(HRule())

    # -------------------- Patient Info --------------------
    flow.append(Paragraph(f"<b>Name:</b> {patient['Name']}", styles["small"]))
    flow.append(Paragraph(f"<b>Age:</b> {patient['Age']}   |   <b>Gender:</b> {patient['Gender']}", styles["small"]))
    flow.append(Paragraph(f"<b>UHID:</b> {patient['Uhid']}   |   {patient['ref_by']}", styles["small"]))
    flow.append(Spacer(1, 6))
    flow.append(HRule())

    # -------------------- Results Table --------------------
    table_data = [["Leukemia Type", "Percentage (%)", "Stage", "Risk/Recommendations"]]

    for cell_type, value in results.items():
        rec_text = risk_recs.get(cell_type, "")
        stage = stages.get(cell_type, "N/A")
        table_data.append([
            Paragraph(cell_type, styles["table_cell"]),
            Paragraph(f"{value:.2f}", styles["table_cell"]),
            Paragraph(stage, styles["table_cell"]),
            Paragraph(rec_text, styles["table_cell"]),
        ])

    col_widths = [40*mm, 30*mm, 40*mm, 70*mm]
    t_main = Table(table_data, colWidths=col_widths, hAlign="LEFT")
    t_main.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e0e0e0")),
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("GRID", (0,1), (-1,-1), 0.5, colors.HexColor("#ccc")),
        ("FONTNAME", (0,0), (-1,-1), BASE_FONT),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    flow.append(t_main)
    flow.append(Spacer(1, 8))
    report_id = data.get("report_id", f"RPT-{datetime.now().strftime('%Y%m%d%H%M')}")
    report_date = datetime.now().strftime("%d-%b-%Y %H:%M")

    flow.append(Paragraph(f"<b>Report ID:</b> {report_id}", styles["small"]))
    flow.append(Paragraph(f"<b>Date:</b> {report_date}", styles["small"]))
    flow.append(Paragraph(f"<b>Contact:</b> +1-800-555-HEMO | info@aihemolab.com", styles["small"]))
    flow.append(Spacer(1, 8))
    flow.append(HRule())

    # --- Risk Highlight ---
    max_type = max(results, key=results.get)
    max_val = results[max_type]

    if max_val > 20:
        risk_text = f"⚠️ High suspicion of {max_type} ({max_val:.1f}%). Immediate referral advised."
        risk_color = colors.HexColor("#FFCDD2")
    elif max_val > 5:
        risk_text = f"⚠️ Moderate risk due to {max_type} ({max_val:.1f}%). Monitoring required."
        risk_color = colors.HexColor("#FFF9C4")
    else:
        risk_text = "✅ No significant leukemic cell population detected."
        risk_color = colors.HexColor("#C8E6C9")

    risk_box = Table([[Paragraph(risk_text, styles["table_cell"])]],
                     colWidths=[180*mm])
    risk_box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), risk_color),
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    flow.append(Spacer(1, 6))
    flow.append(risk_box)
    flow.append(Spacer(1, 10))

    # --- Chart (distribution of leukemia types) ---
    if results:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.pie(results.values(), labels=results.keys(), autopct='%1.1f%%')
        chart_path = "leukemia_chart.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close(fig)

        if os.path.exists(chart_path):
            flow.append(Image(chart_path, width=80*mm, height=60*mm))
            flow.append(Spacer(1, 10))

    # --- Disclaimer & Signature ---
    flow.append(HRule())
    flow.append(Paragraph("<b>Disclaimer:</b> This report is generated using AI-based analysis. "
                          "It should be interpreted by a certified hematologist before final diagnosis.",
                          styles["small"]))
    flow.append(Spacer(1, 20))
    flow.append(Paragraph("Authorized Signatory: ____________________", styles["small"]))
    flow.append(Spacer(1, 10))

    # -------------------- Final Notes --------------------
    flow.append(Paragraph("<b>General Recommendations:</b>", styles["bold"]))
    flow.append(Paragraph("• Interpret results in clinical context.", styles["small"]))
    flow.append(Paragraph("• Consult a hematologist for further evaluation.", styles["small"]))
    flow.append(Paragraph("• Additional bone marrow or genetic tests may be required.", styles["small"]))

    doc.build(flow)
    # print(f"✅ Leukemia PDF generated: {filename}")


# if __name__ == "__main__":
#     # Example usage with a sample DataFrame (replace with actual differential_df from first code)
#     sample_differential_df = pd.DataFrame({
#         'Cell Type': ['Normal_Neutrophils', 'Normal_Lymphocytes', 'Normal_Eosinophils', 'Normal_Monocytes', 'Normal_Basophils'],
#         'Count': [50, 40, 3, 7, 0],
#         'Percentage (%)': [50.0, 40.0, 3.0, 7.0, 0.0]
#     })
#     overall_cnn_differential_df = pd.DataFrame({
#         'Cell Type': ['ALL', 'AML', 'CLL', 'CML'],
#         'Count': [12, 7, 3, 8],
#         'Percentage (%)': [40.0, 23.3, 10.0, 26.7]
#     })
#     make_leukemia_pdf(r"./dlc_report.pdf", leukemia_df=overall_cnn_differential_df)