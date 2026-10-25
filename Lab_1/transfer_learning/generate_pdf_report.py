"""Build the concise Lab 1 PDF report from the checked-in experiment artifacts."""

from pathlib import Path
import csv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "report" / "lab1_transfer_learning_report.pdf"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    pdfmetrics.registerFont(TTFont("Arial", font_path))
    pdfmetrics.registerFont(TTFont("Arial-Bold", bold_path))

    styles = getSampleStyleSheet()
    normal = ParagraphStyle("Russian", parent=styles["BodyText"], fontName="Arial", fontSize=11,
                            leading=15, spaceAfter=8)
    title = ParagraphStyle("TitleRussian", parent=styles["Title"], fontName="Arial-Bold", fontSize=16,
                           leading=21, alignment=TA_CENTER, spaceAfter=18)
    heading = ParagraphStyle("HeadingRussian", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=13,
                             leading=17, spaceBefore=10, spaceAfter=8)
    centered = ParagraphStyle("Centered", parent=normal, alignment=TA_CENTER)

    with (ARTIFACTS / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    data = [["Стратегия", "Accuracy", "Precision", "Recall", "F1"]]
    for row in rows:
        data.append([
            "Feature Extraction" if row["strategy"] == "feature_extraction" else "Fine-tuning",
            f"{float(row['accuracy']):.4f}",
            f"{float(row['precision_macro']):.4f}",
            f"{float(row['recall_macro']):.4f}",
            f"{float(row['f1_macro']):.4f}",
        ])

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=2.5 * cm,
                                 topMargin=2.0 * cm, bottomMargin=2.0 * cm)
    story = [
        Spacer(1, 3 * cm),
        paragraph("УНИВЕРСИТЕТ ИТМО", centered),
        Spacer(1, 2.5 * cm),
        paragraph("ЛАБОРАТОРНАЯ РАБОТА № 1", title),
        paragraph("Классификация изображений с применением Transfer Learning", title),
        Spacer(1, 4 * cm),
        paragraph("Дисциплина: Современные методы искусственного интеллекта", normal),
        paragraph("Студент: ______________________________", normal),
        paragraph("Группа: ________________________________", normal),
        Spacer(1, 2 * cm),
        paragraph("Санкт-Петербург, 2026", centered),
        PageBreak(),
        paragraph("1. Постановка задачи", heading),
        paragraph("Требовалось реализовать классификацию изображений не менее трёх классов, "
                  "использовать предобученную архитектуру и сравнить Feature Extraction с Fine-tuning.", normal),
        paragraph("В качестве датасета выбран MNIST. Использованы изображения цифр 0, 1 и 2. "
                  "Выборка сбалансирована: 900 изображений для обучения и 450 для тестирования "
                  "(по 300 и 150 на класс).", normal),
        paragraph("2. Архитектура и обучение", heading),
        paragraph("Использована MobileNetV2 с весами, предобученными на ImageNet. Это компактная "
                  "архитектура, подходящая для запуска на CPU. Изображения MNIST были преобразованы "
                  "в три канала, масштабированы до 96 x 96 и нормализованы параметрами ImageNet.", normal),
        paragraph("Для Feature Extraction базовая сеть полностью заморожена, а новый трёхклассовый "
                  "классификатор обучен с learning rate 0,001. Для Fine-tuning разморожены последние "
                  "два модуля признаков и применён пониженный learning rate 0,0001. В обеих стратегиях: "
                  "AdamW, seed 42, batch 64 и 3 эпохи.", normal),
        paragraph("3. Результаты", heading),
        Table(data, colWidths=[4.2 * cm, 2.3 * cm, 2.6 * cm, 2.2 * cm, 2.1 * cm], repeatRows=1,
              style=TableStyle([
                  ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Arial"),
                  ("FONTSIZE", (0, 0), (-1, -1), 9), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                  ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
              ])),
        Spacer(1, 14),
        PageBreak(),
        paragraph("Матрица ошибок - Feature Extraction", heading),
        Image(str(ARTIFACTS / "confusion_matrix_feature_extraction.png"), width=13.5 * cm, height=11.25 * cm),
        PageBreak(),
        paragraph("Матрица ошибок - Fine-tuning", heading),
        Image(str(ARTIFACTS / "confusion_matrix_fine_tuning.png"), width=13.5 * cm, height=11.25 * cm),
        paragraph("4. Вывод", heading),
        paragraph("Fine-tuning повысил Accuracy с 0,9378 до 0,9556, то есть на 1,78 процентного пункта. "
                  "Macro Precision и Macro Recall также выросли. Следовательно, для выбранной задачи "
                  "целесообразно дообучать верхнюю часть MobileNetV2 с низким темпом обучения.", normal),
        paragraph("Исходный код, конфигурация запуска, метрики и матрицы ошибок находятся в репозитории "
                  "в каталоге Lab_1/transfer_learning.", normal),
    ]
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
