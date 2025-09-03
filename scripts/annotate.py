import asyncio
from oculodoc.config import OculodocConfig
from oculodoc.layout.doclayout_yolo_analyzer import DocLayoutYOLOAnalyzer
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import numpy as np


# pdf to image
pdf = fitz.open("demo1.pdf")
page_index = 4  # zero-based index; 4 corresponds to page 5
page = pdf.load_page(page_index)
pix = page.get_pixmap(dpi=150)
if pix.alpha:
    pix = fitz.Pixmap(fitz.csRGB, pix)
img_bytes = pix.tobytes("png")
image = Image.open(BytesIO(img_bytes))
pdf.close()


async def main():
    config = OculodocConfig()
    config.layout.device = "auto"
    analyzer = DocLayoutYOLOAnalyzer()
    await analyzer.initialize(config.layout.__dict__)
    results = await analyzer.analyze(image)
    print(results)
    for result in results:
        print(result.category_name)

    # Run raw prediction via underlying model to get a Results object for plotting
    img_array = np.array(image.convert("RGB"))
    det_res = analyzer.model.predict(
        img_array,
        imgsz=1024,
        conf=0.2,
        device=analyzer.device,
        verbose=False,
    )
    res0 = det_res[0]

    # Plot annotated image (handle both MinerU and Ultralytics result APIs)
    annotated_img = None
    try:
        plotted = res0.plot(pil=True, line_width=5, font_size=20)
        if isinstance(plotted, Image.Image):
            annotated_img = plotted
        else:
            # Assume numpy in BGR
            annotated_img = Image.fromarray(plotted[:, :, ::-1])
    except TypeError:
        # Ultralytics: no 'pil' kwarg, returns numpy BGR
        plotted = res0.plot()
        annotated_img = Image.fromarray(plotted[:, :, ::-1])

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "result_page5.jpg"
    annotated_img.save(out_path)
    print(f"Annotated page saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
