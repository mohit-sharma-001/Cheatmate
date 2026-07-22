import os
import io
import re
import json
import html
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Register fonts from local app/fonts directory
# Uses paths relative to this file's directory so it works in any environment.
fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
permanent_marker_path = os.path.join(fonts_dir, "PermanentMarker-Regular.ttf")
patrick_hand_path = os.path.join(fonts_dir, "PatrickHand-Regular.ttf")

pdfmetrics.registerFont(TTFont("PermanentMarker", permanent_marker_path))
pdfmetrics.registerFont(TTFont("PatrickHand", patrick_hand_path))


class HandwrittenDivider(Flowable):
    """
    A custom flowable that draws a hand-drawn-style dashed line divider.
    Occupies exactly 24pt of vertical space to preserve alignment on the 24pt grid.
    """
    def __init__(self, width, color):
        super().__init__()
        self.width = width
        self.color = color

    def wrap(self, availWidth, availHeight):
        # Occupies exactly 24pt (1 line on the ruled grid)
        return availWidth, 24

    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(1.5)
        # Set dashed pattern for the hand-drawn effect
        self.canv.setDash(4, 4)
        # Draw the line in the middle of the 24pt vertical space (y = 12)
        self.canv.line(0, 12, self.width, 12)
        self.canv.restoreState()


def draw_background(canvas_obj, doc_obj):
    """
    Draws a warm off-white notebook paper background with light blue-gray
    ruled lines and a red vertical margin line, along with subtle hole punches.
    """
    canvas_obj.saveState()
    
    # 1. Warm cream background fill (#FDF6E3)
    canvas_obj.setFillColor(colors.HexColor("#FDF6E3"))
    canvas_obj.rect(0, 0, doc_obj.pagesize[0], doc_obj.pagesize[1], fill=True, stroke=False)
    
    # 2. Horizontal ruled lines (light blue-gray, thin, spaced 24pt apart)
    width, height = doc_obj.pagesize
    canvas_obj.setStrokeColor(colors.HexColor("#D2E4F0"))
    canvas_obj.setLineWidth(0.75)
    
    # Starting ruled lines relative to the top of the flowable area (720 pt)
    # The top margin is 72 pt, so the first line of text baseline sits around y = 700 pt.
    y = 720 - 20
    while y > 40:
        canvas_obj.line(0, y, width, y)
        y -= 24
        
    # 3. Vertical red margin line (60pt from the left edge)
    canvas_obj.setStrokeColor(colors.HexColor("#FF7F7F"))
    canvas_obj.setLineWidth(1.0)
    canvas_obj.line(60, 0, 60, height)
    
    # 4. Subtle 3-hole-punch on the left edge (grey circles with white shadow)
    hole_x = 25
    hole_radius = 6
    hole_positions = [height * 0.15, height * 0.5, height * 0.85]
    
    for hole_y in hole_positions:
        # Dark grey ring
        canvas_obj.setFillColor(colors.HexColor("#CCCCCC"))
        canvas_obj.setStrokeColor(colors.HexColor("#A8A8A8"))
        canvas_obj.setLineWidth(1)
        canvas_obj.circle(hole_x, hole_y, hole_radius, fill=True, stroke=True)
        # Inner white to create hole illusion
        canvas_obj.setFillColor(colors.HexColor("#FFFFFF"))
        canvas_obj.circle(hole_x - 1, hole_y + 1, hole_radius - 1, fill=True, stroke=False)
        
    canvas_obj.restoreState()


def generate_handwritten_pdf(content: str, feature: str, title: str) -> bytes:
    """
    Creates a notebook-style PDF using ReportLab flowables, returns raw PDF bytes.
    """
    # Distinct "ink" colors for headings and accent details based on feature type
    FEATURE_COLORS = {
        "long_notes": colors.HexColor("#1B365D"),   # Navy Blue
        "short_notes": colors.HexColor("#1E4620"),  # Forest Green
        "cheat_sheet": colors.HexColor("#5A189A"),  # Deep Purple
        "flashcards": colors.HexColor("#800020"),   # Burgundy
        "quiz": colors.HexColor("#800020")          # Burgundy
    }
    ink_color = FEATURE_COLORS.get(feature, colors.HexColor("#1A1A2E"))
    
    buffer = io.BytesIO()
    
    # Standard letter size page (612 x 792 pt).
    # Margins: left=80 (to clear the red margin line at 60), right=40, top=72, bottom=72
    # Flowable height is exactly 792 - 144 = 648 pt (multiple of 24)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=80,
        rightMargin=40,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Title style: centered, PermanentMarker, 24pt font, 48pt leading (takes 2 grid lines)
    title_style = ParagraphStyle(
        "NotebookTitle",
        parent=styles["Normal"],
        fontName="PermanentMarker",
        fontSize=24,
        leading=48,
        textColor=ink_color,
        alignment=1,  # Center alignment
        spaceAfter=24  # Leaves exactly 1 empty grid line
    )
    
    story = []
    
    # Add escaped document title
    escaped_title = html.escape(title)
    story.append(Paragraph(escaped_title, title_style))
    
    is_json_processed = False
    
    # Process structured JSON data for flashcards and quiz features
    if feature in ("flashcards", "quiz"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                card_q_style = ParagraphStyle(
                    "CardQuestion",
                    parent=styles["Normal"],
                    fontName="PermanentMarker",
                    fontSize=16,
                    leading=24,
                    textColor=ink_color,
                    spaceBefore=0,
                    spaceAfter=0
                )
                card_a_style = ParagraphStyle(
                    "CardAnswer",
                    parent=styles["Normal"],
                    fontName="PatrickHand",
                    fontSize=14,
                    leading=24,
                    textColor=colors.HexColor("#1A1A2E"),
                    spaceBefore=0,
                    spaceAfter=0
                )
                
                for index, item in enumerate(data):
                    # Add gap before cards except the first
                    if index > 0:
                        story.append(Spacer(1, 24))
                        
                    if feature == "flashcards":
                        q = html.escape(item.get("question", ""))
                        a = html.escape(item.get("answer", ""))
                        story.append(Paragraph(f"Q: {q}", card_q_style))
                        story.append(Paragraph(f"A: {a}", card_a_style))
                    else:  # quiz
                        q = html.escape(item.get("question", ""))
                        opts = [html.escape(o) for o in item.get("options", [])]
                        ans = html.escape(item.get("correct_answer", ""))
                        
                        story.append(Paragraph(f"Question: {q}", card_q_style))
                        for opt_idx, opt in enumerate(opts):
                            prefix = chr(65 + opt_idx)
                            story.append(Paragraph(f"{prefix}) {opt}", card_a_style))
                        story.append(Paragraph(f"Correct Answer: {ans}", card_a_style))
                        
                    # Add dashed divider line between cards
                    if index < len(data) - 1:
                        story.append(Spacer(1, 24))
                        story.append(HandwrittenDivider(doc.width, ink_color))
                        
                is_json_processed = True
        except Exception as e:
            # Fallback to standard plain text markdown parsing if JSON decoding fails
            print(f"Failed to parse JSON for {feature}, falling back to plain text: {e}")
            is_json_processed = False

    # Process standard Markdown content
    if not is_json_processed:
        lines = content.splitlines()
        
        body_style = ParagraphStyle(
            "NotebookBody",
            parent=styles["Normal"],
            fontName="PatrickHand",
            fontSize=14,
            leading=24,
            textColor=colors.HexColor("#1A1A2E"),
            spaceBefore=0,
            spaceAfter=24  # leaves one blank line between paragraphs
        )
        
        bullet_style = ParagraphStyle(
            "NotebookBullet",
            parent=styles["Normal"],
            fontName="PatrickHand",
            fontSize=14,
            leading=24,
            textColor=colors.HexColor("#1A1A2E"),
            leftIndent=24,
            firstLineIndent=-12,
            spaceBefore=0,
            spaceAfter=0  # bullet list items are stacked without empty lines
        )
        
        heading_style = ParagraphStyle(
            "NotebookHeading",
            parent=styles["Normal"],
            fontName="PermanentMarker",
            fontSize=18,
            leading=24,
            textColor=ink_color,
            spaceBefore=24,
            spaceAfter=24
        )
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Strip bold marks (**) since reportlab canvas doesn't easily format inline bold
            line_str = line_str.replace("**", "")
            
            # Check if line is a heading
            if line_str.startswith("#"):
                level = 0
                while level < len(line_str) and line_str[level] == "#":
                    level += 1
                heading_text = html.escape(line_str[level:].strip())
                story.append(Paragraph(heading_text, heading_style))
                
            # Check if line is a list item
            elif line_str.startswith(("*", "-")):
                bullet_text = html.escape(line_str[1:].strip())
                story.append(Paragraph("• " + bullet_text, bullet_style))
                
            # Regular text paragraph
            else:
                body_text = html.escape(line_str)
                story.append(Paragraph(body_text, body_style))
                
    # Build the document templates and draw background decorations on all pages
    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
