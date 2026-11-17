
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# --------------------------------------------------
# WILDGUARD AI — WILDLIFE MONITORING & PROTECTION
# --------------------------------------------------

APP_TITLE = "WildGuard AI"
MODEL_PATH = os.getenv("MODEL_PATH", "wildlife_epoch_3.pth")
DATA_PATH = Path("wildlife_sightings.csv")
CONFIDENCE_THRESHOLD = 0.50

SPECIES = [
    "African Elephant",
    "Lion",
    "Panther",
    "Arctic Fox",
    "Cheetah",
    "Panda",
    "Rhinoceros",
    "Chimpanzee",
]

STATUSES = ["New", "Pending Review", "Reviewed", "Flagged"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def build_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, len(SPECIES) + 1
    )
    return model


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}. "
        "Upload wildlife_epoch_3.pth alongside app.py."
    )

model = build_model()
checkpoint = torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE)
model.eval()

print(f"{APP_TITLE}: model loaded on {DEVICE}")
print("WILDGUARD: PYTHON APP STARTED", flush=True)


# --------------------------------------------------
# SIGHTINGS DATA
# Demo entries are explicitly labelled as demo records.
# --------------------------------------------------

DEMO_ROWS = [
    ["WG-001", "African Elephant", "2026-10-27 09:15",
     "Kaziranga, Assam", 0.96, "Reviewed", "Demo record"],
    ["WG-002", "Lion", "2026-10-27 10:30",
     "Gir, Gujarat", 0.91, "Pending Review", "Demo record"],
    ["WG-003", "Panda", "2026-10-26 14:20",
     "Chengdu, China", 0.88, "New", "Demo record"],
    ["WG-004", "Rhinoceros", "2026-10-26 16:45",
     "Kaziranga, Assam", 0.94, "Pending Review", "Demo record"],
    ["WG-005", "Cheetah", "2026-10-25 11:05",
     "Kuno, Madhya Pradesh", 0.89, "New", "Demo record"],
    ["WG-006", "Chimpanzee", "2026-10-24 08:40",
     "Demo Reserve, Africa", 0.85, "Reviewed", "Demo record"],
]

COLUMNS = [
    "ID", "Species", "Date & Time", "Location",
    "Confidence", "Status", "Notes"
]


def load_sightings():
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    return pd.DataFrame(DEMO_ROWS, columns=COLUMNS)


sightings = load_sightings()


def save_sightings():
    sightings.to_csv(DATA_PATH, index=False)


# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------

def detect_animals(image):
    if image is None:
        return None, "Please upload an image first."

    image = image.convert("RGB")
    image.thumbnail((500, 500))
    tensor = TF.to_tensor(image).to(DEVICE)

    print("WILDGUARD: DETECTION STARTED", flush=True)

try:
    with torch.inference_mode():
        prediction = model([tensor])[0]
    print("WILDGUARD: DETECTION COMPLETED", flush=True)

except Exception as e:
    print(f"WILDGUARD DETECTION ERROR: {e}", flush=True)
    raise
        prediction = model([tensor])[0]

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    results = []

    for box, label, score in zip(
        prediction["boxes"].cpu(),
        prediction["labels"].cpu(),
        prediction["scores"].cpu()
    ):
        confidence = float(score)
        class_id = int(label)

        if confidence < CONFIDENCE_THRESHOLD:
            continue
        if not 1 <= class_id <= len(SPECIES):
            continue

        name = SPECIES[class_id - 1]
        x1, y1, x2, y2 = map(int, box.tolist())

        draw.rectangle(
            (x1, y1, x2, y2), outline="#36A269", width=4
        )
        draw.text(
            (x1, max(0, y1 - 22)),
            f"{name} {confidence:.0%}",
            fill="#36A269",
            stroke_width=1,
            stroke_fill="white"
        )
        results.append(f"- **{name}** — {confidence:.1%} confidence")

    if results:
        summary = (
            "### AI predictions — pending human verification\n\n"
            + "\n".join(results)
        )
    else:
        summary = (
            "No supported species detected above the "
            f"{CONFIDENCE_THRESHOLD:.0%} confidence threshold. "
            "This does not prove an animal is absent."
        )

    return result_image, summary


# --------------------------------------------------
# DASHBOARD DATA
# --------------------------------------------------

def get_overview():
    data = load_sightings()
    total = len(data)
    pending = int(data["Status"].isin(
        ["New", "Pending Review", "Flagged"]
    ).sum()) if total else 0
    species_count = int(data["Species"].nunique()) if total else 0

    return (
        f"### Total sightings\n# {total}",
        f"### Needs review\n# {pending}",
        f"### Species recorded\n# {species_count}",
        data
    )


def filter_sightings(search, species_filter, status_filter):
    data = load_sightings()

    if search:
        searchable = data.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        data = data[searchable]

    if species_filter != "All":
        data = data[data["Species"] == species_filter]

    if status_filter != "All":
        data = data[data["Status"] == status_filter]

    return data


def update_sighting(sighting_id, status, location, notes):
    global sightings

    data = load_sightings()
    matches = data.index[data["ID"].astype(str) == str(sighting_id)]

    if len(matches) == 0:
        return "Sighting ID not found.", data

    idx = matches[0]
    data.loc[idx, "Status"] = status
    data.loc[idx, "Location"] = location
    data.loc[idx, "Notes"] = notes

    sightings = data
    save_sightings()
    return f"Updated {sighting_id}.", load_sightings()


def export_report():
    data = load_sightings()
    output = Path("wildguard_sightings_report.csv")
    data.to_csv(output, index=False)
    return str(output)


def add_detection_to_register(image, location):
    if image is None:
        return "Upload an image before registering a detection.", load_sightings()

    _, summary = detect_animals(image)

    # Registering a detection is separate from validating the prediction.
    lines = [
        line.strip("- *")
        for line in summary.splitlines()
        if "confidence" in line.lower()
    ]

    if not lines:
        return (
            "No detections met the confidence threshold; nothing was registered.",
            load_sightings()
        )

    global sightings
    data = load_sightings()

    for line in lines:
        try:
            species_name, confidence_text = line.split(" — ")
            confidence = float(confidence_text.replace("%", "").strip()) / 100
        except (ValueError, TypeError):
            continue

        new_id = f"WG-{datetime.now().strftime('%y%m%d%H%M%S')}-{len(data)+1}"
        row = {
            "ID": new_id,
            "Species": species_name.strip(),
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Location": location.strip() or "Location not provided",
            "Confidence": confidence,
            "Status": "New",
            "Notes": "AI-generated record; requires human verification"
        }
        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)

    sightings = data
    save_sightings()
    return "Detection record(s) added for human review.", load_sightings()


# --------------------------------------------------
# FOREST THEME
# --------------------------------------------------

FOREST_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #071b14 0%, #102b20 55%, #173b2b 100%) !important;
    color: #e8f5e9 !important;
}

body {
    background: #071b14 !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label {
    color: #e8f5e9 !important;
}

.gradio-container .tabs {
    border-bottom: 1px solid #315b43 !important;
}

.gradio-container button {
    background: #245c3b !important;
    color: #ffffff !important;
    border: 1px solid #4d9464 !important;
    border-radius: 10px !important;
    transition: background 0.2s ease;
}

.gradio-container button:hover {
    background: #347b4e !important;
}

.gradio-container button.selected,
.gradio-container button[aria-selected="true"] {
    background: #347b4e !important;
    border-color: #9bcf9e !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #132d22 !important;
    color: #e8f5e9 !important;
    border: 1px solid #42694e !important;
}

.gradio-container .block,
.gradio-container .panel,
.gradio-container .form {
    background: rgba(19, 48, 34, 0.92) !important;
    border-color: #315b43 !important;
    border-radius: 14px !important;
}

.gradio-container table {
    background: #132d22 !important;
    color: #e8f5e9 !important;
}

.gradio-container thead {
    background: #245c3b !important;
}

.gradio-container a {
    color: #a5d6a7 !important;
}
"""

# --------------------------------------------------
# WEBSITE INTERFACE
# --------------------------------------------------

with gr.Blocks() as dashboard:
    gr.Markdown(
        """
        # 🌿 WildGuard AI
        ### Wildlife Monitoring & Protection System

        AI-assisted wildlife detection, sightings management, and
        conservation monitoring. AI predictions require human verification.
        """
    )

    with gr.Tab("Overview"):
        gr.Markdown("## Conservation dashboard")
        with gr.Row():
            total_card = gr.Markdown()
            review_card = gr.Markdown()
            species_card = gr.Markdown()

        overview_table = gr.Dataframe(
            value=load_sightings(),
            label="Recent wildlife sightings",
            interactive=False
        )
        refresh_overview = gr.Button("Refresh overview")
        refresh_overview.click(
            get_overview,
            outputs=[total_card, review_card, species_card, overview_table]
        )
        dashboard.load(
            get_overview,
            outputs=[total_card, review_card, species_card, overview_table]
        )

    with gr.Tab("Detect Species"):
        gr.Markdown(
            "Upload a wildlife image for model predictions. "
            "Results are not confirmed field observations."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload wildlife image")
            output_image = gr.Image(type="pil", label="Detection results")

        detection_summary = gr.Markdown()
        detect_button = gr.Button("Detect wildlife", variant="primary")
        detect_button.click(
            detect_animals,
            inputs=input_image,
            outputs=[output_image, detection_summary]
        )

        location_input = gr.Textbox(
            label="Location (optional)",
            placeholder="Enter a reserve or location"
        )
        register_button = gr.Button("Register detections for review")
        register_message = gr.Markdown()
        register_table = gr.Dataframe(label="Updated sightings register")

        register_button.click(
            add_detection_to_register,
            inputs=[input_image, location_input],
            outputs=[register_message, register_table]
        )

    with gr.Tab("Wildlife Sightings"):
        gr.Markdown("## Search and filter the sightings register")
        search_box = gr.Textbox(label="Search records")
        species_filter = gr.Dropdown(
            ["All"] + SPECIES, value="All", label="Species"
        )
        status_filter = gr.Dropdown(
            ["All"] + STATUSES, value="All", label="Review status"
        )
        search_button = gr.Button("Apply filters")
        sightings_table = gr.Dataframe(
            value=load_sightings(), label="Sightings", interactive=False
        )

        search_button.click(
            filter_sightings,
            inputs=[search_box, species_filter, status_filter],
            outputs=sightings_table
        )
        search_box.submit(
            filter_sightings,
            inputs=[search_box, species_filter, status_filter],
            outputs=sightings_table
        )

    with gr.Tab("Protection Monitoring"):
        gr.Markdown(
            "## Review a record\n"
            "Use the ID from the sightings register. Update the review "
            "status and add verified notes where available."
        )
        sighting_id = gr.Textbox(label="Sighting ID")
        new_status = gr.Dropdown(
            STATUSES, value="Pending Review", label="Review status"
        )
        new_location = gr.Textbox(label="Location")
        new_notes = gr.Textbox(label="Notes", lines=3)
        update_button = gr.Button("Save review")
        update_message = gr.Markdown()
        updated_table = gr.Dataframe(label="Updated register")

        update_button.click(
            update_sighting,
            inputs=[sighting_id, new_status, new_location, new_notes],
            outputs=[update_message, updated_table]
        )

    with gr.Tab("Reports"):
        gr.Markdown(
            "## Wildlife reports\n"
            "Export the current sightings register as a CSV file."
        )
        report_button = gr.Button("Generate CSV report")
        report_file = gr.File(label="Download report")
        report_button.click(export_report, outputs=report_file)

    with gr.Tab("Location Overview"):
        gr.Markdown(
            """
            ## Location overview

            This version displays the locations entered in sightings records.
            It does not collect GPS data or show live animal positions.
            """
        )
        location_table = gr.Dataframe(
            value=load_sightings()[["Species", "Location", "Status"]],
            label="Recorded locations",
            interactive=False
        )
        refresh_locations = gr.Button("Refresh locations")
        refresh_locations.click(
            lambda: load_sightings()[["Species", "Location", "Status"]],
            outputs=location_table
        )

    with gr.Tab("Settings & About"):
        gr.Markdown(
            f"""
            ## About WildGuard AI

            - **Model:** Faster R-CNN with ResNet-50 FPN
            - **Supported classes:** 8 animal species
            - **Inference device:** {DEVICE}
            - **Checkpoint:** {Path(MODEL_PATH).name}
            - **Confidence threshold:** {CONFIDENCE_THRESHOLD:.0%}

            **Important:** Demo records are illustrative, not real wildlife
            reports. Predictions can be wrong and should be checked by
            qualified people. This prototype does not automatically contact
            wildlife authorities or track animals live.
            """
        )

dashboard.queue()

if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate"
        ),
        css=FOREST_CSS
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )
