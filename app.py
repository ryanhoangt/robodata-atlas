import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import csv
import re
from pathlib import Path
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="RoboData Atlas", layout="wide")

# --- 1. Load Data ---
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "data/Open-X-Embodiment-Dataset.tsv"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = _SLUG_PATTERN.sub("-", cleaned).strip("-")
    return cleaned or "dataset"


def _parse_int(value) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    text = text.replace(",", "")
    try:
        return int(float(text))
    except ValueError:
        return 0


@st.cache_data
def load_data():
    datasets = []
    if not DATASET_PATH.exists():
        return datasets

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return datasets
        num_columns = len(header)

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue

            version = None
            if len(row) == num_columns + 1 and row[0].strip().lower().startswith("v"):
                version = row[0].strip()
                row = row[1:]

            if len(row) > num_columns:
                overflow = row[num_columns:]
                row = row[:num_columns]
                if overflow:
                    combined = " ".join(part for part in [row[-1], *overflow] if part.strip())
                    row[-1] = combined.strip()
            elif len(row) < num_columns:
                row = row + [""] * (num_columns - len(row))

            record = dict(zip(header, row))
            if version:
                record["Version"] = version

            dataset_name = record.get("Dataset", "").strip()
            if not dataset_name:
                continue

            dataset_id = record.get("Registered Dataset Name", "").strip()
            if not dataset_id:
                dataset_id = _slugify(dataset_name)

            dataset_url = record.get("Dataset URL", "").strip()
            if not dataset_url:
                dataset_url = None

            rgb_cams = _parse_int(record.get("# RGB Cams"))
            depth_cams = _parse_int(record.get("# Depth Cams"))
            wrist_cams = _parse_int(record.get("# Wrist Cams"))

            sensors = []
            if rgb_cams > 0:
                sensors.append("RGB")
            if depth_cams > 0:
                sensors.append("Depth")
            if record.get("Has Proprioception?", "").strip().lower() == "yes":
                sensors.append("Proprioception")
            language_annotation = record.get("Language Annotations", "").strip()
            if language_annotation and language_annotation.lower() not in {"none", "no"}:
                sensors.append("Language")
            sensors = list(dict.fromkeys(sensors))

            viewpoints = []
            if wrist_cams > 0:
                viewpoints.append("Wrist")
            if rgb_cams > 0 or depth_cams > 0:
                viewpoints.append("External")
            viewpoints = list(dict.fromkeys(viewpoints))

            frequency_raw = record.get("Control Frequency", "").strip()
            if frequency_raw and not frequency_raw.lower().endswith("hz"):
                frequency = f"{frequency_raw} Hz"
            elif frequency_raw:
                frequency = frequency_raw
            else:
                frequency = "Unknown"

            datasets.append(
                {
                    "id": dataset_id,
                    "name": dataset_name,
                    "description": record.get("Description", "").strip(),
                    "url": dataset_url,
                    "hardware": {
                        "robot": record.get("Robot", "Unknown").strip() or "Unknown",
                        "end_effector": record.get("Gripper", "Unknown").strip() or "Unknown",
                        "morphology": record.get("Robot Morphology", "").strip(),
                    },
                    "modality": {
                        "sensors": sensors,
                        "viewpoints": viewpoints,
                    },
                    "task_env": {
                        "environment": record.get("Scene Type", "Unknown").strip() or "Unknown",
                        "domain": record.get("Data Collect Method", "").strip(),
                        "language_labels": language_annotation.lower() not in {"", "none", "no"},
                    },
                    "engineering": {
                        "format": record.get("Action Space", "Unknown").strip() or "Unknown",
                        "frequency": frequency,
                    },
                    "stats": {
                        "episodes": record.get("# Episodes", "").strip(),
                        "file_size_gb": record.get("File Size (GB)", "").strip(),
                        "language_annotations": language_annotation,
                        "data_collect_method": record.get("Data Collect Method", "").strip(),
                        "has_suboptimal": record.get("Has Suboptimal?", "").strip(),
                        "has_camera_calibration": record.get("Has Camera Calibration?", "").strip(),
                        "has_proprioception": record.get("Has Proprioception?", "").strip(),
                    },
                    "registered_name": record.get("Registered Dataset Name", "").strip(),
                    "citation": record.get("Citation", "").strip(),
                    "latex_reference": record.get("Latex Reference", "").strip(),
                    "version": record.get("Version"),
                }
            )

    return datasets

ROBOT_IMAGE_FILES = {
    "Cobotta": "assets/images/cobotta.png",
    "DLR EDAN": "assets/images/dlr-edan.jpeg",
    "DLR SARA": "assets/images/dlr-sara.jpeg",
    "Fanuc Mate": "assets/images/fanuc-mate.png",
    "Franka": "assets/images/Franka-Emika-Panda.png",
    "Google Robot": "assets/images/google-robot.png",
    "Hello Stretch": "assets/images/hello-stretch.png",
    "Jackal": "assets/images/jackal.jpg",
    "Jaco 2": "assets/images/jaco-2.webp",
    "Kinova Gen3": "assets/images/kinova-gen3.webp",
    "Kuka iiwa": "assets/images/kuka-iiwa.png",
    "MobileALOHA": "assets/images/mobile-aloha.avif",
    # "Multi-Robot": "",
    "PAMY2": "assets/images/pamy2.png",
    "PR2": "assets/images/pr2.jpg",
    "RC Car": "assets/images/rc-car.webp",
    "Sawyer": "assets/images/sawyer.jpg",
    "Spot": "assets/images/spot.jpg",
    "TidyBot": "assets/images/tidybot.png",
    "TurtleBot 2": "assets/images/turtlebot-2.png",
    "UR5": "assets/images/ur5.png",
    "Unitree A1": "assets/images/unitree-a1.png",
    "ViperX Bimanual": "assets/images/viperx-bimanual.png",
    "WidowX": "assets/images/WidowX-250.png",
    "xArm": "assets/images/xarm.avif",
    "xArm Bimanual": "assets/images/xarm-bimanual.png",
    ### Unused images
    # "Fetch Mobile Manipulator": "assets/images/Fetch-Mobile-Manipulator.png",
}


@st.cache_resource(show_spinner=False)
def load_robot_images():
    images = {}
    for robot_name, relative_path in ROBOT_IMAGE_FILES.items():
        abs_path = (BASE_DIR / relative_path).resolve()
        if abs_path.exists():
            suffix = abs_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".gif":
                mime = "image/gif"
            elif suffix == ".webp":
                mime = "image/webp"
            elif suffix == ".avif":
                mime = "image/avif"
            else:
                mime = "image/png"
            encoded = base64.b64encode(abs_path.read_bytes()).decode("utf-8")
            images[robot_name] = f"data:{mime};base64,{encoded}"
    return images


robot_images = load_robot_images()


data = load_data()
if "last_clicked_node" not in st.session_state:
    st.session_state["last_clicked_node"] = None


# --- 2. Sidebar Filters ---
st.sidebar.title("RoboData Atlas")
page = st.sidebar.radio("Navigate", ("Atlas", "All datasets"), index=0)
st.sidebar.divider()
st.sidebar.header("🔍 Filter Atlas")

# Extract unique options for filters
all_robots = sorted({d['hardware']['robot'] for d in data})
all_envs = sorted({d['task_env']['environment'] for d in data})

selected_robots = st.sidebar.multiselect("Select Robot Hardware", all_robots, default=all_robots)
selected_envs = st.sidebar.multiselect("Select Scene Type", all_envs, default=all_envs)

# Filter the dataset list based on selection
filtered_data = [
    d for d in data
    if d['hardware']['robot'] in selected_robots
    and d['task_env']['environment'] in selected_envs
]

st.sidebar.markdown(f"**Showing {len(filtered_data)} datasets**")

if page == "Atlas":
    nodes = []
    edges = []
    existing_nodes = set()

    def add_node(id, label, color=None, shape="dot", size=25, title=None, **node_kwargs):
        if id not in existing_nodes:
            node_title = title if title is not None else (label if label not in (None, "") else id)
            nodes.append(
                Node(
                    id=id,
                    label=label,
                    title=node_title,
                    color=color,
                    shape=shape,
                    size=size,
                    **node_kwargs,
                )
            )
            existing_nodes.add(id)

    for d in filtered_data:
        dataset_node_id = d['id']
        robot_name = d['hardware']['robot']
        scene_type = d['task_env']['environment']
        action_space = d['engineering']['format']
        stats = d.get("stats", {})
        episodes = stats.get("episodes")

        dataset_title_lines = [
            d['name'],
            f"Robot: {robot_name}",
            f"Scene: {scene_type}",
            f"Action space: {action_space}",
        ]
        if episodes:
            dataset_title_lines.append(f"Episodes: {episodes}")
        if d.get("version"):
            dataset_title_lines.append(f"Version: {d['version']}")
        dataset_title = "\n".join(dataset_title_lines)
        add_node(dataset_node_id, d['name'], "#00C49F", size=12, title=dataset_title, image="")

        robot_id = f"robot_{robot_name}"
        robot_image = robot_images.get(robot_name)
        if robot_image:
            add_node(
                robot_id,
                robot_name,
                color={"border": "#FFBB28", "background": "#ffffff"},
                shape="circularImage",
                size=36,
                title=f"Robot: {robot_name}",
                image=robot_image,
            )
        else:
            add_node(
                robot_id,
                robot_name,
                "#FFBB28",
                size=18,
                title=f"Robot: {robot_name}",
            )
        edges.append(
            Edge(
                source=robot_id,
                target=dataset_node_id,
                color="#FFBB28",
                title=f"Robot: {robot_name}",
                arrows="to",
            )
        )

        scene_id = f"scene_{scene_type}"
        add_node(scene_id, "", "#FF8042", size=10, title=f"Scene: {scene_type}")
        edges.append(
            Edge(
                source=dataset_node_id,
                target=scene_id,
                color="#FF8042",
                title=f"Scene: {scene_type}",
                arrows="to",
            )
        )

        action_id = f"action_{action_space}"
        add_node(action_id, "", "#8884d8", size=10, title=f"Action space: {action_space}")
        edges.append(
            Edge(
                source=dataset_node_id,
                target=action_id,
                color="#8884d8",
                title=f"Action space: {action_space}",
                arrows="to",
            )
        )

    st.title("RoboData Atlas 🗺️")

    legend_html = """
    <style>
    .atlas-legend {display:flex; flex-wrap:wrap; gap:0.75rem; margin-bottom:1rem;}
    .atlas-legend__item {display:flex; align-items:center; gap:0.4rem; font-size:0.9rem;}
    .atlas-legend__swatch {width:14px; height:14px; border-radius:50%;}
    </style>
    <div class="atlas-legend">
        <div class="atlas-legend__item">
            <span class="atlas-legend__swatch" style="background-color:#00C49F;"></span>
            Dataset
        </div>
        <div class="atlas-legend__item">
            <span class="atlas-legend__swatch" style="background-color:#FFBB28;"></span>
            Robot attribute
        </div>
        <div class="atlas-legend__item">
            <span class="atlas-legend__swatch" style="background-color:#FF8042;"></span>
            Scene attribute
        </div>
        <div class="atlas-legend__item">
            <span class="atlas-legend__swatch" style="background-color:#8884d8;"></span>
            Action space attribute
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("Interactive map of robotics datasets. **Drag nodes** to explore connections.")
    st.caption("Tip: Click a dataset node to open its dataset page in a new tab.")

    config = Config(
        width="100%",
        height=800,
        directed=True,
        physics=True,
        hierarchy=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        fit=False,
    )

    return_value = agraph(nodes=nodes, edges=edges, config=config)

    if return_value:
        clicked_node_id = return_value
        is_new_click = clicked_node_id != st.session_state.get("last_clicked_node")
        st.session_state["last_clicked_node"] = clicked_node_id

        dataset_info = next((item for item in data if item["id"] == clicked_node_id), None)

        if dataset_info:
            dataset_url = dataset_info.get("url")
            if dataset_url and is_new_click:
                components.html(
                    f"<script>window.open({json.dumps(dataset_url)}, '_blank');</script>",
                    height=0,
                    width=0,
                )

            st.divider()
            st.subheader(f"📂 {dataset_info['name']}")

            meta_badges = []
            if dataset_info.get("version"):
                meta_badges.append(f"**Version:** {dataset_info['version']}")
            if dataset_info.get("registered_name"):
                meta_badges.append(f"**Registered name:** {dataset_info['registered_name']}")
            if meta_badges:
                st.markdown(" • ".join(meta_badges))

            description_text = dataset_info.get("description") or "No description provided."
            st.write(description_text)

            hardware = dataset_info.get("hardware", {})
            modality = dataset_info.get("modality", {})
            task_env = dataset_info.get("task_env", {})
            engineering = dataset_info.get("engineering", {})
            stats = dataset_info.get("stats", {})

            sensors_list = modality.get("sensors") or []
            views_list = modality.get("viewpoints") or []
            sensors_text = ", ".join(sensors_list) if sensors_list else "Unspecified"
            views_text = ", ".join(views_list) if views_list else "Unspecified"
            scene_text = task_env.get("environment", "Unknown")
            language_text = stats.get("language_annotations") or "None"
            data_collect_text = stats.get("data_collect_method") or "Unspecified"
            episodes_text = stats.get("episodes") or "Unspecified"
            file_size_text = stats.get("file_size_gb") or "Unspecified"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**🤖 Robot:** {hardware.get('robot', 'Unknown')}")
                st.markdown(f"**🛠️ Morphology:** {hardware.get('morphology') or 'Unspecified'}")
                st.markdown(f"**🖐️ Gripper:** {hardware.get('end_effector', 'Unknown')}")
            with col2:
                st.markdown(f"**👁️ Sensors:** {sensors_text}")
                st.markdown(f"**📷 Views:** {views_text}")
                st.markdown(f"**🗺️ Scene Type:** {scene_text}")
                st.markdown(f"**🗣️ Language Annotations:** {language_text}")
            with col3:
                st.markdown(f"**⚙️ Action Space:** {engineering.get('format', 'Unknown')}")
                st.markdown(f"**⏱️ Control Freq:** {engineering.get('frequency', 'Unknown')}")
                st.markdown(f"**🧠 Data Collection:** {data_collect_text}")
                st.markdown(f"**📦 Episodes:** {episodes_text}")
                st.markdown(f"**💾 File Size (GB):** {file_size_text}")

            quality_flags = " | ".join(
                [
                    f"Suboptimal data: {stats.get('has_suboptimal') or 'Unknown'}",
                    f"Camera calibration: {stats.get('has_camera_calibration') or 'Unknown'}",
                    f"Proprioception: {stats.get('has_proprioception') or 'Unknown'}",
                ]
            )
            st.markdown(f"**Quality flags:** {quality_flags}")

            if dataset_url:
                st.markdown(f"[🔗 Go to Dataset Website]({dataset_url})")

            if dataset_info.get("citation"):
                with st.expander("📚 Citation"):
                    st.markdown(dataset_info["citation"])

            if dataset_info.get("latex_reference"):
                st.markdown(f"**LaTeX reference:** `{dataset_info['latex_reference']}`")
else:
    st.title("All datasets")
    st.markdown("Browse the filtered datasets in a tabular view.")

    table_rows = []
    for d in filtered_data:
        hardware = d.get("hardware", {})
        modality = d.get("modality", {})
        stats = d.get("stats", {})
        task_env = d.get("task_env", {})
        engineering = d.get("engineering", {})
        sensors_text = ", ".join(modality.get("sensors") or []) or "Unspecified"
        views_text = ", ".join(modality.get("viewpoints") or []) or "Unspecified"

        table_rows.append(
            {
                "ID": d["id"],
                "Dataset": d["name"],
                "Dataset URL": d.get("url") or "",
                "Robot": hardware.get("robot", "Unknown"),
                "Morphology": hardware.get("morphology") or "Unspecified",
                "Scene Type": task_env.get("environment", "Unknown"),
                "Action Space": engineering.get("format", "Unknown"),
                "Control Frequency": engineering.get("frequency", "Unknown"),
                "Data Collect Method": stats.get("data_collect_method") or "Unspecified",
                "Episodes": stats.get("episodes") or "Unspecified",
                "File Size (GB)": stats.get("file_size_gb") or "Unspecified",
                "Language Annotations": stats.get("language_annotations") or "None",
                "Sensors": sensors_text,
                "Views": views_text,
                "Version": d.get("version") or "",
                "Registered Name": d.get("registered_name") or "",
            }
        )

    df = pd.DataFrame(table_rows)
    column_config = {}
    if not df.empty:
        df.rename(columns={"Dataset URL": "Dataset Link"}, inplace=True)
        df["Dataset Link"] = df["Dataset Link"].apply(lambda url: url or None)
        column_config["Dataset Link"] = st.column_config.LinkColumn(
            "Dataset Link",
            help="Open dataset website in a new tab",
            display_text="Open",
        )
    st.dataframe(df, height=600, column_config=column_config)

