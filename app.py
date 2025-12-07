import streamlit as st
import streamlit.components.v1 as components
import json
import base64
from pathlib import Path
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="RoboData Atlas", layout="wide")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    with open('data/datasets.json', 'r') as f:
        return json.load(f)
BASE_DIR = Path(__file__).parent

ROBOT_IMAGE_FILES = {
    "Franka Emika Panda": "assets/images/Franka-Emika-Panda.png",
    "Fetch Mobile Manipulator": "assets/images/Fetch-Mobile-Manipulator.png",
    "WidowX 250s": "assets/images/WidowX-250.png",
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
all_robots = sorted(list(set([d['hardware']['robot'] for d in data])))
all_envs = sorted(list(set([d['task_env']['environment'] for d in data])))

selected_robots = st.sidebar.multiselect("Select Robot Hardware", all_robots, default=all_robots)
selected_envs = st.sidebar.multiselect("Select Environment", all_envs, default=all_envs)

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
        env_name = d['task_env']['environment']
        fmt_name = d['engineering']['format']

        dataset_title_lines = [
            d['name'],
            f"Robot: {robot_name}",
            f"Environment: {env_name}",
            f"Format: {fmt_name}",
        ]
        dataset_title = "\n".join(dataset_title_lines)
        add_node(dataset_node_id, d['name'], "#00C49F", size=35, title=dataset_title, image="")

        robot_id = f"robot_{robot_name}"
        robot_image = robot_images.get(robot_name)
        if robot_image:
            add_node(
                robot_id,
                "",
                color={"border": "#FFBB28", "background": "#ffffff"},
                shape="circularImage",
                size=42,
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

        env_id = f"env_{env_name}"
        add_node(env_id, "", "#FF8042", size=18, title=f"Environment: {env_name}")
        edges.append(
            Edge(
                source=dataset_node_id,
                target=env_id,
                color="#FF8042",
                title=f"Environment: {env_name}",
                arrows="to",
            )
        )

        fmt_id = f"fmt_{fmt_name}"
        add_node(fmt_id, "", "#8884d8", size=18, title=f"Format: {fmt_name}")
        edges.append(
            Edge(
                source=dataset_node_id,
                target=fmt_id,
                color="#8884d8",
                title=f"Format: {fmt_name}",
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
            Environment attribute
        </div>
        <div class="atlas-legend__item">
            <span class="atlas-legend__swatch" style="background-color:#8884d8;"></span>
            Format attribute
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("Interactive map of robotics datasets. **Drag nodes** to explore connections.")

    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics=True,
        hierarchy=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
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
            st.write(dataset_info['description'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**🤖 Robot:** {dataset_info['hardware']['robot']}")
                st.markdown(f"**🖐️ Gripper:** {dataset_info['hardware']['end_effector']}")
            with col2:
                st.markdown(f"**👁️ Sensors:** {', '.join(dataset_info['modality']['sensors'])}")
                st.markdown(f"**🎥 Views:** {', '.join(dataset_info['modality']['viewpoints'])}")
            with col3:
                st.markdown(f"**💾 Format:** {dataset_info['engineering']['format']}")
                st.markdown(f"**⚡ Freq:** {dataset_info['engineering']['frequency']}")

            if dataset_url:
                st.markdown(f"[🔗 Go to Dataset Website]({dataset_url})")
else:
    st.title("All datasets")
    st.markdown("Browse the filtered datasets in a tabular view.")

    table_rows = [
        {
            "ID": d["id"],
            "Name": d["name"],
            "Robot": d['hardware']['robot'],
            "Environment": d['task_env']['environment'],
            "Format": d['engineering']['format'],
            "Frequency": d['engineering']['frequency'],
            "Sensors": ", ".join(d['modality']['sensors']),
            "Views": ", ".join(d['modality']['viewpoints']),
            "URL": d['url'],
        }
        for d in filtered_data
    ]

    st.dataframe(pd.DataFrame(table_rows))