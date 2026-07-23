import streamlit as st
import requests
import base64
import io
from PIL import Image
import pandas as pd
from typing import Tuple, Dict

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="Deteksi Gizi Makanan",
    page_icon="🍽️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "UI Streamlit untuk Deteksi Gizi Makanan (YOLO + LLM)."
    }
)

# -----------------------------
# Helper Functions
# -----------------------------
def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    if not data_url.startswith("data:"):
        raise ValueError("Bukan data URL yang valid")

    header, b64data = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]
    raw = base64.b64decode(b64data)
    return mime, raw


def post_image(api_url: str, file_name: str, file_bytes: bytes, mime: str, timeout: int = 60) -> Dict:
    files = {
        "image": (file_name, file_bytes, mime or "application/octet-stream")
    }

    response = requests.post(api_url, files=files, timeout=timeout)
    response.raise_for_status()
    return response.json()


def inject_style():
    st.markdown(
        """
        <style>
        .chip{
            display:inline-block;
            padding:6px 12px;
            margin:4px 6px 0 0;
            border-radius:16px;
            background:#EEF2FF;
            color:#3730A3;
            font-size:12px;
            border:1px solid #E0E7FF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_style()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("⚙️ Pengaturan")

    api_base = st.text_input(
        "API Base URL",
        value="https://food-api.onrender.com"
    )

    endpoint_path = st.text_input(
        "Endpoint Deteksi",
        value="/detect-gizi"
    )

    conf_filter = st.slider(
        "Filter Confidence",
        0.0,
        1.0,
        0.0,
        0.05
    )

    render_markdown_table = st.toggle(
        "Render Tabel Gizi sebagai Markdown",
        value=True
    )

    st.caption("Pastikan API sudah berjalan.")

# -----------------------------
# Header
# -----------------------------
st.title("🍽️ Deteksi Gizi Makanan")

st.write(
    "Unggah gambar makanan, kemudian sistem akan mendeteksi makanan "
    "menggunakan YOLO dan menghasilkan informasi gizi dari LLM."
)

# -----------------------------
# Input
# -----------------------------
input_method = st.radio(
    "Pilih metode input",
    ["📁 Upload Gambar", "📸 Ambil Foto"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None
cam_image = None

if input_method == "📁 Upload Gambar":

    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

else:

    cam_image = st.camera_input(
        "Ambil Foto",
        label_visibility="collapsed"
    )

preview = uploaded or cam_image

col1, col2 = st.columns([3, 2], vertical_alignment="bottom")

with col1:

    if preview:
        st.image(
            preview,
            caption="Pratinjau Gambar",
            width="stretch"
        )

    detect_btn = st.button(
        "🔎 Deteksi Gizi",
        type="primary",
        width="stretch",
        disabled=not preview
    )

# -----------------------------
# Proses
# -----------------------------
if detect_btn:

    api_url = api_base.rstrip("/") + "/" + endpoint_path.lstrip("/")

    try:

        with st.spinner("Memproses gambar..."):

            source = uploaded or cam_image

            file_bytes = source.read()

            mime = source.type or "image/jpeg"

            file_name = getattr(source, "name", "camera.jpg")

            result = post_image(
                api_url,
                file_name,
                file_bytes,
                mime,
                timeout=90
            )

        if not isinstance(result, dict):
            st.error("Format respon API tidak valid.")
            st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        img_data = result.get("image")

        objects = result.get("objects", [])

        gizi = result.get("gizi") or result.get("response", "")

        # -----------------------------
        # Gambar hasil
        # -----------------------------
        st.subheader("📷 Hasil Deteksi")

        if img_data:

            mime, img_bytes = parse_data_url(img_data)

            image = Image.open(io.BytesIO(img_bytes))

            st.image(
                image,
                caption="Bounding Box",
                width="stretch"
            )

            st.download_button(
                "⬇️ Download Hasil",
                img_bytes,
                file_name="hasil_deteksi.jpg",
                mime=mime,
                width="stretch"
            )

        else:
            st.info("Tidak ada gambar hasil.")

        # -----------------------------
        # Detail Objek
        # -----------------------------
        st.subheader("📦 Detail Objek")

        filtered = [
            o for o in objects
            if float(o.get("confidence", 0)) >= conf_filter
        ]

        if filtered:

            rows = []

            for obj in filtered:

                bbox = obj.get("bbox", [])

                area = None

                if len(bbox) == 4:
                    area = max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1])

                rows.append({
                    "Nama": obj.get("nama"),
                    "Confidence": round(float(obj.get("confidence", 0)), 4),
                    "BBox": bbox,
                    "Luas(px²)": area
                })

            df = pd.DataFrame(rows)

            st.dataframe(
                df,
                width="stretch",
                hide_index=True
            )

        else:

            st.info("Tidak ada objek yang memenuhi filter.")

        # -----------------------------
        # Informasi Gizi
        # -----------------------------
        st.subheader("🥗 Informasi Gizi")

        if gizi:

            if render_markdown_table:
                st.markdown(gizi)
            else:
                st.text(gizi)

        else:
            st.info("Informasi gizi tidak tersedia.")

        st.toast("Selesai memproses", icon="✅")

    except requests.exceptions.ConnectionError:
        st.error("Tidak dapat terhubung ke API.")

    except requests.exceptions.Timeout:
        st.error("Request timeout.")

    except requests.HTTPError as e:

        try:
            err = e.response.json()
        except Exception:
            err = e.response.text

        st.error(f"HTTP {e.response.status_code}: {err}")

    except Exception as e:
        st.exception(e)
