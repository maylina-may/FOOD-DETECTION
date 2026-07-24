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
    """
    Mengubah data URL menjadi (mime, bytes)
    """
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


def chip(text: str) -> str:
    return f"""
    <span class="chip">{text}</span>
    """


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
            white-space:nowrap;
        }

        .metric-card{
            padding:14px 16px;
            border-radius:12px;
            border:1px solid #E5E7EB;
            background:#FFFFFF;
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
        value="https://food-api.onrender.com",
        help="Alamat API Flask."
    )

    endpoint_path = st.text_input(
        "Endpoint Deteksi",
        value="/detect-gizi"
    )

    conf_filter = st.slider(
        "Filter Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
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
    "Unggah foto makanan. Sistem akan mendeteksi objek menggunakan YOLO "
    "dan menghasilkan informasi kandungan gizi menggunakan Large Language Model."
)

# -----------------------------
# Upload
# -----------------------------
input_method = st.radio(
    "Pilih metode input",
    ["📁 Upload Gambar", "📸 Ambil Foto"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None
camera = None

if input_method == "📁 Upload Gambar":

    uploaded = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

else:

    camera = st.camera_input(
        "Ambil Foto",
        label_visibility="collapsed"
    )

preview = uploaded or camera

col1, col2 = st.columns([3, 2], vertical_alignment="bottom")

with col1:

    if preview:
        st.image(
            preview,
            caption="Pratinjau Gambar",
            width="stretch"
        )

    st.write("")
    st.write("")

    detect_btn = st.button(
        "🔎 Deteksi Gizi",
        type="primary",
        width="stretch",
        disabled=not preview
    )

# -----------------------------
# Hasil Deteksi
# -----------------------------
if detect_btn and preview:

    api_url = api_base.rstrip("/") + "/" + endpoint_path.lstrip("/")

    try:

        with st.spinner("Memproses gambar..."):

            file_bytes = preview.read()
            mime = preview.type or "image/jpeg"
            file_name = getattr(preview, "name", "camera.jpg")

            result = post_image(
                api_url,
                file_name,
                file_bytes,
                mime,
                timeout=90
            )

        if not isinstance(result, dict):
            st.error("Format respon API tidak valid.")

        elif "error" in result:

            st.error(result["error"])

        else:

            img_data = result.get("image")
            objects = result.get("objects", [])
            gizi = result.get("gizi") or result.get("response", "")

            col_result, _ = st.columns([3, 2])

            with col_result:

                st.subheader("📷 Hasil Deteksi")

                if img_data:

                    try:

                        mime, img_bytes = parse_data_url(img_data)

                        image = Image.open(io.BytesIO(img_bytes))

                        st.image(
                            image,
                            caption="Bounding Box",
                            width="stretch"
                        )

                        st.download_button(
                            "📥 Unduh Gambar",
                            data=img_bytes,
                            file_name="hasil_deteksi.jpg",
                            mime=mime,
                            width="stretch"
                        )

                    except Exception as e:
                        st.warning(f"Gagal menampilkan gambar: {e}")

                else:
                    st.info("Tidak ada gambar hasil.")

            # -----------------------------
            # Detail Objek
            # -----------------------------
            st.subheader("📦 Detail Objek")

            filtered = [
                obj
                for obj in objects
                if float(obj.get("confidence", 0)) >= conf_filter
            ]

            if filtered:

                def bbox_area(bbox):
                    try:
                        x1, y1, x2, y2 = bbox
                        return max(0, x2 - x1) * max(0, y2 - y1)
                    except:
                        return None

                rows = []

                for obj in filtered:

                    bbox = obj.get("bbox", [None] * 4)

                    rows.append({
                        "Nama": obj.get("nama", "-"),
                        "Confidence": round(float(obj.get("confidence", 0)), 4),
                        "BBox": bbox,
                        "Luas (px²)": bbox_area(bbox)
                    })

                df = pd.DataFrame(rows)

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info("Tidak ada objek yang memenuhi filter confidence.")

            # -----------------------------
            # Informasi Gizi
            # -----------------------------
            st.subheader("🥗 Kandungan Gizi")

            if gizi:

                if render_markdown_table:
                    st.markdown(gizi)
                else:
                    st.text(gizi)

            else:

                st.info("Informasi gizi belum tersedia.")

        st.toast("Deteksi berhasil", icon="✅")

    except requests.exceptions.ConnectionError:
        st.error(f"Tidak dapat terhubung ke API:\n{api_url}")

    except requests.exceptions.Timeout:
        st.error("Permintaan ke API melebihi batas waktu.")

    except requests.exceptions.HTTPError as e:

        try:
            err = e.response.json()
        except:
            err = e.response.text

        st.error(f"HTTP {e.response.status_code}\n\n{err}")

    except Exception as e:
        st.exception(e)
