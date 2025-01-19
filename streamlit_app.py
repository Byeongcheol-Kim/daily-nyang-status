from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image
from PIL.ImageFile import ImageFile
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.services.structure_gemini import GeminiService

st.set_page_config(page_title="고양이 분석", layout="wide")

# 메인 대시보드
st.title("고양이 분석")

gemini_service = GeminiService()
# 이미지 업로드
image_file: Optional[UploadedFile] = st.file_uploader(
    "이미지를 업로드하면 AI가 고양이를 분석합니다",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if image_file:
    # 이미지와 분석 결과를 나란히 배치
    img_col, result_col = st.columns([1, 1.5])

    with img_col:
        print(image_file.name)
        image: ImageFile = Image.open(image_file)
        image.save("temp.jpg", "JPEG")
        st.image(image, caption="업로드된 이미지", use_container_width=True)

    with result_col:
        result: Dict[str, Any] = gemini_service.evaluate_image("temp.jpg")
        if result:
            if not result.get("is_cat", False):
                st.error("⚠️ 이미지에서 고양이를 찾을 수 없습니다.")
                st.stop()

            # 자세/행동 태그 표시
            st.subheader("🐱 고양이 자세/행동")
            tags_html = " ".join(
                [
                    f'<span style="background-color: #f0f2f6; padding: 0.2rem 0.6rem; '  # noqa: E501
                    f'border-radius: 1rem; margin-right: 0.3rem">{tag}</span>'
                    for tag in result.get("image_tags", [])
                ]
            )
            st.markdown(f"<p>{tags_html}</p>", unsafe_allow_html=True)

            # 색상 정보를 가로로 배치
            if result.get("color_codes"):
                st.subheader("🎨 고양이 색상 정보")
                color_cols = st.columns(
                    len(result["color_codes"]),
                    gap="small",
                    border=True,
                )  # noqa: E501
                for idx, color_code in enumerate(result["color_codes"]):
                    with color_cols[idx]:
                        st.color_picker(
                            f"색상 {idx + 1}",
                            color_code,
                            disabled=True,
                            key=f"color_{idx}",
                            label_visibility="visible",
                        )
                        st.write(color_code)
