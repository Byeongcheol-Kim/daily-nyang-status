import json
import os
from typing import Any, Dict, TypedDict

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class ResponseSchema(TypedDict):
    is_cat: bool
    image_tags: list[str]
    color_codes: list[str]
    breed_type: str
    age: int


class GeminiService:
    # Create the model
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            enum=[],
            required=["is_cat"],
            properties={
                "is_cat": content.Schema(
                    type=content.Type.BOOLEAN,
                ),
                "image_tags": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "color_codes": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "breed_type": content.Schema(
                    type=content.Type.STRING,
                ),
                "age": content.Schema(
                    type=content.Type.INTEGER,
                ),
            },
        ),
        "response_mime_type": "application/json",
    }
    SYSTEM_PROMPT = """
    # 고양이 자세 판별 기준\n\n
    ## 1. 기본 자세 분류\n\n
    ### 1.1 앉은 자세\n
    - **SIT**: 앞다리를 가지런히 모으고 앉아있는 자세\n
    - **BREAD_SIT**: 앞다리를 몸통 아래에 숨기고 웅크린 자세\n
    - **SPINKS_SIT**: 엎드린 채 앞다리를 앞으로 뻗은 자세\n
    - **SIDE_SIT**: 몸을 옆으로 기울여 앉아있는 자세\n\n
    ### 1.2 서 있는 자세\n
    - **STAND**: 네 다리로 바닥을 딛고 서 있는 자세\n
    - **BOUND_STAND**: 몸을 살짝 숙이거나 다리를 모으고 주변을 살피는 자세\n
    - **BACK_STAND**: 뒷다리로만 서서 앞다리를 들고 있는 자세\n\n
    ### 1.3 누워 있는 자세\n
    - **LIE**: 배를 바닥에 대고 엎드려 있는 자세\n
    - **SIDE_LIE**: 몸을 옆으로 쭉 뻗고 누워있는 자세\n
    - **BALL_LIE**: 배를 위로 향하게 하고 누워있는 자세\n
    - **CURL_LIE**: 몸을 둥글게 말고 자는 자세\n\
    ### 1.4 움직이는 자세\n
    - **WALKING**: 네 다리를 움직이며 걷는 자세\n
    - **RUNNING**: 공중에 떠 있는 듯한 달리는 자세\n- **JUMPING**: 뛰어오르는 순간의 자세\n
    - **HUNTING**: 몸을 낮추고 먹잇감을 노리는 자세\n\n
    ### 1.5 특정 행동 자세\n
    - **GROOMING**: 혀로 털을 핥는 자세\n
    - **SNIFF**: 입을 크게 벌리고 하품하는 자세\n
    - **STRETCH**: 몸을 쭉 뻗는 자세\n
    - **PLAY**: 장난감을 잡거나 쫓는 자세\n\n
    ## 2. 촬영 구도\n\n
    ### 2.1 앵글\n
    - **FRONT_ANGLE**: 고양이의 정면을 향해 촬영\n
    - **SIDE_ANGLE**: 고양이의 옆모습 촬영\n
    - **BACK_ANGLE**: 고양이의 뒷모습 촬영\n
    - **TOP_VIEW**: 위에서 내려다보며 촬영\n
    - **LOW_ANGLE**: 아래에서 올려다보며 촬영\n\n
    ### 2.2 초점\n
    - **FACE_FOCUS**: 고양이 얼굴에 초점\n
    - **BODY_FOCUS**: 고양이 전체 모습에 초점\n
    - **PART_FOCUS**: 눈이나 발 등 특정 부위에 초점\n\n
    ### 2.3 촬영 거리\n
    - **CLOSE_UP**: 얼굴이나 특정 부위를 아주 가깝게 촬영\n
    - **MID_SHOT**: 상반신이나 전신 일부를 포함한 촬영\n
    - **FULL_SHOT**: 전신과 주변 환경을 모두 포함한 촬영\n\n
    # 요구사항\n
    1. 사진에 고양이가 있는지를 판단해서 is_cat 필드에 true 또는 false를 반환합니다.
      고양이가 여러마리 있는 경우에는 true를 반환합니다.\n
    2. 고양이 자세 판별 기준에 따라 고양이 자세를 분류하고 image_tags 필드에 태그를 부여합니다.
      태그는 중복으로 부여할 수 있습니다.
      고양이가 여러마리 있는 경우에는 가장 큰 고양이를 기준으로 합니다.\n
    기대하는 응답 ex) "SIT", "FRONT_ANGLE", "FACE_FOCUS", "FULL_SHOT"\n
    3. 고양이 색상을 분류하고 color_codes 필드에 색상 코드를 부여합니다. 색상 코드는 중복으로 부여할 수 있습니다.
      고양이가 여러마리 있는 경우에는 가장 큰 고양이를 기준으로 합니다.\n
    기대하는 응답 ex) "#000000", "#FFFFFF", "#FF0000"\n
    4. 고양이 종류를 분류하고 breed_type 필드에 종류를 부여합니다.
      고양이가 여러마리 있는 경우에는 가장 큰 고양이를 기준으로 합니다.\n
    기대하는 응답 ex) "러시안블루", "브리티시쇼트헤어", "랙돌", "랙돌"\n
    5. 고양이 나이를 분류하고 age 필드에 나이(개월수)를 부여합니다.
      고양이가 여러마리 있는 경우에는 가장 큰 고양이를 기준으로 합니다.\n
    기대하는 응답 ex) 2, 4, 12, 45\n
    """
    MODEL = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=GENERATION_CONFIG,
        system_instruction=SYSTEM_PROMPT,
    )
    PROMPT = "사진을 보고 응답 스키마에 따라 응답을 줘"

    def evaluate_image(self, image_file: str) -> Dict[str, Any]:
        input_file = genai.upload_file(path=image_file, mime_type="image/jpeg")

        result = self.MODEL.generate_content(
            [
                input_file,
                self.PROMPT,
            ]
        )
        return json.loads(result.text)  # type: ignore
