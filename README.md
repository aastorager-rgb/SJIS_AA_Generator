SJIS_AA_Generator

AA_for_ComfyUI의 간략화/포터블 버전입니다.
- 눈 인식을 자동 인식에서 수동 색칠로 변경하였습니다.


다운로드 방법(.py):
1. 파이썬 설치
2. 명령 프롬포트(cmd)에서 pip install PyQt5 opencv-python torch pandas numpy pillow 로 필요한 라이브러리 설치
3. 생행
  
사용법:
<img width="1920" height="1032" alt="스크린샷 2026-02-28 014553" src="https://github.com/user-attachments/assets/1250daaf-88ea-4702-a942-aaf270f3b1b1" />
1. 이미지 선택
   1-1. 창에 뜬 이미지의 눈 부분에 마스킹 영역 색칠
2. char_list_freq.csv, char_tone.txt, Saitamaar.ttf 다운로드 (아마 HeadKansen도 가능할 듯) 후 선택
3. 이미지 선화 모드 설정 및 파라미터 조정
4. 선화 줄이기 알고리즘 선택
5. 기본 파라미터 조정
6. Generate AA 클릭
