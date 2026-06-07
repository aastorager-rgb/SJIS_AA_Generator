import sys
import os
import re
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ==========================================
# Script directory (for default file paths)
# ==========================================
if getattr(sys, 'frozen', False):
    _SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _rel(filename):
    return os.path.join(_SCRIPT_DIR, filename)

# ==========================================
# 1. Global Settings & Typography Helpers
# ==========================================
FONT_SIZE = 16
LINE_SPACING = 2
ROW_HEIGHT = FONT_SIZE + LINE_SPACING

DENSE_KANJI = set(['瀟', '憎', '浄', '占', '李', '斗', '狄', '灘', '濾', '鼎', '撼'])
EYE_UP_LEFT = set(list("だ灯衍行仍了乍仡乞云伝芸茫忙它佗俐仗なｨｪｵｴﾃﾇﾏﾓfrvx{"))
EYE_UP_CENTER = set(list("不示宍亦兀亢万迩尓禾乏弌弍弐泛夾赱符≡女乍气旡まみてテチｪｫｭｮｰｴｵｻﾁﾆﾓﾕﾖェュョヵ="))
EYE_UP_RIGHT = set(list("豺犾狄勿下卞抃圷圦坏心沁气汽斥拆仔竹刃刈付以雫爿なうかて刈ﾊ､ｧｩｪｫｬｭｱｦｳｵｷｹﾁﾃﾇﾕﾖェュヵヶァvx}"))
EYE_LOW_LEFT = set(list("芍弋爪心父戈弌弍弐式汽辷込乂癶廴匕丈叱杙之比仆トヽヾゝゞﾞｬｾﾀﾊﾋﾏﾓ㌧､tVU{:xviｰ"))
EYE_LOW_RIGHT = set(list("歹万久刋升刈乃汐沙少炒梦斗孑才必瓜欠次亥圦乂ノソルツ八ｧｨｩｫｱｦｳｶｸｹｻｼｽｾﾀﾁﾂﾃﾅﾇﾈﾉﾊﾑﾒﾔﾗﾘﾙﾚﾜﾝァヵ㌧㌢㌃㍗㌣㌻jivxVU'},;"))
EYE_IDIOMS_ALL = EYE_UP_LEFT | EYE_UP_CENTER | EYE_UP_RIGHT | EYE_LOW_LEFT | EYE_LOW_RIGHT

DOT_PUNCT_CHARS = set(list(".,，:’;¨･・…'‘’“”；ﾞﾟ．′xｘXＸ_ ﾟ‘‘｡。’~`"))
VERT_CHARS = set(['|', '│', '┃', '!', 'l', 'I', '1', 'ｉ', 'Ｉ', '！', '｜', ':', '⋮'])
HORIZ_CHARS = set(list("-￣_＿ーT＋+┐└┌┘├┤┬┴┼/＼／＜＞ヽヾゝゞ^v~〜")) 

def get_char_width(char, font, draw_ctx):
    if hasattr(draw_ctx, 'textlength'): return int(draw_ctx.textlength(char, font=font))
    else: return int(draw_ctx.textbbox((0,0), char, font=font)[2])

def calculate_orientation_map(img):
    img_f = img.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_f, (3, 3), 0.7)
    gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0); gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    vx = cv2.boxFilter(gx**2 - gy**2, -1, (5, 5)); vy = cv2.boxFilter(2 * gx * gy, -1, (5, 5))
    theta = 0.5 * np.arctan2(vy, vx)
    theta[vx >= 0] += (np.pi / 2)
    return np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32), theta

# ==========================================
# 2. Algorithm Core (SJIS Pipeline)
# ==========================================
class SJISPipeline:
    def __init__(self):
        self._char_list_cache = None
        self._char_groups_cache = None
        self._tone_chars_cache = None
        self._current_csv_path = None
        self._current_txt_path = None
        self._current_font_path = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._fw_width = 16
        self._hw_width = 8
        self._dot_width = 4

    def extract_lines(self, img_rgb, text_lines, method, threshold, line_thickness, kmeans_k, invert_output):
        target_h = text_lines * ROW_HEIGHT
        target_w = int(target_h * (img_rgb.shape[1] / img_rgb.shape[0]))
        
        if method == "Segmentation (K-means)":
            img_res = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            Z = img_res.reshape((-1, 3)).astype(np.float32)
            _, labels, _ = cv2.kmeans(Z, kmeans_k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            labels_2d = labels.reshape((target_h, target_w))
            edges = (labels_2d != np.roll(labels_2d, 1, axis=0)) | (labels_2d != np.roll(labels_2d, 1, axis=1))
            binary = edges.astype(np.uint8) * 255
            binary[0, :] = 0; binary[:, 0] = 0
        else:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            gray = cv2.bilateralFilter(gray, 5, 50, 50)
            
            if method == "XDoG (Soft Sketch)":
                sigma = max(0.5, line_thickness * 0.5)
                g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
                g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma * 2.0)
                tau = (threshold / 255.0) * 10.0 - 5.0 
                xdog = g1 - 0.98 * g2
                binary = np.where(xdog < tau, 255, 0).astype(np.uint8)
                
            elif method == "Laplacian (Brush/Ink)":
                blur = cv2.GaussianBlur(gray, (0,0), max(0.5, line_thickness * 0.5))
                lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
                lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
                _, binary = cv2.threshold(lap, threshold, 255, cv2.THRESH_BINARY)
                
            elif method == "Sobel (Directional Edge)":
                blur = cv2.GaussianBlur(gray, (0,0), max(0.5, line_thickness * 0.5))
                sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                sobel = np.uint8(np.clip(sobel, 0, 255))
                _, binary = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)
                
            elif method == "Canny (Hard Edges)": 
                binary = cv2.Canny(gray, threshold // 2, threshold)
                
            elif method == "Adaptive Threshold":
                c_val = (threshold - 127) / 5.0 + 5.0 
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, c_val)
                
            else: 
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        scale = target_h / (40.0 * ROW_HEIGHT); act_t = max(1, int(round(line_thickness * scale)))
        if act_t > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (act_t, act_t))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary if invert_output else 255 - binary

    def process_thinning(self, binary_img, clean_strength, method):
        binary = binary_img.copy()
        if np.mean(binary) > 127: binary = 255 - binary
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        
        if clean_strength > 0:
            min_area = int(clean_strength * 10) 
            if min_area > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                cleaned_binary = np.zeros_like(binary)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        cleaned_binary[labels == i] = 255
                binary = cleaned_binary
        
        if "None" in method:
            return binary
        try:
            if "K3M" in method: 
                return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            elif "Guo" in method: 
                return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            elif "KMM" in method:
                pass
        except: pass
        

        if "KMM" in method:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        elif "K3M" in method:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
        skel = np.zeros(binary.shape, np.uint8)
        temp_img = binary.copy()
        while True:
            eroded = cv2.erode(temp_img, kernel)
            temp = cv2.subtract(temp_img, cv2.dilate(eroded, kernel))
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0: break
        return skel

    def load_resources(self, char_list_path, font_path, char_tone_path):
        if not os.path.exists(font_path): raise Exception(f"폰트 파일 누락: {font_path}")
        try: font = ImageFont.truetype(font_path, FONT_SIZE)
        except Exception as e: raise Exception(f"폰트 로드 실패: {str(e)}")

        dummy_draw = ImageDraw.Draw(Image.new("L", (1,1)))
        self._fw_width = max(1, get_char_width('　', font, dummy_draw))
        self._hw_width = max(1, get_char_width(' ', font, dummy_draw))
        self._dot_width = max(1, get_char_width('.', font, dummy_draw))
        
        if self._current_txt_path != char_tone_path:
            tone_list = []
            max_t = 0.01
            if os.path.exists(char_tone_path):
                with open(char_tone_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.replace('\n', '').replace('\r', '').split('\t')
                        if len(parts) >= 2 and parts[0] != "Chars":
                            try:
                                t = float(parts[1]); cw = get_char_width(parts[0], font, dummy_draw)
                                if cw > 0:
                                    tone_list.append((t, parts[0], cw))
                                    if t > max_t: max_t = t
                            except: pass
            self._tone_chars_cache = [(t/max_t, c, cw) for t, c, cw in tone_list]
            self._current_txt_path = char_tone_path

        if self._char_groups_cache is None or self._current_font_path != font_path or self._current_csv_path != char_list_path:
            chars, raw_freqs = [], []
            if os.path.exists(char_list_path):
                lines = []
                try: 
                    with open(char_list_path, 'r', encoding='cp932') as f: lines = f.readlines()
                except: 
                    with open(char_list_path, 'r', encoding='utf-8') as f: lines = f.readlines()
                if lines:
                    for line in lines[1:]:
                        parts = line.rstrip('\n\r').split(',')
                        if len(parts) >= 2:
                            try:
                                char_str = ",".join(parts[1:-1]) if len(parts) > 2 else parts[0]
                                if len(char_str) >= 2 and char_str.startswith('"') and char_str.endswith('"'): char_str = char_str[1:-1]
                                chars.append(char_str if char_str and char_str != 'nan' else ' ')
                                raw_freqs.append(float(parts[-1]))
                            except: pass
            if not chars: return None
            
            max_f = max(raw_freqs) if raw_freqs else 1.0
            freq_scores = [np.log1p(f) / np.log1p(max_f) for f in raw_freqs]
            
            char_data_cache = []
            char_groups = {}
            # 공간 여유를 주어 Missing penalty를 경감시키기 위한 Dilation 커널 준비 (Condition 3 적용)
            kernel_dil = np.ones((3, 3), np.uint8)
            
            for i, c in enumerate(chars):
                w = get_char_width(c, font, dummy_draw)
                flags = 0
                if c in EYE_IDIOMS_ALL: flags |= 1
                if c in EYE_UP_LEFT or c in EYE_LOW_LEFT: flags |= 2
                if c in EYE_UP_CENTER: flags |= 4
                if c in EYE_UP_RIGHT or c in EYE_LOW_RIGHT: flags |= 8
                if c in EYE_UP_LEFT or c in EYE_UP_CENTER or c in EYE_UP_RIGHT: flags |= 16
                if c in EYE_LOW_LEFT or c in EYE_LOW_RIGHT: flags |= 32
                if c in DOT_PUNCT_CHARS: flags |= 64
                
                if c.strip() and w > 0:
                    img_char = Image.new("L", (w, ROW_HEIGHT), 0)
                    ImageDraw.Draw(img_char).text((0, 0), c, font=font, fill=255)
                    char_arr = np.array(img_char).astype(np.uint8)
                    c_cos, c_sin, _ = calculate_orientation_map(char_arr)
                    c_mask = (char_arr > 0).astype(np.float32)
                    
                    # 팽창(Dilation)된 마스크를 캐시에 함께 생성하여 저장
                    c_mask_dil = cv2.dilate((char_arr > 0).astype(np.uint8), kernel_dil, iterations=1).astype(np.float32)
                    
                    char_data_cache.append({'char': c, 'mask': c_mask, 'mask_dilated': c_mask_dil, 'cos_strict': c_cos * c_mask, 'sin_strict': c_sin * c_mask, 'width': w, 'freq_score': freq_scores[i], 'ink': np.sum(c_mask), 'flags': flags})
                else: 
                    char_data_cache.append({'char': c, 'mask': None, 'width': max(1, w), 'freq_score': freq_scores[i], 'ink': 0, 'flags': flags})

            for idx, data in enumerate(char_data_cache):
                cw = data['width']
                if data['mask'] is None: continue
                if cw not in char_groups:
                    char_groups[cw] = {'indices': [], 'masks': [], 'masks_dilated': [], 'cos_stricts': [], 'sin_stricts': [], 'inks': [], 'freqs': [], 'flags': [], 'is_dense': []}
                char_groups[cw]['indices'].append(idx)
                char_groups[cw]['masks'].append(torch.from_numpy(data['mask']).unsqueeze(0))
                char_groups[cw]['masks_dilated'].append(torch.from_numpy(data['mask_dilated']).unsqueeze(0))
                char_groups[cw]['cos_stricts'].append(torch.from_numpy(data['cos_strict']).unsqueeze(0))
                char_groups[cw]['sin_stricts'].append(torch.from_numpy(data['sin_strict']).unsqueeze(0))
                char_groups[cw]['inks'].append(data['ink'])
                char_groups[cw]['freqs'].append(data['freq_score'])
                char_groups[cw]['flags'].append(data['flags'])
                char_groups[cw]['is_dense'].append(data['char'] in DENSE_KANJI)
            
            for cw in char_groups:
                for k in ['masks', 'masks_dilated', 'cos_stricts', 'sin_stricts']: char_groups[cw][k] = torch.stack(char_groups[cw][k]).to(self._device).to(torch.float32)
                for k in ['inks', 'freqs']: char_groups[cw][k] = torch.tensor(char_groups[cw][k], dtype=torch.float32, device=self._device)
                char_groups[cw]['flags'] = torch.tensor(char_groups[cw]['flags'], dtype=torch.int32, device=self._device)
                char_groups[cw]['is_dense'] = torch.tensor(char_groups[cw]['is_dense'], dtype=torch.bool, device=self._device)
                
            self._char_list_cache = chars
            self._char_groups_cache = char_groups
            self._char_data_cache = char_data_cache
            self._current_font_path = font_path
            self._current_csv_path = char_list_path

        return self._char_data_cache, font

    def _get_gap_string(self, target_w, is_start, last_char='', exact_match=False):
        target_w = round(target_w)
        if target_w <= 0: return "", 0.0

        fw_w = self._fw_width
        hw_w = self._hw_width
        dt_w = self._dot_width

        best_diff = float('inf')
        best_cost = float('inf')
        best_counts = (0, 0, 0)
        
        # [추가] 3픽셀 이하의 가로 오차는 페널티 없이 허용하여 점(.) 사용을 최소화
        max_allowed_diff = 0 if exact_match else 3
        diff_penalty = 1000 if exact_match else 40

        max_fw = int((target_w + max_allowed_diff) // fw_w) + 1
        for fw in range(max_fw, -1, -1):
            rem1 = target_w - fw * fw_w
            
            # [추가] 반각 공백은 유연하게 탐색하되 Cost로 연속 배치를 억제
            max_hw = int(max(0, rem1 + max_allowed_diff) // hw_w) + 1
            for hw in range(max_hw, -1, -1):
                rem2 = rem1 - hw * hw_w
                dt_base = int(round(rem2 / dt_w)) if rem2 > 0 else 0
                
                for d in [max(0, dt_base-1), dt_base, dt_base+1]:
                    # [추가] 맨 왼쪽 공백이 아니면 점(.)은 최대 2개로 강제 제한
                    max_d = 10 if is_start else 2
                    if d > max_d:
                        continue
                        
                    cw = fw * fw_w + hw * hw_w + d * dt_w
                    diff = abs(target_w - cw)
                    
                    if exact_match and diff > 0 and best_diff == 0:
                        continue
                        
                    # 3픽셀 이내의 오차일 경우 페널티를 대폭 감소시켜 불필요한 점(.) 추가 방지
                    if diff <= max_allowed_diff:
                        error_score = diff * 2 
                    else:
                        error_score = diff * diff_penalty
                    
                    # 반각 공백(hw)이 2개 이상 들어갈 경우 기하급수적인 페널티 부과 (연속 억제)
                    hw_cost = (hw * 5) if hw <= 1 else (hw * 50)
                    dt_cost = d * 15
                    fw_cost = fw * 1
                    
                    total_score = error_score + hw_cost + dt_cost + fw_cost
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_cost = total_score
                        best_counts = (fw, hw, d)
                    elif diff == best_diff and total_score < best_cost:
                        best_cost = total_score
                        best_counts = (fw, hw, d)
        
        c_fw, c_hw, c_dt = best_counts
        actual_w = float(c_fw * fw_w + c_hw * hw_w + c_dt * dt_w)
        
        res = ""
        
        # 1. 점(.)이 할당된 경우 최우선으로 전면에 배치
        while c_dt > 0:
            res += '.'
            c_dt -= 1
            
        # 2. 기존 코드를 활용한 다이나믹 전각/반각 공백 배치 로직 복구
        total_spaces = c_fw + c_hw
        for _ in range(total_spaces):
            score_fw = c_fw * 1.0 + (0.5 if res and res[-1] != '　' else 0)
            score_hw = c_hw * 1.2 + (0.5 if res and res[-1] != ' ' else 0)
            
            can_hw = c_hw > 0
            if is_start and not res: can_hw = False
            if res and res[-1] == ' ': can_hw = False
            if last_char == ' ' and not res: can_hw = False
            
            can_fw = c_fw > 0
            
            best_choice = None
            best_s = -1
            
            if can_fw and score_fw > best_s: 
                best_choice = '　'
                best_s = score_fw
            if can_hw and score_hw > best_s: 
                best_choice = ' '
                best_s = score_hw
                
            if best_choice == '　':
                res += '　'
                c_fw -= 1
            elif best_choice == ' ':
                res += ' '
                c_hw -= 1
            else:
                if c_hw > 0: 
                    res += ' '
                    c_hw -= 1
                elif c_fw > 0: 
                    res += '　'
                    c_fw -= 1
                    
        # 후처리: 첫 시작이 반각 공백이면 안전망으로 언더바(_) 교체
        if is_start and res.startswith(' '):
            res = '_' + res[1:]
        res = res.replace('  ', ' _')
        
        return res, actual_w
    
    def _solve_stripe_sequential(self, scores, char_data_list, w, spacing, last_ink_x, is_roi_mask, prev_anchors):
        cur_x = 0.0
        line = ""
        placements = []
        is_l = True
        last_char = ''
        
        horiz_indices = [i for i, d in enumerate(char_data_list) if d['char'] in HORIZ_CHARS]

        for pa_x, pa_char, pa_w in prev_anchors:
            ix = int(round(pa_x))
            if 0 <= ix < w:
                search_min = max(0, ix - 1)
                search_max = min(w, ix + 2)
                pa_idx = next((i for i, d in enumerate(char_data_list) if d['char'] == pa_char), -1)
                
                # [Condition 1 적용] 이전 줄에서 하단경계 문자(＿, _)가 찍힌 곳과 겹치면 상단 문자(￣, T, ^) 억제
                if pa_char in ['＿', '_']:
                    top_chars = ['￣', 'T', '^', 'ー']
                    top_indices = [i for i, d in enumerate(char_data_list) if d['char'] in top_chars]
                    if top_indices:
                        scores[search_min:search_max, top_indices] -= 9999.0

                if pa_idx != -1:
                    best_natural_vert = np.max(scores[search_min:search_max, pa_idx])
                    best_horiz = np.max(scores[search_min:search_max, horiz_indices]) if horiz_indices else -99999.0
                    
                    is_corner = (best_horiz > -10.0) and (best_horiz >= best_natural_vert - 5.0)

                    if not is_corner and best_natural_vert > -30.0:
                        scores[ix, pa_idx] += 150.0 

        valid_profile = np.max(scores, axis=1) > 0.0
        
        while int(round(cur_x)) <= last_ink_x and int(round(cur_x)) < w:
            sx = int(round(cur_x))
            future_valids = np.where(valid_profile[sx:w])[0]
            
            if future_valids.size == 0:
                gap_req = last_ink_x - cur_x
                if gap_req >= self._dot_width:
                    gs, gs_w = self._get_gap_string(gap_req, is_l, last_char, exact_match=False)
                    if gs:
                        line += gs
                        cx_temp = cur_x
                        for gc in gs:
                            gcw = float(self._fw_width) if gc == ' ' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                            ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                            placements.append((gc, cx_temp, gcw, ir))
                            cx_temp += gcw
                        cur_x += gs_w  
                break

            valid_x = sx + future_valids[0]
            
            window_len = min(32, w - valid_x)
            dp = np.full(window_len + 1, -99999.0)
            dp[0] = 0.0
            backtrack = {}
            
            for i in range(window_len):
                if i > 0 and dp[i-1] > dp[i]:
                    dp[i] = dp[i-1]
                    backtrack[i] = (i-1, -1) 
                    
                if dp[i] <= -90000.0: continue
                    
                valid_c_indices = np.where(scores[valid_x + i, :] > 0.0)[0]
                for c_idx in valid_c_indices:
                    cw = char_data_list[c_idx]['width']
                    if i + cw <= window_len:
                        new_score = dp[i] + scores[valid_x + i, c_idx]
                        if new_score > dp[i + cw]:
                            dp[i + cw] = new_score
                            backtrack[i + cw] = (i, c_idx)
                            
            if window_len > 0 and dp[window_len-1] > dp[window_len]:
                dp[window_len] = dp[window_len-1]
                backtrack[window_len] = (window_len-1, -1)

            best_end = np.argmax(dp[1:]) + 1
            if dp[best_end] > -90000.0:
                curr = best_end
                path = []
                while curr > 0:
                    if curr not in backtrack: break
                    prev, c_idx = backtrack[curr]
                    if c_idx != -1: path.append((prev, c_idx))
                    curr = prev
                
                if not path:
                    gap_req = float(valid_x + 1) - cur_x
                    if gap_req < self._dot_width: gap_req = float(self._dot_width)
                    gs, gs_w = self._get_gap_string(gap_req, is_l, last_char, exact_match=False)
                    if gs:
                        line += gs
                        cx_temp = cur_x
                        for gc in gs:
                            gcw = float(self._fw_width) if gc == ' ' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                            ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                            placements.append((gc, cx_temp, gcw, ir))
                            cx_temp += gcw
                        cur_x += gs_w 
                        is_l = False
                        last_char = gs[-1]
                    else:
                        break
                    continue
                    
                first_char_pos, best_c_idx = path[-1]
                char_info = char_data_list[best_c_idx]
                cw = char_info['width']
                c = char_info['char']
                
                best_sx = valid_x + first_char_pos
                target_x = float(best_sx)
                is_anchor_snap = False
                
                if c in VERT_CHARS:
                    for pa_x, pa_char, pa_w in prev_anchors:
                        curr_center = target_x + cw / 2.0
                        prev_center = pa_x + pa_w / 2.0
                        if abs(curr_center - prev_center) <= 2.0:
                            target_x = pa_x + (pa_w - cw) / 2.0
                            is_anchor_snap = True
                            break

                gap_req = target_x - cur_x
                if gap_req < 0:
                    target_x = cur_x
                    gap_req = 0.0

                if gap_req < self._dot_width * 0.8:
                    ir = np.any(is_roi_mask[int(cur_x):min(w, int(cur_x+cw))])
                    placements.append((c, cur_x, cw, ir))
                    line += c
                    cur_x += float(cw) + spacing
                    is_l = False
                    last_char = c
                else:
                    gs, gs_w = self._get_gap_string(gap_req, is_l, last_char, exact_match=is_anchor_snap)
                    if gs:
                        line += gs
                        cx_temp = cur_x
                        for gc in gs:
                            gcw = float(self._fw_width) if gc == ' ' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                            ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                            placements.append((gc, cx_temp, gcw, ir))
                            cx_temp += gcw
                            
                        cur_x += gs_w 
                        is_l = False
                        last_char = gs[-1]

                        ir = np.any(is_roi_mask[int(cur_x):min(w, int(cur_x+cw))])
                        placements.append((c, cur_x, cw, ir))
                        line += c
                        cur_x += float(cw) + spacing
                        last_char = c
                    else:
                        ir = np.any(is_roi_mask[int(cur_x):min(w, int(cur_x+cw))])
                        placements.append((c, cur_x, cw, ir))
                        line += c
                        cur_x += float(cw) + spacing
                        is_l = False
                        last_char = c
            else:
                gap_req = float(valid_x + 1) - cur_x
                if gap_req < self._dot_width: gap_req = float(self._dot_width)
                gs, gs_w = self._get_gap_string(gap_req, is_l, last_char, exact_match=False)
                if gs:
                    line += gs
                    cx_temp = cur_x
                    for gc in gs:
                        gcw = float(self._fw_width) if gc == ' ' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                        ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                        placements.append((gc, cx_temp, gcw, ir))
                        cx_temp += gcw
                    cur_x += gs_w
                    is_l = False
                    last_char = gs[-1]
                else:
                    break

        return line, placements

    def _solve_stripe_score_priority(self, score_matrix, char_data_list, w, spacing, last_ink_x, is_roi_mask, prev_anchors):
        placements = []
        scores = score_matrix.copy()
        if last_ink_x < w: scores[last_ink_x:, :] = -99999.0
        
        horiz_indices = [i for i, d in enumerate(char_data_list) if d['char'] in HORIZ_CHARS]

        for pa_x, pa_char, pa_w in prev_anchors:
            ix = int(round(pa_x))
            if 0 <= ix < w:
                search_min = max(0, ix - 1)
                search_max = min(w, ix + 2)
                pa_idx = next((i for i, d in enumerate(char_data_list) if d['char'] == pa_char), -1)
                
                # [Condition 1 적용] 이전 줄에서 하단경계 문자(＿, _)가 찍힌 곳과 겹치면 상단 문자(￣, T, ^) 억제
                if pa_char in ['＿', '__']:
                    top_chars = ['￣', 'T', '^', '⌒']
                    top_indices = [i for i, d in enumerate(char_data_list) if d['char'] in top_chars]
                    if top_indices:
                        scores[search_min:search_max, top_indices] -= 9999.0

                if pa_idx != -1:
                    best_natural_vert = np.max(scores[search_min:search_max, pa_idx])
                    best_horiz = np.max(scores[search_min:search_max, horiz_indices]) if horiz_indices else -99999.0
                    
                    is_corner = (best_horiz > -10.0) and (best_horiz >= best_natural_vert - 5.0)

                    if not is_corner and best_natural_vert > -30.0:
                        scores[ix, pa_idx] += 150.0
            
        l_i = [i for i, d in enumerate(char_data_list) if (d['flags'] & 2) > 0]
        c_i = [i for i, d in enumerate(char_data_list) if (d['flags'] & 4) > 0]
        r_i = [i for i, d in enumerate(char_data_list) if (d['flags'] & 8) > 0]
        
        max_it = w * 2; it = 0
        while it < max_it:
            it += 1
            bx, bc = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[bx, bc] <= 0.0: break
                
            char_info = char_data_list[bc]
            cw, c, cf = char_info['width'], char_info['char'], char_info['flags']
            ir = np.any(is_roi_mask[bx:min(w, bx+cw)]) and (cf & 1) > 0
            
            target_x = float(bx)
            is_anchor_snap = False
            if c in VERT_CHARS:
                for pa_x, pa_char, pa_w in prev_anchors:
                    curr_center = target_x + cw / 2.0
                    prev_center = pa_x + pa_w / 2.0
                    if abs(curr_center - prev_center) <= 2.0:
                        target_x = pa_x + (pa_w - cw) / 2.0
                        is_anchor_snap = True
                        break
                        
            placements.append((c, target_x, cw, ir, is_anchor_snap))
            
            s_margin = 2 if (ir and (cf & 4) > 0) else 0
            eb = min(w, int(bx + cw + spacing + s_margin))
            sb = max(0, int(bx - spacing - s_margin))
            
            for i, d in enumerate(char_data_list):
                scores[max(0, sb - d['width'] + 1):eb, i] = -99999.0
                
            if ir:
                bs, be = bx, min(w-1, bx+cw-1)
                while bs > 0 and is_roi_mask[bs-1]: bs -= 1
                while be < w-1 and is_roi_mask[be+1]: be += 1
                if c in DENSE_KANJI: scores[:, bc] = -100.0
                if cw > 6: scores[bs:be+1, bc] = -9999.0
                
                bw = 10.0
                if (cf & 2) > 0:
                    target_range = scores[eb:be+1, c_i + r_i]
                    if target_range.size > 0: scores[eb:be+1, c_i + r_i] += bw
                elif (cf & 8) > 0:
                    target_range = scores[bs:sb, l_i + c_i]
                    if target_range.size > 0: scores[bs:sb, l_i + c_i] += bw
                elif (cf & 4) > 0:
                    l_range = scores[bs:sb, l_i]; r_range = scores[eb:be+1, r_i]
                    if l_range.size > 0: scores[bs:sb, l_i] += bw
                    if r_range.size > 0: scores[eb:be+1, r_i] += bw

        placements.sort(key=lambda x: x[1])
        line = ""; cx = 0.0; isl = True; ap = []; last_char = ''
        for c, x, cw, ir, is_anchor in placements:
            if x < cx:
                x = cx
            gp = x - cx
            if gp >= self._dot_width * 0.8:
                gs, gs_w = self._get_gap_string(gp, isl, last_char, exact_match=is_anchor)
                if gs:
                    line += gs
                    cx_temp = cx
                    for gc in gs:
                        gcw = float(self._fw_width) if gc == ' ' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                        gc_ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                        ap.append((gc, cx_temp, gcw, gc_ir))
                        cx_temp += gcw
                    cx += gs_w 
                    isl = False
                    last_char = gs[-1]
            
            ap.append((c, cx, cw, ir))
            line += c
            cx += float(cw) + spacing
            isl = False
            last_char = c
            
        if cx < last_ink_x - self._dot_width:
            gs, gs_w = self._get_gap_string(last_ink_x - cx, isl, last_char, exact_match=False)
            if gs:
                line += gs
                cx_temp = cx
                for gc in gs:
                    gcw = float(self._fw_width) if gc == '\u3000' else (float(self._hw_width) if gc in [' ', '_'] else float(self._dot_width))
                    gc_ir = np.any(is_roi_mask[int(cx_temp):min(w, int(cx_temp+gcw))])
                    ap.append((gc, cx_temp, gcw, gc_ir))
                    cx_temp += gcw
        return line, ap

    def solve_stripe_hybrid(self, row_img_bin, row_tone, row_cos, row_sin, row_roi, row_roi_weights, row_cmplx, char_data_list, spacing, y_t, params, prev_anchors):
        bg_mode = params['bg_mode']
        bg_weight = params['tone_weight']
        
        H, W = row_img_bin.shape
        h = ROW_HEIGHT
        scores = np.full((W, len(char_data_list)), -99999.0, dtype=np.float32)
        
        ink_p = np.where(row_img_bin > 0)[1]
        force_full_width = (bg_mode.startswith("1") or bg_mode.startswith("2")) and bg_weight > 0
        last_x = W if force_full_width else (int(ink_p.max() + 5) if ink_p.size > 0 else 0)
        
        tone_map = (255.0 - row_tone.astype(np.float32)) / 255.0
        cy_s, cy_e = max(0, y_t), min(H, y_t + h)
        stripe_roi = np.bitwise_or.reduce(row_roi[cy_s:cy_e, :], axis=0)
        is_roi_mask = (stripe_roi & 1) > 0 
        
        if cv2.countNonZero(row_img_bin) == 0 and not np.any(is_roi_mask) and not force_full_width:
            return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask, prev_anchors)

        custom_chars = set(list(params.get('custom_chars', '')))
        custom_pen = params.get('custom_penalty', 0.0)

        with torch.no_grad():
            t_str = torch.from_numpy(row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_cos = torch.from_numpy(row_cos * row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_sin = torch.from_numpy(row_sin * row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_tne = torch.from_numpy(tone_map).unsqueeze(0).unsqueeze(0).to(self._device)
            t_roi = torch.from_numpy(row_roi).unsqueeze(0).unsqueeze(0).to(torch.int32).to(self._device)
            t_wgt = torch.from_numpy(row_roi_weights).unsqueeze(0).unsqueeze(0).to(self._device)
            # [Condition 4 적용] 복잡도(Complexity) 맵 입력
            t_cmpx = torch.from_numpy(row_cmplx).unsqueeze(0).unsqueeze(0).to(self._device)
            
            Y_out = H - h + 1
            if Y_out <= 0: return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask, prev_anchors)
            y_pen = (torch.abs(torch.arange(Y_out, device=self._device).view(Y_out, 1) - y_t) * params['y_shift_penalty']).unsqueeze(0)

            for cw, g in self._char_groups_cache.items():
                N = len(g['indices'])
                ones_k = torch.ones((1, 1, h, cw), dtype=torch.float32, device=self._device)
                
                ov2 = torch.nn.functional.conv2d(t_str, g['masks']).squeeze(0)
                # [Condition 3 적용] Dilation된 마스크를 활용하여 missing 패널티 공간 허용치 마련
                ov2_dilated = torch.nn.functional.conv2d(t_str, g['masks_dilated']).squeeze(0)
                
                m_cos = torch.nn.functional.conv2d(t_cos, g['cos_stricts']).squeeze(0)
                m_sin = torch.nn.functional.conv2d(t_sin, g['sin_stricts']).squeeze(0)
                
                # [Condition 3 적용] 위상(Phase) 매칭 패널티 완화 - 약 20%의 허용치를 주어 각도가 살짝 어긋나도 점수를 유지하도록 (곡선 vs 직선 구별 보존)
                pha2 = (ov2 - (m_cos + m_sin)) * 0.5
                pha2 = torch.clamp(pha2 - (ov2 * 0.20), min=0.0)
                
                t_ink2 = torch.nn.functional.conv2d(t_str, ones_k).squeeze(0)
                
                b_list = [(torch.nn.functional.conv2d((t_roi & (1<<k)).float(), ones_k).squeeze(0) > 0) for k in range(6)]
                b1, b2, b4, b8, b16, b32 = b_list[0], b_list[1], b_list[2], b_list[3], b_list[4], b_list[5]
                
                blob_centrality = torch.nn.functional.conv2d(t_wgt, ones_k).squeeze(0) / (h * cw)
                
                # [Condition 4 적용] 주변 선 픽셀의 복잡도를 계산
                local_cmpx = torch.nn.functional.conv2d(t_cmpx, ones_k).squeeze(0) / (h * cw)
                
                exc = torch.relu(g['inks'].view(N, 1, 1) - ov2)
                # [Condition 3 적용] 공간적 유연성을 줘서 missing penalty 대폭 경감
                mis = torch.relu(t_ink2 - ov2_dilated) 
                
                if params['use_roi']:
                    lerp_w = lambda r, b: b + (r - b) * blob_centrality
                    cur_w_den = lerp_w(params['roi_den_w'], params['density_w'])
                    cur_w_mis = lerp_w(params['roi_mis_w'], params['missing_w'])
                    cur_w_pha = lerp_w(params['roi_pha_w'], params['phase_w'])
                    cur_w_frq = lerp_w(params['roi_frq_w'], params['freq_w'])
                else:
                    cur_w_den, cur_w_mis, cur_w_pha, cur_w_frq = params['density_w'], params['missing_w'], params['phase_w'], params['freq_w']

                # [Condition 4 적용] 복잡도 맵(local_cmpx)에 따른 동적 가중치 할당
                # 외곽선/윤곽(cmpx 낮음): Density 패널티 적게, Missing 패널티 높게 (선을 잘 따라가게)
                # 내부 복잡구역(cmpx 높음): Density 패널티 높게, Missing 패널티 적게 (선을 삐져나오지 않게)
                dyn_den = cur_w_den * (1.0 + local_cmpx * 2.5)  # 내부에선 3.5배까지 패널티 증폭
                dyn_mis = cur_w_mis * (1.0 - local_cmpx * 0.5) + (1.0 - local_cmpx) * 0.5 

                calc = ov2 - pha2*cur_w_pha - exc*dyn_den - mis*dyn_mis + g['freqs'].view(N,1,1)*cur_w_frq - y_pen
                
                is_dot = (g['flags'] & 64) > 0
                dot_penalty = params.get('dot_penalty', 2.0)
                calc = torch.where(is_dot.view(N, 1, 1).bool(), calc - dot_penalty, calc)
                
                if custom_chars and custom_pen > 0:
                    is_custom = torch.tensor([(char_data_list[idx]['char'] in custom_chars) for idx in g['indices']], dtype=torch.bool, device=self._device)
                    calc = torch.where(is_custom.view(N, 1, 1), calc - custom_pen, calc)

                c_req = g['flags'] & ~1
                spatial_match = (~((c_req & 2).bool().view(N,1,1)) | b2) & (~((c_req & 4).bool().view(N,1,1)) | b4) & (~((c_req & 8).bool().view(N,1,1)) | b8) & (~((c_req & 16).bool().view(N,1,1)) | b16) & (~((c_req & 32).bool().view(N,1,1)) | b32)
                
                is_eye = (g['flags'] & 1) > 0
                val_s = b1 & spatial_match & (ov2 > 0)
                
                eye_weight = params.get('eye_char_w', 100.0)
                calc = torch.where((is_eye.view(N, 1, 1).bool() & val_s.bool()).bool(), calc + eye_weight, calc)
                calc = torch.where((is_eye.view(N, 1, 1).bool() & ~val_s.bool() & b1.bool()).bool(), calc - 99999.0, calc)

                if params['use_roi']:
                    calc = torch.where(b1.bool() & (cw >= 8), calc + params['roi_weight'], calc)

                if bg_mode.startswith("1") and bg_weight > 0:
                    patch_avg_tone = torch.nn.functional.conv2d(t_tne, ones_k).squeeze(0) / (h * cw)
                    char_avg_tone = g['inks'].view(N, 1, 1) / (h * cw)
                    tone_diff = torch.abs(patch_avg_tone - char_avg_tone)
                    
                    bg_calc = (2.5 * bg_weight) - (tone_diff * 8.0 * bg_weight) + (g['freqs'].view(N,1,1) * 0.5)
                    is_valid_bg = (~b1.bool()) & (patch_avg_tone > 0.02) & (ov2 == 0)
                    
                    calc = torch.where(is_valid_bg, bg_calc, calc)
                    calc = torch.where((~b1.bool()) & (ov2 > 0), calc - (tone_diff * 8.0 * bg_weight), calc)
                    
                    strict_mask = ((ov2 > 0) | (is_eye.view(N,1,1).bool() & val_s.bool()) | is_valid_bg).bool()
                    calc = torch.where(strict_mask, calc, torch.full_like(calc, -99999.0))
                else:
                    calc = torch.where(((ov2 > 0) | (is_eye.view(N,1,1).bool() & val_s.bool())).bool(), calc, torch.full_like(calc, -99999.0))
                
                b_np = torch.max(calc, dim=1)[0].cpu().numpy()
                for i, idx in enumerate(g['indices']):
                    vx = np.where(b_np[i] > -99990.0)[0]
                    if vx.size > 0: scores[vx, idx] = b_np[i][vx]
                    
        if params['p_method'] == "Score-Priority":
            return self._solve_stripe_score_priority(scores, char_data_list, W, spacing, last_x, is_roi_mask, prev_anchors)
        return self._solve_stripe_sequential(scores, char_data_list, W, spacing, last_x, is_roi_mask, prev_anchors)

    def generate(self, ori_rgb, thinned_bin, mask_bin, params, progress_callback, log_callback, cancel_check=None):
        log_callback("진행 1/5: 리소스 로드 중...")
        res = self.load_resources(params['char_csv'], params['font_path'], params['char_tone'])
        if not res: return None, None, None
        char_data_list, font = res
        
        if cancel_check and cancel_check(): return None, None, None
        
        target_h = params['text_lines'] * ROW_HEIGHT
        target_w = int(target_h * (thinned_bin.shape[1] / thinned_bin.shape[0]))
        
        img_bin = cv2.resize(thinned_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        y_shift = params['global_y_shift']
        if y_shift != 0:
            s_bin = np.zeros_like(img_bin)
            if y_shift > 0: s_bin[y_shift:, :] = img_bin[:-y_shift, :]
            else: s_bin[:y_shift, :] = img_bin[-y_shift:, :]
            img_bin = s_bin
        img_bin_f32 = (img_bin > 0).astype(np.float32)
        
        # [Condition 4 적용] Density 조절을 위한 전역 선 복잡도(Complexity) 맵 생성
        img_cmplx = cv2.GaussianBlur(img_bin_f32, (31, 31), 10.0)
        img_cmplx = np.clip(img_cmplx / (img_cmplx.max() + 1e-5), 0, 1).astype(np.float32)

        gray = cv2.cvtColor(ori_rgb, cv2.COLOR_RGB2GRAY)
        img_tone_raw = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_tone_f = (img_tone_raw.astype(np.float32) - 127.5) * params['contrast'] + 127.5 + (params['brightness'] * 255.0)
        img_tone = np.clip(img_tone_f, 0, 255).astype(np.uint8)
        
        if y_shift != 0:
            s_tone = np.full_like(img_tone, 255)
            if y_shift > 0: s_tone[y_shift:, :] = img_tone[:-y_shift, :]
            else: s_tone[:y_shift, :] = img_tone[-y_shift:, :]
            img_tone = s_tone
            
        img_cos, img_sin, _ = calculate_orientation_map(img_bin)
        roi_map = np.zeros((target_h, target_w), dtype=np.int32)
        roi_weight_map = np.zeros((target_h, target_w), dtype=np.float32)
        
        if params['use_roi'] and mask_bin is not None:
            m_np = cv2.resize(mask_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if y_shift != 0:
                s_mask = np.zeros_like(m_np)
                if y_shift > 0: s_mask[y_shift:, :] = m_np[:-y_shift, :]
                else: s_mask[:y_shift, :] = m_np[-y_shift:, :]
                m_np = s_mask
            _, mt, st, _ = cv2.connectedComponentsWithStats(m_np, connectivity=8)
            for i in range(1, len(st)):
                x, y, w, h, _ = st[i]
                if w > 0 and h > 0: 
                    blob_mask = (mt[y:y+h, x:x+w] == i)
                    roi_map[y:y+h, x:x+w][blob_mask] |= 1
                    dist = cv2.distanceTransform(blob_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    if dist.max() > 0:
                        blob_weight = np.zeros((h, w), dtype=np.float32)
                        blob_weight[blob_mask] = dist[blob_mask] / dist.max()
                        sigma = max(2.0, min(h, w) * 0.15)
                        ksize = int(sigma * 6) | 1
                        blob_weight = cv2.GaussianBlur(blob_weight, (ksize, ksize), sigma)
                        if blob_weight.max() > 0:
                            blob_weight = blob_weight / blob_weight.max()
                        roi_weight_map[y:y+h, x:x+w] = np.maximum(
                            roi_weight_map[y:y+h, x:x+w], blob_weight)
                    hh, hw = h//2, w//2
                    roi_map[y:y+hh, x:x+w][blob_mask[0:hh, :]] |= 16
                    roi_map[y+hh:y+h, x:x+w][blob_mask[hh:h, :]] |= 32
                    roi_map[y:y+h, x:x+hw][blob_mask[:, 0:hw]] |= 2
                    roi_map[y:y+h, x+hw:x+w][blob_mask[:, hw:w]] |= 8
                    
        tasks = []
        m_y_base = 0 if params['bg_mode'].startswith("1") else params['y_tolerance']
        
        for r in range(params['text_lines']):
            # [Condition 1] 상하 boundary 겹침 방지 (가운데 부분 위주로 오차 허용)
            dist_to_edge = min(r, params['text_lines'] - 1 - r)
            if dist_to_edge <= 1:
                row_m_y = min(m_y_base, 1) 
            elif dist_to_edge <= 3:
                row_m_y = min(m_y_base, 2) 
            else:
                row_m_y = m_y_base         

            ys = r * ROW_HEIGHT; ye = min(ys + ROW_HEIGHT, target_h)
            sys, sye = max(0, ys-row_m_y), min(target_h, ye+row_m_y)
            pt, pb = max(0, -(ys-row_m_y)), max(0, (ys+ROW_HEIGHT+row_m_y)-target_h)
            tasks.append([
                np.pad(img_bin_f32[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_tone[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_cos[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_sin[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_weight_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_cmplx[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                row_m_y 
            ])
            
        res_l = [""] * len(tasks)
        all_p = [[] for _ in range(len(tasks))]
        replaced_boxes = [[] for _ in range(len(tasks))]
        
        log_callback(f"진행 2/5: 글자 배치 진행 중... (총 {len(tasks)}줄)")
        
        prev_anchors = []
        for idx, t in enumerate(tasks):
            if cancel_check and cancel_check():
                log_callback("❌ 작업이 취소되었습니다.")
                return None, None, None
                
            res_l[idx], all_p[idx] = self.solve_stripe_hybrid(t[0], t[1], t[2], t[3], t[4], t[5], t[6], char_data_list, 0.0, t[7], params, prev_anchors)
            
            current_anchors = []
            for (char, x, cw, ir) in all_p[idx]:
                if char in VERT_CHARS or char in ['＿', '_']:
                    current_anchors.append((float(x), char, float(cw)))
            prev_anchors = current_anchors
            
            progress_callback(int((idx / len(tasks)) * 50))
            if idx % 10 == 0: torch.cuda.empty_cache()

        if params['bg_mode'].startswith("2") and params['tone_weight'] > 0 and self._tone_chars_cache:
            log_callback("진행 3/5: 빈 공간 톤 채우기...")
            tone_chars = self._tone_chars_cache
            REPLACEABLE_CHARS = set([' ', ' ', '.', ',', "'", '．', '，', '_']) 
            
            def get_best_tone_string_dynamic(patch, exact_width, bg_weight):
                if exact_width <= 0: return None
                dp = {0: (0.0, 0, "")}
                
                for w in range(1, exact_width + 1):
                    best_cost = float('inf'); best_prev = 0; best_char = ""
                    for tc_tone, tc_char, tc_width in tone_chars:
                        if tc_width <= 0 or w < tc_width: continue
                        if (w - tc_width) in dp:
                            p_start = w - tc_width
                            p_end = min(w, patch.shape[1])
                            
                            # 패치가 잘린 영역 밖은 흰색(Tone 0.0)으로 처리하여 너비 오차를 방지
                            if p_start < patch.shape[1]:
                                char_patch = patch[:, p_start : p_end]
                                local_tone = (1.0 - (np.mean(char_patch) / 255.0)) * bg_weight if char_patch.size > 0 else 0.0
                            else:
                                local_tone = 0.0
                                
                            cost = dp[w - tc_width][0] + abs(tc_tone - local_tone)
                            if cost < best_cost: 
                                best_cost = cost; best_prev = w - tc_width; best_char = tc_char
                                
                    if best_cost < float('inf'): dp[w] = (best_cost, best_prev, best_char)
                    
                if exact_width in dp:
                    res_str = ""; curr = exact_width
                    while curr > 0:
                        _, prev, ch = dp[curr]
                        res_str = ch + res_str; curr = prev
                    return res_str
                return None 

            for idx, t in enumerate(tasks):
                if cancel_check and cancel_check():
                    return None, None, None
                current_m_y = t[7]
                row_tone = t[1][current_m_y:current_m_y+ROW_HEIGHT, :]
                new_line = ""; chunk_chars = ""; chunk_w = 0.0; chunk_start_x = 0.0
                
                def flush_chunk():
                    nonlocal new_line, chunk_chars, chunk_w, chunk_start_x
                    if chunk_w > 0:
                        ix, icw = int(round(chunk_start_x)), int(round(chunk_w))
                        patch = row_tone[:, ix:ix+icw]
                        if patch.size > 0 and (1.0 - (np.min(patch) / 255.0)) * params['tone_weight'] > 0.05:
                            filled = get_best_tone_string_dynamic(patch, icw, params['tone_weight'])
                            if filled: 
                                new_line += filled
                                replaced_boxes[idx].append((ix, icw))
                            else: new_line += chunk_chars
                        else: new_line += chunk_chars
                        chunk_chars = ""; chunk_w = 0.0

                for (char, x, cw, is_roi) in all_p[idx]:
                    if not is_roi and char in REPLACEABLE_CHARS:
                        if chunk_w == 0: chunk_start_x = x
                        chunk_chars += char; chunk_w += cw
                    else: flush_chunk(); new_line += char
                flush_chunk(); res_l[idx] = new_line
                progress_callback(50 + int((idx / len(tasks)) * 30))

        if cancel_check and cancel_check(): return None, None, None

        log_callback("진행 4/5: 좌우 공백 교정 및 그리드 생성 중...")
        for i in range(len(res_l)):
            if res_l[i].startswith(' '):
                res_l[i] = '_' + res_l[i][1:]
            res_l[i] = re.sub(r'[  \.,\'．，_]+$', '', res_l[i])
        
        grid_vis = cv2.cvtColor(255 - img_bin, cv2.COLOR_GRAY2RGB)
        overlay = grid_vis.copy()
        for r, boxes in enumerate(replaced_boxes):
            y = r * ROW_HEIGHT
            for (x, w) in boxes:
                cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + ROW_HEIGHT)), (255, 210, 150), -1)
        cv2.addWeighted(overlay, 0.5, grid_vis, 0.5, 0, grid_vis)
        
        for r, pl in enumerate(all_p):
            y = r * ROW_HEIGHT
            cv2.line(grid_vis, (0, int(y)), (target_w, int(y)), (0, 0, 255), 1)
            for (_, x, cw, ir) in pl:
                color = (0, 255, 0) if ir else (255, 0, 0)
                thickness = 2 if ir else 1
                cv2.rectangle(grid_vis, (int(x), int(y)), (int(x+cw), int(y+ROW_HEIGHT)), color, thickness)

        if cancel_check and cancel_check(): return None, None, None

        log_callback("진행 5/5: 최종 아스키 이미지 렌더링 중...")
        aa_text = "\n".join(res_l)
        
        dummy_draw = ImageDraw.Draw(Image.new("L", (1,1)))
        max_w = max([sum([get_char_width(c, font, dummy_draw) for c in l]) for l in res_l] + [64.0])
        canvas = Image.new("RGB", (int(max_w), target_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(res_l):
            x = 0.0
            y = i * ROW_HEIGHT
            for char in line:
                draw.text((int(round(x)), y), char, font=font, fill=(0,0,0))
                x += max(1.0, get_char_width(char, font, dummy_draw))
        aa_img_arr = np.array(canvas)

        progress_callback(100)
        return aa_text, grid_vis, aa_img_arr

# ==========================================
# 3. Custom UI Widgets 
# ==========================================
class SliderSpinBox(QWidget):
    def __init__(self, label, min_val, max_val, step, default_val, is_float=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        
        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(85)
        self.lbl.setStyleSheet("font-size: 11px;")
        
        self.is_float = is_float
        self.scale = 100 if is_float else 1
        self.step = step
        
        if is_float:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(2)
            self.spin.setSingleStep(float(step))
            self.spin.setRange(float(min_val), float(max_val))
            self.spin.setValue(float(default_val))
        else:
            self.spin = QSpinBox()
            self.spin.setSingleStep(int(step))
            self.spin.setRange(int(min_val), int(max_val))
            self.spin.setValue(int(default_val))
            
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setFixedWidth(40)
        self.spin.setAlignment(Qt.AlignCenter)
        
        self.btn_minus = QPushButton("◀")
        self.btn_minus.setFixedWidth(16)
        self.btn_minus.clicked.connect(self.decrement)
        
        self.btn_plus = QPushButton("▶")
        self.btn_plus.setFixedWidth(16)
        self.btn_plus.clicked.connect(self.increment)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.scale), int(max_val * self.scale))
        self.slider.setValue(int(default_val * self.scale))
        
        self.spin.valueChanged.connect(self.update_slider)
        self.slider.valueChanged.connect(self.update_spin)
        
        layout.addWidget(self.lbl)
        layout.addWidget(self.btn_minus)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn_plus)
        layout.addWidget(self.spin)

    def decrement(self):
        self.spin.setValue(self.spin.value() - self.step)

    def increment(self):
        self.spin.setValue(self.spin.value() + self.step)
        
    def update_slider(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self.scale))
        self.slider.blockSignals(False)
        
    def update_spin(self, val):
        self.spin.blockSignals(True)
        if self.is_float: self.spin.setValue(val / self.scale)
        else: self.spin.setValue(int(val / self.scale))
        self.spin.blockSignals(False)
        
    def value(self):
        return self.spin.value()
        
    def setEnabled(self, state):
        self.lbl.setEnabled(state)
        self.slider.setEnabled(state)
        self.spin.setEnabled(state)
        self.btn_minus.setEnabled(state)
        self.btn_plus.setEnabled(state)

class AspectRatioLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setMinimumSize(100, 100)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None

    def setPixmap(self, p):
        self._pixmap = p
        self.update_scaled()

    def resizeEvent(self, event):
        self.update_scaled()
        super().resizeEvent(event)

    def update_scaled(self):
        if self._pixmap and not self._pixmap.isNull():
            w, h = self.width(), self.height()
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)

class PaintableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self.setAlignment(Qt.AlignCenter)
        self.original_pixmap = None
        self.mask_img = None
        self._current_rgba = None 
        self.drawing = False
        self.drawing_enabled = True
        self.brush_size = 5
        self.undo_stack = []

    def set_image(self, qpixmap):
        self.original_pixmap = qpixmap
        self.mask_img = np.zeros((qpixmap.height(), qpixmap.width()), dtype=np.uint8)
        self.undo_stack = []
        self.update_display()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def update_display(self):
        if not self.original_pixmap: return
        w, h = self.width(), self.height()
        composite = QPixmap(self.original_pixmap.size())
        composite.fill(Qt.transparent)
        painter = QPainter(composite)
        painter.drawPixmap(0, 0, self.original_pixmap)
        
        if self.mask_img is not None and self.drawing_enabled:
            mask_rgba = np.zeros((self.mask_img.shape[0], self.mask_img.shape[1], 4), dtype=np.uint8)
            mask_rgba[self.mask_img > 0] = [255, 0, 0, 100] 
            
            self._current_rgba = mask_rgba 
            mask_qimg = QImage(self._current_rgba.data, self._current_rgba.shape[1], self._current_rgba.shape[0], self._current_rgba.strides[0], QImage.Format_RGBA8888)
            painter.drawImage(0, 0, mask_qimg)
            
        painter.end()
        scaled = composite.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)

    def mousePressEvent(self, event):
        if not self.drawing_enabled: return
        if event.button() == Qt.LeftButton and self.original_pixmap:
            self.undo_stack.append(self.mask_img.copy())
            if len(self.undo_stack) > 10: self.undo_stack.pop(0)
            self.drawing = True
            self.draw_mask(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing and self.original_pixmap and self.drawing_enabled:
            self.draw_mask(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def draw_mask(self, pos):
        if not self.original_pixmap or not self.drawing_enabled: return
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self.original_pixmap.width(), self.original_pixmap.height()
        
        scale = min(label_w / pix_w, label_h / pix_h)
        disp_w, disp_h = pix_w * scale, pix_h * scale
        x_off = (label_w - disp_w) / 2
        y_off = (label_h - disp_h) / 2
        
        px = (pos.x() - x_off) / scale
        py = (pos.y() - y_off) / scale
        
        if 0 <= px < pix_w and 0 <= py < pix_h:
            adj_brush = max(1, int(self.brush_size / scale))
            cv2.circle(self.mask_img, (int(px), int(py)), adj_brush, 255, -1)
            self.update_display()

    def undo(self):
        if self.undo_stack and self.drawing_enabled:
            self.mask_img = self.undo_stack.pop()
            self.update_display()

    def clear_mask(self):
        if self.mask_img is not None:
            self.undo_stack.append(self.mask_img.copy())
            self.mask_img.fill(0)
            self.update_display()

# ==========================================
# 4. Worker Thread & Main Window
# ==========================================
class WorkerThread(QThread):
    sig_progress = pyqtSignal(int)
    sig_log = pyqtSignal(str)
    sig_line_done = pyqtSignal(object)
    sig_thinned_done = pyqtSignal(object)
    sig_finished = pyqtSignal(str, object, object) 

    def __init__(self, pipeline, img_rgb, mask_bin, params):
        super().__init__()
        self.p = pipeline
        self.img_rgb = img_rgb
        self.mask_bin = mask_bin
        self.params = params
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def check_cancel(self):
        return self._is_cancelled

    def run(self):
        try:
            self.sig_log.emit("▶ [1/3] 선화 추출 시작...")
            binary = self.p.extract_lines(self.img_rgb, self.params['text_lines'], self.params['line_method'], self.params['threshold'], self.params['thickness'], self.params['kmeans_k'], True)
            self.sig_line_done.emit(binary)
            
            if self._is_cancelled: return self.sig_finished.emit("", None, None)
            
            self.sig_log.emit("▶ [2/3] 세선화 (Thinning) 시작...")
            thinned = self.p.process_thinning(binary, self.params['clean'], self.params['thin_method'])
            self.sig_thinned_done.emit(thinned)
            
            if self._is_cancelled: return self.sig_finished.emit("", None, None)
            
            self.sig_log.emit("▶ [3/3] 아스키 아트 생성 중...")
            aa_text, grid_vis, aa_img = self.p.generate(self.img_rgb, thinned, self.mask_bin, self.params, self.sig_progress.emit, self.sig_log.emit, self.check_cancel)
            
            if self._is_cancelled:
                self.sig_finished.emit("", None, None)
            else:
                self.sig_finished.emit(aa_text, grid_vis, aa_img)
        except Exception as e:
            self.sig_log.emit(f"❌ 오류 발생:\n{str(e)}")
            self.sig_finished.emit("", None, None)

class SweepWorkerThread(QThread):
    """파라미터 스윕용 단일 셀 렌더링 워커"""
    sig_done = pyqtSignal(int, int, object)  # (row, col, aa_img_arr or None)
    sig_log  = pyqtSignal(str)

    def __init__(self, pipeline, img_rgb, mask_bin, params, row, col):
        super().__init__()
        self.p        = pipeline
        self.img_rgb  = img_rgb
        self.mask_bin = mask_bin
        self.params   = params
        self.row      = row
        self.col      = col

    def run(self):
        try:
            binary = self.p.extract_lines(
                self.img_rgb, self.params['text_lines'],
                self.params['line_method'], self.params['threshold'],
                self.params['thickness'], self.params['kmeans_k'], True)
            thinned = self.p.process_thinning(
                binary, self.params['clean'], self.params['thin_method'])
            _, _, aa_img = self.p.generate(
                self.img_rgb, thinned, self.mask_bin, self.params,
                lambda v: None, lambda s: self.sig_log.emit(s))
            self.sig_done.emit(self.row, self.col, aa_img)
        except Exception as e:
            self.sig_log.emit(f"❌ 스윕 오류 [{self.row},{self.col}]: {str(e)}")
            self.sig_done.emit(self.row, self.col, None)


# ── 파라미터 1개 설정 위젯 ────────────────────────────────────────────────
class _ParamAxisWidget(QGroupBox):
    """
    콤보 + Min/Max/Step 스핀박스 한 세트.
    'Step' = 실제 수치 간격 (예: 0.5 이면 2.0, 2.5, 3.0, …)
    values() 로 해당 축의 값 리스트를 반환한다.
    """
    SWEEP_PARAMS = [
        # (표시 이름,        키,            전체Min, 전체Max, 기본값, 기본Step)
        # ※ 기본값·범위는 메인 UI의 SliderSpinBox 설정과 일치시킴
        ("Phase W",          "phase_w",      0.0, 10.0,  5.0,  1.0),
        ("Density Pen",      "density_w",    0.0,  1.0,  0.5,  0.1),
        ("Missing Pen",      "missing_w",    0.0,  1.0,  0.1,  0.05),
        ("Freq Bonus",       "freq_w",       0.0, 10.0,  1.0,  1.0),
        ("Dot Pen",          "dot_penalty",  0.0, 20.0,  2.0,  2.0),
        ("Eye Char W",       "eye_char_w",   0.0,500.0,100.0, 50.0),
        ("ROI Phase",        "roi_pha_w",    0.0,  5.0,  0.0,  0.5),
        ("ROI Density",      "roi_den_w",    0.0,  2.0,  0.0,  0.2),
        ("ROI Missing",      "roi_mis_w",    0.0,  2.0,  0.0,  0.2),
        ("Tone Weight",      "tone_weight",  0.0, 10.0,  1.0,  1.0),
    ]

    def __init__(self, title):
        super().__init__(title)
        lay = QGridLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setHorizontalSpacing(6)

        lay.addWidget(QLabel("파라미터:"), 0, 0)
        self.combo = QComboBox()
        for label, *_ in self.SWEEP_PARAMS:
            self.combo.addItem(label)
        lay.addWidget(self.combo, 0, 1, 1, 3)

        lay.addWidget(QLabel("Min:"),  1, 0)
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-9999, 9999); self.spin_min.setDecimals(3)
        self.spin_min.setFixedWidth(75)
        lay.addWidget(self.spin_min, 1, 1)

        lay.addWidget(QLabel("Max:"),  1, 2)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-9999, 9999); self.spin_max.setDecimals(3)
        self.spin_max.setFixedWidth(75)
        lay.addWidget(self.spin_max, 1, 3)

        lay.addWidget(QLabel("Step:"), 2, 0)
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.001, 9999); self.spin_step.setDecimals(3)
        self.spin_step.setFixedWidth(75)
        lay.addWidget(self.spin_step, 2, 1)

        self.lbl_count = QLabel("→ 0 단계")
        self.lbl_count.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(self.lbl_count, 2, 2, 1, 2)

        # 콤보 변경 시 자동 기본값 설정은 *사용자가 직접 바꿀 때만* 동작
        # (set_state 복원 시 덮어쓰지 않도록 _ignore_combo 플래그 사용)
        self._ignore_combo = False
        self.combo.currentIndexChanged.connect(self._on_param_changed)
        self.spin_min.valueChanged.connect(self._update_count)
        self.spin_max.valueChanged.connect(self._update_count)
        self.spin_step.valueChanged.connect(self._update_count)
        self._on_param_changed(0)

    def _on_param_changed(self, idx):
        if self._ignore_combo:
            return
        _, _, mn, mx, default, default_step = self.SWEEP_PARAMS[idx]
        self.spin_step.setValue(default_step)
        half = default_step * 2
        self.spin_min.setValue(max(mn, default - half))
        self.spin_max.setValue(min(mx, default + half))
        self._update_count()

    def _update_count(self):
        n = len(self.values())
        self.lbl_count.setText(f"→ {n} 단계")

    # ── 상태 저장/복원 (창 재오픈 시 값 유지) ──────────────────────────
    def get_state(self) -> dict:
        return {
            'combo_idx': self.combo.currentIndex(),
            'min':       self.spin_min.value(),
            'max':       self.spin_max.value(),
            'step':      self.spin_step.value(),
        }

    def set_state(self, state: dict):
        """저장된 상태를 복원 — 콤보 변경에 의한 자동 덮어쓰기를 억제"""
        self._ignore_combo = True
        try:
            self.combo.setCurrentIndex(state.get('combo_idx', 0))
            self.spin_min.setValue(state.get('min',  self.spin_min.value()))
            self.spin_max.setValue(state.get('max',  self.spin_max.value()))
            self.spin_step.setValue(state.get('step', self.spin_step.value()))
        finally:
            self._ignore_combo = False
        self._update_count()

    # ── 값 목록 계산 ───────────────────────────────────────────────────
    def values(self):
        v_min  = self.spin_min.value()
        v_max  = self.spin_max.value()
        step   = self.spin_step.value()
        if step <= 0 or v_min > v_max:
            return []
        result = []
        v = v_min
        while v <= v_max + step * 1e-6:
            result.append(round(v, 6))
            v += step
        return result

    # ── 선택된 파라미터 키와 레이블 반환 ───────────────────────────────
    def param_info(self):
        idx = self.combo.currentIndex()
        label, key, *_ = self.SWEEP_PARAMS[idx]
        return key, label


class SweepDialog(QDialog):
    """
    2개의 파라미터를 각각 X축(열) / Y축(행) 으로 스윕하여
    결과 렌더링 이미지를 행×열 격자로 비교하는 다이얼로그.

    ● 셀 클릭 → 확대 뷰어 (ImageZoomDialog)
    ● "전체 저장" → 격자 이미지를 파라미터 정보가 포함된 파일명으로 저장
                   + 각 셀 개별 PNG도 동시 저장
    """

    def __init__(self, parent, pipeline, img_rgb, mask_bin, base_params):
        super().__init__(parent)
        self.setWindowTitle("2D Parameter Sweep — 비교 미리보기")
        self.resize(1300, 880)
        self.pipeline    = pipeline
        self.img_rgb     = img_rgb
        self.mask_bin    = mask_bin
        self.base_params = base_params   # ★ 메인 UI에서 넘어온 전체 파라미터 스냅샷
        self._workers    = []
        self._pending    = 0
        self._total      = 0
        # {(row, col): QPixmap}  — 원본(전체 해상도) 픽스맵 보관
        self._result_pixmaps = {}
        # {(row, col): AspectRatioLabel}
        self._result_labels  = {}
        # 현재 스윕 축 정보 (저장 시 파일명 생성용)
        self._key_x = self._lbl_x = ""
        self._key_y = self._lbl_y = ""
        self._vals_x: list = []
        self._vals_y: list = []
        # 마지막 저장 폴더
        self._last_save_dir = (parent._last_save_dir
                               if hasattr(parent, '_last_save_dir') else _SCRIPT_DIR)
        self._build_ui()

        # ── 축 설정 복원 (창을 껐다 다시 열 때 이전 값 유지) ──────────
        # 부모 창에 저장된 상태가 있으면 복원한다
        saved = getattr(parent, '_sweep_axis_state', None)
        if saved:
            self.axis_x.set_state(saved.get('x', {}))
            self.axis_y.set_state(saved.get('y', {}))

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── 상단: 두 축 설정 + 버튼 그룹 ─────────────────────────────
        top = QHBoxLayout()
        self.axis_x = _ParamAxisWidget("X 축 (열 방향)")
        self.axis_y = _ParamAxisWidget("Y 축 (행 방향)")
        top.addWidget(self.axis_x, stretch=1)
        top.addWidget(self.axis_y, stretch=1)

        right_ctrl = QVBoxLayout()
        right_ctrl.setSpacing(6)

        self.btn_run = QPushButton("▶ 스윕 실행")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet(
            "font-weight:bold; background:#9C27B0; color:white; font-size:13px;")
        self.btn_run.clicked.connect(self._run_sweep)
        right_ctrl.addWidget(self.btn_run)

        self.btn_stop_sweep = QPushButton("■ 중단")
        self.btn_stop_sweep.setMinimumHeight(30)
        self.btn_stop_sweep.setStyleSheet(
            "font-weight:bold; background:#F44336; color:white;")
        self.btn_stop_sweep.setEnabled(False)
        self.btn_stop_sweep.clicked.connect(self._stop_sweep)
        right_ctrl.addWidget(self.btn_stop_sweep)

        # ── 전체 저장 버튼 ──────────────────────────────────────────
        self.btn_save_all = QPushButton("💾 전체 저장")
        self.btn_save_all.setMinimumHeight(30)
        self.btn_save_all.setStyleSheet(
            "font-weight:bold; background:#1565C0; color:white;")
        self.btn_save_all.setEnabled(False)
        self.btn_save_all.setToolTip(
            "격자 전체 이미지 + 각 셀 개별 PNG를\n"
            "파라미터 정보가 포함된 파일명으로 저장합니다.")
        self.btn_save_all.clicked.connect(self._save_all)
        right_ctrl.addWidget(self.btn_save_all)

        self.sweep_pbar = QProgressBar()
        self.sweep_pbar.setValue(0)
        right_ctrl.addWidget(self.sweep_pbar)

        self.lbl_progress = QLabel("대기 중")
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        self.lbl_progress.setStyleSheet("font-size:11px; color:#555;")
        right_ctrl.addWidget(self.lbl_progress)
        right_ctrl.addStretch()

        top.addLayout(right_ctrl)
        root.addLayout(top)

        # ── 로그 ──────────────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(52)
        self.log_box.setStyleSheet("font-size:11px;")
        root.addWidget(self.log_box)

        # ── 격자 스크롤 영역 ──────────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(4)
        self.scroll.setWidget(self.grid_container)
        root.addWidget(self.scroll, stretch=1)

        # ── 하단 힌트 ─────────────────────────────────────────────────
        hint = QLabel("💡 셀 이미지를 클릭하면 확대 뷰어가 열립니다.")
        hint.setStyleSheet("color:#777; font-size:11px;")
        root.addWidget(hint)

    # ── 스윕 실행 ────────────────────────────────────────────────────────
    def _run_sweep(self):
        vals_x = self.axis_x.values()
        vals_y = self.axis_y.values()
        key_x, lbl_x = self.axis_x.param_info()
        key_y, lbl_y = self.axis_y.param_info()

        if len(vals_x) < 1 or len(vals_y) < 1:
            QMessageBox.warning(
                self, "오류",
                "각 축에 값이 최소 1개 이상 필요합니다.\nMin ≤ Max, Step > 0 을 확인하세요.")
            return

        total = len(vals_x) * len(vals_y)
        if total > 25:
            ans = QMessageBox.question(
                self, "주의",
                f"총 {total}개의 렌더링이 필요합니다.\n"
                f"시간이 오래 걸릴 수 있습니다. 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        # 이전 워커 정리
        self._stop_sweep()
        self._workers.clear()

        # 상태 저장
        self._key_x, self._lbl_x = key_x, lbl_x
        self._key_y, self._lbl_y = key_y, lbl_y
        self._vals_x = vals_x
        self._vals_y = vals_y

        # 격자/결과 초기화
        self._result_labels.clear()
        self._result_pixmaps.clear()
        self.btn_save_all.setEnabled(False)
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # ── 교차점 코너 ──────────────────────────────────────────────
        corner = QLabel(f"{lbl_y} ↓ \\ {lbl_x} →")
        corner.setAlignment(Qt.AlignCenter)
        corner.setStyleSheet(
            "font-size:10px; color:#444; background:#eee;"
            "border:1px solid #ccc; padding:2px;")
        self.grid_layout.addWidget(corner, 0, 0)

        # ── 열 헤더 (X 값) ────────────────────────────────────────────
        for c, vx in enumerate(vals_x):
            h = QLabel(f"{lbl_x}\n{vx:.3g}")
            h.setAlignment(Qt.AlignCenter)
            h.setStyleSheet(
                "font-weight:bold; font-size:10px; background:#D1C4E9;"
                "border:1px solid #9C27B0; padding:2px;")
            self.grid_layout.addWidget(h, 0, c + 1)

        # ── 행 헤더 + 셀 ─────────────────────────────────────────────
        for r, vy in enumerate(vals_y):
            row_h = QLabel(f"{lbl_y}\n{vy:.3g}")
            row_h.setAlignment(Qt.AlignCenter)
            row_h.setStyleSheet(
                "font-weight:bold; font-size:10px; background:#E8F5E9;"
                "border:1px solid #4CAF50; padding:2px;")
            self.grid_layout.addWidget(row_h, r + 1, 0)

            for c, vx in enumerate(vals_x):
                cell_lbl = _ClickableImageLabel(r, c, "⏳")
                cell_lbl.setMinimumSize(160, 130)
                cell_lbl.setFrameShape(QFrame.Box)
                cell_lbl.setAlignment(Qt.AlignCenter)
                cell_lbl.setStyleSheet("background:#f9f9f9;")
                cell_lbl.sig_clicked.connect(self._on_cell_clicked)
                self.grid_layout.addWidget(cell_lbl, r + 1, c + 1)
                self._result_labels[(r, c)] = cell_lbl

        # ── 진행 초기화 ──────────────────────────────────────────────
        self._total   = total
        self._pending = total
        self.sweep_pbar.setMaximum(total)
        self.sweep_pbar.setValue(0)
        self.lbl_progress.setText(f"0 / {total} 완료")

        # ── 작업 큐 ──────────────────────────────────────────────────
        self._job_queue = []
        for r, vy in enumerate(vals_y):
            for c, vx in enumerate(vals_x):
                # base_params 전체를 복사한 뒤 스윕 두 파라미터만 덮어씀
                # → 나머지 파라미터는 메인 UI 현재값 그대로 유지됨
                params = dict(self.base_params)
                params[key_x] = vx
                params[key_y] = vy
                self._job_queue.append((r, c, params))

        # ── 고정 파라미터 로그 (확인용) ──────────────────────────────
        sweep_keys = {key_x, key_y}
        fixed_summary = ", ".join(
            f"{k}={v!r}" for k, v in self.base_params.items()
            if k not in sweep_keys
        )
        self.log_box.append(
            f"▶ 스윕 시작: X={lbl_x}({len(vals_x)}단계)  "
            f"Y={lbl_y}({len(vals_y)}단계)  총 {total}개")
        self.log_box.append(f"   고정 파라미터: {fixed_summary}")

        self._job_idx  = 0
        self._cancelled = False
        self.btn_run.setEnabled(False)
        self.btn_stop_sweep.setEnabled(True)
        self._launch_next()

    # ── 워커 관리 ─────────────────────────────────────────────────────────
    def _stop_sweep(self):
        self._cancelled = True
        for w in self._workers:
            if w.isRunning():
                w.quit(); w.wait(500)
        self._workers.clear()
        self.btn_run.setEnabled(True)
        self.btn_stop_sweep.setEnabled(False)

    def _launch_next(self):
        if self._cancelled or self._job_idx >= len(self._job_queue):
            return
        r, c, params = self._job_queue[self._job_idx]
        self._job_idx += 1
        worker = SweepWorkerThread(
            self.pipeline, self.img_rgb, self.mask_bin, params, r, c)
        worker.sig_done.connect(self._on_cell_done)
        worker.sig_log.connect(lambda s: self.log_box.append(s))
        self._workers.append(worker)
        worker.start()

    def _on_cell_done(self, row, col, aa_img):
        if self._cancelled:
            return

        lbl = self._result_labels.get((row, col))
        if aa_img is not None:
            h, w, ch = aa_img.shape
            qimg = QImage(aa_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix  = QPixmap.fromImage(qimg)
            self._result_pixmaps[(row, col)] = pix   # 원본 해상도 보관
            if lbl:
                lbl.setPixmap(pix)
                lbl.setToolTip(
                    f"{self._lbl_x}={self._vals_x[col]:.3g}  "
                    f"{self._lbl_y}={self._vals_y[row]:.3g}\n"
                    "클릭하면 확대 뷰어가 열립니다.")
        else:
            if lbl:
                lbl.setText("❌ 실패")

        done = self._total - self._pending + 1
        self.sweep_pbar.setValue(done)
        self.lbl_progress.setText(f"{done} / {self._total} 완료")
        self._pending -= 1

        self._launch_next()

        if self._pending <= 0:
            self.log_box.append(f"✅ 스윕 완료! (총 {self._total}개)")
            self.btn_run.setEnabled(True)
            self.btn_stop_sweep.setEnabled(False)
            self.btn_save_all.setEnabled(bool(self._result_pixmaps))

    # ── 셀 클릭 → 확대 뷰어 ──────────────────────────────────────────────
    def _on_cell_clicked(self, row, col):
        pix = self._result_pixmaps.get((row, col))
        if pix is None or pix.isNull():
            return
        vx = self._vals_x[col] if col < len(self._vals_x) else "?"
        vy = self._vals_y[row] if row < len(self._vals_y) else "?"
        title = (f"{self._lbl_x}={vx:.3g}  |  {self._lbl_y}={vy:.3g}")
        viewer = ImageZoomDialog(pix, title, self)
        viewer.exec_()

    # ── 전체 저장 ─────────────────────────────────────────────────────────
    def _save_all(self):
        if not self._result_pixmaps:
            QMessageBox.information(self, "알림", "저장할 이미지가 없습니다.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "저장할 폴더 선택", self._last_save_dir)
        if not folder:
            return
        self._last_save_dir = folder

        lbl_x_safe = re.sub(r'[^\w]', '_', self._lbl_x)
        lbl_y_safe = re.sub(r'[^\w]', '_', self._lbl_y)

        # ① 각 셀 개별 저장
        saved = 0
        for (r, c), pix in self._result_pixmaps.items():
            vx = self._vals_x[c] if c < len(self._vals_x) else 0
            vy = self._vals_y[r] if r < len(self._vals_y) else 0
            # 소수점을 'p'로 치환해 파일명에 사용
            vx_str = f"{vx:.4g}".replace('.', 'p').replace('-', 'm')
            vy_str = f"{vy:.4g}".replace('.', 'p').replace('-', 'm')
            fname = (f"sweep_{lbl_x_safe}{vx_str}"
                     f"_{lbl_y_safe}{vy_str}.png")
            path = os.path.join(folder, fname)
            pix.save(path)
            saved += 1

        # ② 격자 전체를 하나의 이미지로 합성하여 저장
        grid_pix = self._compose_grid_image()
        if grid_pix:
            n_x = len(self._vals_x); n_y = len(self._vals_y)
            x_range = (f"{self._vals_x[0]:.3g}"
                       f"to{self._vals_x[-1]:.3g}"
                       f"s{self.axis_x.spin_step.value():.3g}"
                       ).replace('.', 'p').replace('-', 'm')
            y_range = (f"{self._vals_y[0]:.3g}"
                       f"to{self._vals_y[-1]:.3g}"
                       f"s{self.axis_y.spin_step.value():.3g}"
                       ).replace('.', 'p').replace('-', 'm')
            grid_fname = (f"sweep_grid"
                          f"_{lbl_x_safe}_{x_range}"
                          f"_{lbl_y_safe}_{y_range}"
                          f"_{n_x}x{n_y}.png")
            grid_path = os.path.join(folder, grid_fname)
            grid_pix.save(grid_path)

        self.log_box.append(
            f"💾 저장 완료: 개별 {saved}개 + 격자 1개 → {folder}")
        QMessageBox.information(
            self, "저장 완료",
            f"개별 이미지 {saved}개와 격자 이미지 1개를\n"
            f"다음 폴더에 저장했습니다:\n{folder}")

    def _compose_grid_image(self):
        """현재 결과 픽스맵들을 하나의 격자 이미지로 합성"""
        if not self._result_pixmaps:
            return None

        n_rows = len(self._vals_y)
        n_cols = len(self._vals_x)

        # 셀 크기: 보관된 픽스맵 중 최대 크기 기준
        cell_w = max((p.width()  for p in self._result_pixmaps.values()), default=200)
        cell_h = max((p.height() for p in self._result_pixmaps.values()), default=160)
        # 크기 제한 (격자가 너무 커지지 않도록)
        cell_w = min(cell_w, 480)
        cell_h = min(cell_h, 380)

        HEADER_H = 36   # 열 헤더 높이
        HEADER_W = 90   # 행 헤더 폭
        PAD      = 3

        total_w = HEADER_W + n_cols * (cell_w + PAD) + PAD
        total_h = HEADER_H + n_rows * (cell_h + PAD) + PAD

        canvas = QPixmap(total_w, total_h)
        canvas.fill(QColor("#e0e0e0"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        font_h = QFont("Arial", 8, QFont.Bold)
        font_v = QFont("Arial", 8)
        painter.setFont(font_h)

        # 교차점 배경
        painter.fillRect(0, 0, HEADER_W, HEADER_H, QColor("#cccccc"))
        painter.setPen(QColor("#333"))
        painter.drawText(QRect(0, 0, HEADER_W, HEADER_H),
                         Qt.AlignCenter,
                         f"{self._lbl_y[:6]}↓\\{self._lbl_x[:6]}→")

        # 열 헤더
        painter.setFont(font_h)
        for c, vx in enumerate(self._vals_x):
            x0 = HEADER_W + c * (cell_w + PAD) + PAD
            painter.fillRect(x0, 0, cell_w, HEADER_H, QColor("#D1C4E9"))
            painter.setPen(QColor("#6A1B9A"))
            painter.drawText(QRect(x0, 0, cell_w, HEADER_H),
                             Qt.AlignCenter,
                             f"{self._lbl_x[:8]}\n{vx:.3g}")
            painter.setPen(QColor("#9C27B0"))
            painter.drawRect(x0, 0, cell_w - 1, HEADER_H - 1)

        # 행 헤더 + 셀
        painter.setFont(font_v)
        for r, vy in enumerate(self._vals_y):
            y0 = HEADER_H + r * (cell_h + PAD) + PAD
            painter.fillRect(0, y0, HEADER_W, cell_h, QColor("#E8F5E9"))
            painter.setPen(QColor("#2E7D32"))
            painter.drawText(QRect(0, y0, HEADER_W, cell_h),
                             Qt.AlignCenter,
                             f"{self._lbl_y[:8]}\n{vy:.3g}")
            painter.setPen(QColor("#4CAF50"))
            painter.drawRect(0, y0, HEADER_W - 1, cell_h - 1)

            for c in range(n_cols):
                x0 = HEADER_W + c * (cell_w + PAD) + PAD
                pix = self._result_pixmaps.get((r, c))
                if pix and not pix.isNull():
                    scaled = pix.scaled(
                        cell_w, cell_h,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    dx = (cell_w - scaled.width())  // 2
                    dy = (cell_h - scaled.height()) // 2
                    painter.fillRect(x0, y0, cell_w, cell_h, QColor("white"))
                    painter.drawPixmap(x0 + dx, y0 + dy, scaled)
                else:
                    painter.fillRect(x0, y0, cell_w, cell_h, QColor("#ffeeee"))
                    painter.setPen(QColor("#cc0000"))
                    painter.drawText(QRect(x0, y0, cell_w, cell_h),
                                     Qt.AlignCenter, "❌")
                painter.setPen(QColor("#aaa"))
                painter.drawRect(x0, y0, cell_w - 1, cell_h - 1)

        painter.end()
        return canvas

    def closeEvent(self, event):
        # 다음 번 열릴 때 복원할 수 있도록 부모 창에 축 설정 저장
        if self.parent() is not None:
            self.parent()._sweep_axis_state = {
                'x': self.axis_x.get_state(),
                'y': self.axis_y.get_state(),
            }
        self._stop_sweep()
        super().closeEvent(event)


# ── 클릭 가능한 셀 레이블 ────────────────────────────────────────────────
class _ClickableImageLabel(AspectRatioLabel):
    """클릭 시 (row, col) 시그널을 방출하는 이미지 레이블"""
    sig_clicked = pyqtSignal(int, int)

    def __init__(self, row, col, text=""):
        super().__init__(text)
        self._row = row
        self._col = col
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.sig_clicked.emit(self._row, self._col)
        super().mousePressEvent(event)


# ── 확대 뷰어 다이얼로그 ─────────────────────────────────────────────────
class ImageZoomDialog(QDialog):
    """
    단일 이미지를 확대해서 보여주는 뷰어.
    마우스 휠로 줌 인/아웃, 드래그로 패닝.
    """

    def __init__(self, pixmap: QPixmap, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"확대 뷰 — {title}")
        self.resize(900, 700)
        self._orig_pix   = pixmap
        self._scale      = 1.0
        self._pan_start  = None
        self._offset     = QPoint(0, 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 타이틀 / 조작 힌트
        hdr = QHBoxLayout()
        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_hint  = QLabel("휠: 줌  |  드래그: 이동  |  더블클릭: 원래 크기")
        lbl_hint.setStyleSheet("color:#888; font-size:11px;")
        lbl_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(lbl_title)
        hdr.addWidget(lbl_hint)
        root.addLayout(hdr)

        # 이미지 영역
        self._view = _ZoomableImageWidget(pixmap)
        root.addWidget(self._view, stretch=1)

        # 저장 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_save = QPushButton("💾 이미지 저장")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_save)
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        self._title  = title
        self._parent = parent

    def _save(self):
        safe_title = re.sub(r'[^\w\-]', '_', self._title)
        default_dir = (self._parent._last_save_dir
                       if hasattr(self._parent, '_last_save_dir') else _SCRIPT_DIR)
        fname, _ = QFileDialog.getSaveFileName(
            self, "이미지 저장",
            os.path.join(default_dir, f"zoom_{safe_title}.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg)")
        if fname:
            self._orig_pix.save(fname)
            if hasattr(self._parent, '_last_save_dir'):
                self._parent._last_save_dir = os.path.dirname(fname)


class _ZoomableImageWidget(QWidget):
    """휠 줌 + 드래그 패닝이 가능한 이미지 위젯"""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._pix    = pixmap
        self._scale  = 1.0
        self._offset = QPoint(0, 0)
        self._drag_start = None
        self._drag_offset_start = None
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        if self._pix.isNull():
            return
        sw = int(self._pix.width()  * self._scale)
        sh = int(self._pix.height() * self._scale)
        x  = (self.width()  - sw) // 2 + self._offset.x()
        y  = (self.height() - sh) // 2 + self._offset.y()
        painter.drawPixmap(x, y,
                           self._pix.scaled(sw, sh,
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))
        painter.end()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._scale = max(0.05, min(self._scale * factor, 50.0))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_offset_start = QPoint(self._offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._offset = self._drag_offset_start + delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        # 더블클릭 → 원래 크기로 리셋
        self._scale  = 1.0
        self._offset = QPoint(0, 0)
        self.update()


class SJISApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = SJISPipeline()
        self.loaded_rgb = None
        self._last_open_dir = _SCRIPT_DIR   # 마지막으로 열었던 폴더 기억
        self._last_save_dir = _SCRIPT_DIR   # 마지막으로 저장했던 폴더 기억
        self._sweep_axis_state = {}          # Sweep 창 축 설정 유지용
        self.initUI()

    def create_load_row(self, label, default_txt, filter_ext):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        lbl = QLabel(label); lbl.setMinimumWidth(40)
        line_edit = QLineEdit(default_txt)
        btn = QPushButton("Load")
        btn.clicked.connect(lambda: self.browse_file(line_edit, filter_ext))
        layout.addWidget(lbl); layout.addWidget(line_edit); layout.addWidget(btn)
        return layout, line_edit

    def browse_file(self, line_edit, filter_ext):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", self._last_open_dir, filter_ext)
        if fname:
            line_edit.setText(fname)
            self._last_open_dir = os.path.dirname(fname)

    def save_image(self, label_widget, default_name):
        if label_widget._pixmap and not label_widget._pixmap.isNull():
            default_path = os.path.join(self._last_save_dir, default_name)
            fname, _ = QFileDialog.getSaveFileName(self, "Save Image", default_path, "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
            if fname:
                label_widget._pixmap.save(fname)
                self._last_save_dir = os.path.dirname(fname)
                self.log(f"▶ 이미지 저장 완료: {fname}")
        else:
            self.log("❌ 에러: 저장할 이미지가 생성되지 않았습니다.")

    def initUI(self):
        self.setWindowTitle("SJIS-Art Generator V20 (v0.2.0 — Sweep Fix / Param Persist / Slider Ranges)")
        self.resize(1600, 1000)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ----------------------------------------------------
        # 좌측 패널 (Left Panel - Width: 450px)
        # ----------------------------------------------------
        left_panel = QWidget()
        left_panel.setFixedWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Inputs Group
        group_input = QGroupBox("1. Inputs")
        ilay = QVBoxLayout(group_input)
        self.btn_load = QPushButton("Load Main Image")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        self.btn_load.clicked.connect(self.load_image)
        ilay.addWidget(self.btn_load)
        
        r1, self.line_font = self.create_load_row("Font:", _rel("Saitamaar.ttf"), "Font Files (*.ttf *.otf)")
        r2, self.line_csv = self.create_load_row("Chars:", _rel("char_list_freq.csv"), "CSV Files (*.csv)")
        r3, self.line_tone = self.create_load_row("Tone:", _rel("char_tone.txt"), "Text Files (*.txt)")
        w1=QWidget(); w1.setLayout(r1); ilay.addWidget(w1)
        w2=QWidget(); w2.setLayout(r2); ilay.addWidget(w2)
        w3=QWidget(); w3.setLayout(r3); ilay.addWidget(w3)
        
        self.spin_lines = SliderSpinBox("Text Lines:", 10, 200, 1, 40, is_float=False)
        ilay.addWidget(self.spin_lines)
        left_layout.addWidget(group_input)

        # 2. Parameters Columns (HBox)
        param_cols = QHBoxLayout()
        
        col1_widget = QWidget()
        col1_lay = QVBoxLayout(col1_widget)
        col1_lay.setContentsMargins(0,0,0,0)
        
        group_line = QGroupBox("2. Line Art")
        glay = QVBoxLayout(group_line)
        self.combo_line = QComboBox()
        self.combo_line.addItems(["Adaptive Threshold", "XDoG (Soft Sketch)", "Laplacian (Brush/Ink)", "Sobel (Directional Edge)", "Segmentation (K-means)", "Canny (Hard Edges)", "Simple (Grayscale)"])
        self.combo_line.currentIndexChanged.connect(self.toggle_line_params)
        glay.addWidget(self.combo_line)
        self.spin_kmeans_k = SliderSpinBox("K-means K:", 2, 16, 1, 3, is_float=False)
        self.spin_kmeans_k.setEnabled(False)
        glay.addWidget(self.spin_kmeans_k)
        self.spin_thresh = SliderSpinBox("Threshold:", 0, 255, 1, 127, False)
        self.spin_thick = SliderSpinBox("Thickness:", 0.1, 10.0, 0.1, 2.5)
        self.spin_clean = SliderSpinBox("Clean Area:", 0.0, 50.0, 0.5, 1.5, is_float=True)
        glay.addWidget(self.spin_thresh); glay.addWidget(self.spin_thick); glay.addWidget(self.spin_clean)
        
        self.combo_thin = QComboBox()
        self.combo_thin.addItems(["K3M", "KMM", "Guo-hall", "None (No Thinning)"])
        glay.addWidget(self.combo_thin)
        col1_lay.addWidget(group_line)

        group_aa = QGroupBox("3. Generation")
        c2_lay = QVBoxLayout(group_aa)
        
        self.combo_place = QComboBox()
        self.combo_place.addItems(["Score-Priority", "Sequential"])
        c2_lay.addWidget(self.combo_place)
        
        self.spin_pha = SliderSpinBox("Phase W:", 0.0, 10.0, 0.5, 5.0)
        self.spin_den = SliderSpinBox("Density Pen:", 0.0, 1.0, 0.05, 0.45)
        self.spin_mis = SliderSpinBox("Missing Pen:", 0.0, 0.5, 0.05, 0.05)
        self.spin_frq = SliderSpinBox("Freq Bonus:", 0.0, 10.0, 0.1, 1.0)
        self.spin_dot_pen = SliderSpinBox("Dot Pen:", 0.0, 20.0, 0.5, 2.0)
        
        c2_lay.addWidget(self.spin_pha); c2_lay.addWidget(self.spin_den); c2_lay.addWidget(self.spin_mis)
        c2_lay.addWidget(self.spin_frq); c2_lay.addWidget(self.spin_dot_pen)
        
        self.spin_ytol = SliderSpinBox("Y-Tolerance:", 0, 10, 1, 1, False)
        self.spin_yp_pen = SliderSpinBox("Y-Shift Pen:", 0.0, 20.0, 1, 5)
        self.spin_yshi = SliderSpinBox("Global Y-Shift:", -16, 16, 1, 0, False)
        
        c2_lay.addWidget(self.spin_ytol); c2_lay.addWidget(self.spin_yp_pen); c2_lay.addWidget(self.spin_yshi)
        col1_lay.addWidget(group_aa)
        col1_lay.addStretch()

        col2_widget = QWidget()
        col2_lay = QVBoxLayout(col2_widget)
        col2_lay.setContentsMargins(0,0,0,0)

        group_eye = QGroupBox("4. Eye Detailing")
        c3_lay = QVBoxLayout(group_eye)
        self.chk_roi = QCheckBox("[Mask] Enable")
        self.chk_roi.setChecked(True)
        self.chk_roi.toggled.connect(self.toggle_mask_logic)
        c3_lay.addWidget(self.chk_roi)
        
        self.spin_eye_w = SliderSpinBox("Eye Char W:", 0.0, 200.0, 5.0, 100.0)
        self.spin_roi_w = SliderSpinBox("Wide Char W:", 0.0, 50.0, 5.0, 20.0)
        self.spin_roi_pha = SliderSpinBox("ROI Phase:", 0.0, 5.0, 0.1, 0.0)
        self.spin_roi_den = SliderSpinBox("ROI Density:", 0.0, 1.0, 0.05, 0.0)
        self.spin_roi_mis = SliderSpinBox("ROI Missing:", 0.0, 1.0, 0.05, 0.0)
        self.spin_roi_frq = SliderSpinBox("ROI Freq:", 0.0, 10.0, 1, 1.0)
        
        c3_lay.addWidget(self.spin_eye_w); c3_lay.addWidget(self.spin_roi_w)
        c3_lay.addWidget(self.spin_roi_pha); c3_lay.addWidget(self.spin_roi_den)
        c3_lay.addWidget(self.spin_roi_mis); c3_lay.addWidget(self.spin_roi_frq)
        
        mask_tools = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_clear = QPushButton("Clear")
        self.btn_undo.clicked.connect(lambda: self.lbl_img1.undo())
        self.btn_clear.clicked.connect(lambda: self.lbl_img1.clear_mask())
        mask_tools.addWidget(self.btn_undo); mask_tools.addWidget(self.btn_clear)
        c3_lay.addLayout(mask_tools)
        
        self.spin_brush = SliderSpinBox("Brush Size:", 1, 100, 1, 15, False)
        self.spin_brush.slider.valueChanged.connect(lambda: setattr(self.lbl_img1, 'brush_size', self.spin_brush.value()))
        self.spin_brush.spin.valueChanged.connect(lambda: setattr(self.lbl_img1, 'brush_size', self.spin_brush.value()))
        c3_lay.addWidget(self.spin_brush)
        col2_lay.addWidget(group_eye)

        group_tone = QGroupBox("5. BG Tone")
        c4_lay = QVBoxLayout(group_tone)
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(["0: Line-art Only", "1: Full Area", "2: Fill Empty"])
        self.combo_bg.currentIndexChanged.connect(self.toggle_tone_params)
        c4_lay.addWidget(self.combo_bg)
        self.spin_tone_w = SliderSpinBox("Tone W:", 0.0, 10.0, 0.1, 1.0)
        self.spin_contrast = SliderSpinBox("Contrast:", 0.1, 3.0, 0.1, 1.0)
        self.spin_bright = SliderSpinBox("Brightness:", 0, 1.0, 0.05, 0.6)
        
        self.spin_tone_w.setEnabled(False)
        self.spin_contrast.setEnabled(False)
        self.spin_bright.setEnabled(False)
        
        c4_lay.addWidget(self.spin_tone_w); c4_lay.addWidget(self.spin_contrast); c4_lay.addWidget(self.spin_bright)
        col2_lay.addWidget(group_tone)
        
        group_custom = QGroupBox("6. Custom Char Penalty")
        c5_lay = QVBoxLayout(group_custom)
        lbl_c = QLabel("Custom Chars (제외할 문자 입력):")
        lbl_c.setStyleSheet("font-size: 11px;")
        c5_lay.addWidget(lbl_c)
        self.text_custom_chars = QTextEdit("．，‘’“”¨･…：；ﾟ°｡。~`！∫λ〝?")
        self.text_custom_chars.setPlaceholderText("ex) abc!@# (여러 줄로 입력 가능)")
        self.text_custom_chars.setFixedHeight(60)
        c5_lay.addWidget(self.text_custom_chars)
        self.spin_custom_pen = SliderSpinBox("Custom Pen:", 0.0, 20.0, 1, 10.0)
        c5_lay.addWidget(self.spin_custom_pen)
        col2_lay.addWidget(group_custom)
        col2_lay.addStretch()

        param_cols.addWidget(col1_widget)
        param_cols.addWidget(col2_widget)
        left_layout.addLayout(param_cols)

        # 3. Generate & Stop Buttons
        btn_layout = QHBoxLayout()
        self.btn_gen = QPushButton("Generate ASCII Art")
        self.btn_gen.setMinimumHeight(45)
        self.btn_gen.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_gen.clicked.connect(self.start_generation)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(45)
        self.btn_stop.setStyleSheet("font-weight: bold; background-color: #F44336; color: white;")
        self.btn_stop.clicked.connect(self.stop_generation)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_gen)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)
        
        # Parameter Sweep 버튼
        self.btn_sweep = QPushButton("🔍 Parameter Sweep (미리보기 비교)")
        self.btn_sweep.setMinimumHeight(30)
        self.btn_sweep.setStyleSheet("font-weight: bold; background-color: #9C27B0; color: white;")
        self.btn_sweep.clicked.connect(self.open_sweep_dialog)
        left_layout.addWidget(self.btn_sweep)
        
        self.pbar = QProgressBar()
        left_layout.addWidget(self.pbar)

        # 4. Execution Logs
        left_layout.addWidget(QLabel("Execution Logs:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        left_layout.addWidget(self.log_box)

        main_layout.addWidget(left_panel)

        # ----------------------------------------------------
        # 우측 패널 (Right Panel - 6 Windows Grid)
        # ----------------------------------------------------
        grid_imgs = QGridLayout()
        
        w1_lay = QVBoxLayout(); w1_lay.addWidget(QLabel("1. Original Image (+ Draw Mask Here)"))
        self.lbl_img1 = PaintableLabel(); self.lbl_img1.setFrameShape(QFrame.Box)
        w1_lay.addWidget(self.lbl_img1)
        
        w2_lay = QVBoxLayout()
        w2_header = QHBoxLayout()
        w2_header.addWidget(QLabel("2. Line Art (Extracted & Inverted)"))
        btn_save2 = QPushButton("Save"); btn_save2.setFixedWidth(50)
        btn_save2.clicked.connect(lambda: self.save_image(self.lbl_img2, "line_art.png"))
        w2_header.addWidget(btn_save2)
        w2_lay.addLayout(w2_header)
        self.lbl_img2 = AspectRatioLabel(); self.lbl_img2.setFrameShape(QFrame.Box)
        w2_lay.addWidget(self.lbl_img2)

        w3_lay = QVBoxLayout()
        w3_header = QHBoxLayout()
        w3_header.addWidget(QLabel("3. Thinned (Skeleton & Inverted)"))
        btn_save3 = QPushButton("Save"); btn_save3.setFixedWidth(50)
        btn_save3.clicked.connect(lambda: self.save_image(self.lbl_img3, "thinned.png"))
        w3_header.addWidget(btn_save3)
        w3_lay.addLayout(w3_header)
        self.lbl_img3 = AspectRatioLabel(); self.lbl_img3.setFrameShape(QFrame.Box)
        w3_lay.addWidget(self.lbl_img3)

        w4_lay = QVBoxLayout()
        w4_header = QHBoxLayout()
        w4_header.addWidget(QLabel("4. Grid Image (Visual Analysis)"))
        btn_save4 = QPushButton("Save"); btn_save4.setFixedWidth(50)
        btn_save4.clicked.connect(lambda: self.save_image(self.lbl_img4, "grid_image.png"))
        w4_header.addWidget(btn_save4)
        w4_lay.addLayout(w4_header)
        self.lbl_img4 = AspectRatioLabel(); self.lbl_img4.setFrameShape(QFrame.Box)
        w4_lay.addWidget(self.lbl_img4)

        w5_lay = QVBoxLayout()
        w5_header = QHBoxLayout()
        w5_header.addWidget(QLabel("5. Rendered AA Image"))
        btn_save5 = QPushButton("Save"); btn_save5.setFixedWidth(50)
        btn_save5.clicked.connect(lambda: self.save_image(self.lbl_img5, "rendered_aa.png"))
        w5_header.addWidget(btn_save5)
        w5_lay.addLayout(w5_header)
        self.lbl_img5 = AspectRatioLabel(); self.lbl_img5.setFrameShape(QFrame.Box)
        w5_lay.addWidget(self.lbl_img5)
        
        w6_lay = QVBoxLayout()
        w6_header = QHBoxLayout()
        w6_header.addWidget(QLabel("6. Output (Selectable Text)"))
        btn_save_txt = QPushButton("Save .txt"); btn_save_txt.setFixedWidth(70)
        btn_save_txt.clicked.connect(self.save_output_text)
        w6_header.addWidget(btn_save_txt)
        w6_lay.addLayout(w6_header)
        self.text_out = QTextEdit(); self.text_out.setLineWrapMode(QTextEdit.NoWrap)
        font = QFont("Saitamaar", 10); font.setStyleHint(QFont.Monospace); self.text_out.setFont(font)
        w6_lay.addWidget(self.text_out)

        grid_imgs.addLayout(w1_lay, 0, 0)
        grid_imgs.addLayout(w2_lay, 0, 1)
        grid_imgs.addLayout(w3_lay, 0, 2)
        grid_imgs.addLayout(w4_lay, 1, 0)
        grid_imgs.addLayout(w5_lay, 1, 1)
        grid_imgs.addLayout(w6_lay, 1, 2)

        for i in range(3): grid_imgs.setColumnStretch(i, 1)
        for i in range(2): grid_imgs.setRowStretch(i, 1)
        
        main_layout.addLayout(grid_imgs, stretch=1)

    def save_output_text(self):
        """생성된 AA 텍스트를 .txt 파일로 저장"""
        text = self.text_out.toPlainText()
        if not text.strip():
            self.log("❌ 에러: 저장할 텍스트가 없습니다. 먼저 생성해 주세요.")
            return
        default_path = os.path.join(self._last_save_dir, "output_aa.txt")
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Output Text", default_path,
            "Text Files (*.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(text)
                self._last_save_dir = os.path.dirname(fname)
                self.log(f"▶ 텍스트 저장 완료: {fname}")
            except Exception as e:
                self.log(f"❌ 저장 실패: {str(e)}")

    def open_sweep_dialog(self):
        """파라미터 스윕 다이얼로그를 열어 여러 값에 대한 렌더링 결과를 비교"""
        if self.loaded_rgb is None:
            self.log("❌ 에러: 먼저 이미지를 로드하세요.")
            return
        
        # 현재 파라미터 기반으로 기본값 구성
        current_params = {
            'text_lines': self.spin_lines.value(),
            'font_path': self.line_font.text(),
            'char_csv': self.line_csv.text(),
            'char_tone': self.line_tone.text(),
            'line_method': self.combo_line.currentText(),
            'kmeans_k': self.spin_kmeans_k.value(),
            'thin_method': self.combo_thin.currentText(),
            'threshold': self.spin_thresh.value(),
            'thickness': self.spin_thick.value(),
            'clean': self.spin_clean.value(),
            'p_method': self.combo_place.currentText(),
            'phase_w': self.spin_pha.value(),
            'density_w': self.spin_den.value(),
            'missing_w': self.spin_mis.value(),
            'freq_w': self.spin_frq.value(),
            'dot_penalty': self.spin_dot_pen.value(),
            'custom_chars': self.text_custom_chars.toPlainText().replace('\n', ''),
            'custom_penalty': self.spin_custom_pen.value(),
            'y_tolerance': self.spin_ytol.value(),
            'y_shift_penalty': self.spin_yp_pen.value(),
            'global_y_shift': self.spin_yshi.value(),
            'use_roi': self.chk_roi.isChecked(),
            'eye_char_w': self.spin_eye_w.value(),
            'roi_weight': self.spin_roi_w.value(),
            'roi_pha_w': self.spin_roi_pha.value(),
            'roi_den_w': self.spin_roi_den.value(),
            'roi_mis_w': self.spin_roi_mis.value(),
            'roi_frq_w': self.spin_roi_frq.value(),
            'bg_mode': self.combo_bg.currentText(),
            'tone_weight': self.spin_tone_w.value(),
            'contrast': self.spin_contrast.value(),
            'brightness': self.spin_bright.value()
        }
        mask_bin = self.lbl_img1.mask_img.copy() if self.lbl_img1.mask_img is not None else None
        
        dlg = SweepDialog(self, self.pipeline, self.loaded_rgb, mask_bin, current_params)
        dlg.exec_()

    def log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def toggle_line_params(self, index):
        method = self.combo_line.currentText()
        if "K-means" in method:
            self.spin_kmeans_k.setEnabled(True)
            self.spin_thresh.setEnabled(False)
        else:
            self.spin_kmeans_k.setEnabled(False)
            self.spin_thresh.setEnabled(True)

    def toggle_mask_logic(self, is_checked):
        self.btn_undo.setEnabled(is_checked)
        self.btn_clear.setEnabled(is_checked)
        self.spin_brush.setEnabled(is_checked)
        self.spin_eye_w.setEnabled(is_checked)
        self.spin_roi_w.setEnabled(is_checked)
        self.spin_roi_pha.setEnabled(is_checked)
        self.spin_roi_den.setEnabled(is_checked)
        self.spin_roi_mis.setEnabled(is_checked)
        self.spin_roi_frq.setEnabled(is_checked)
        
        self.lbl_img1.drawing_enabled = is_checked
        if not is_checked:
            if self.lbl_img1.mask_img is not None:
                self.lbl_img1.mask_img.fill(0)
            self.lbl_img1.undo_stack.clear()
            self.lbl_img1.update_display()

    def toggle_tone_params(self, index):
        mode_text = self.combo_bg.currentText()
        if mode_text.startswith("0"): 
            self.spin_tone_w.setEnabled(False)
            self.spin_contrast.setEnabled(False)
            self.spin_bright.setEnabled(False)
            self.spin_ytol.setEnabled(True)
        elif mode_text.startswith("1"): 
            self.spin_tone_w.setEnabled(True)
            self.spin_contrast.setEnabled(True)
            self.spin_bright.setEnabled(True)
            self.spin_ytol.setEnabled(False) 
        else: 
            self.spin_tone_w.setEnabled(True)
            self.spin_contrast.setEnabled(True)
            self.spin_bright.setEnabled(True)
            self.spin_ytol.setEnabled(True)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", self._last_open_dir, "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname:
            self._last_open_dir = os.path.dirname(fname)
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                self.log("❌ 에러: 이미지를 읽을 수 없습니다.")
                return

            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bgr = img[:, :, :3]
                bg_white = np.ones_like(bgr, dtype=np.uint8) * 255
                for c in range(3):
                    bg_white[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * bg_white[:, :, c]).astype(np.uint8)
                img = bg_white
            elif len(img.shape) == 2: 
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.loaded_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = self.loaded_rgb.shape
            
            qimg = QImage(self.loaded_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.lbl_img1.set_image(QPixmap.fromImage(qimg))
            
            self.lbl_img2.setPixmap(QPixmap())
            self.lbl_img3.setPixmap(QPixmap())
            self.lbl_img4.setPixmap(QPixmap())
            self.lbl_img5.setPixmap(QPixmap())
            self.text_out.clear()
            self.log("▶ 메인 이미지 로드 완료!")

    def stop_generation(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.btn_stop.setEnabled(False)
            self.log("⏸️ 정지 버튼 클릭됨... 작업을 안전하게 종료하는 중입니다.")

    def start_generation(self):
        if self.loaded_rgb is None:
            self.log("❌ 에러: 먼저 이미지를 로드하세요.")
            return
            
        params = {
            'text_lines': self.spin_lines.value(),
            'font_path': self.line_font.text(),
            'char_csv': self.line_csv.text(),
            'char_tone': self.line_tone.text(),
            'line_method': self.combo_line.currentText(),
            'kmeans_k': self.spin_kmeans_k.value(),
            'thin_method': self.combo_thin.currentText(),
            'threshold': self.spin_thresh.value(),
            'thickness': self.spin_thick.value(),
            'clean': self.spin_clean.value(),
            
            'p_method': self.combo_place.currentText(),
            'phase_w': self.spin_pha.value(),
            'density_w': self.spin_den.value(),
            'missing_w': self.spin_mis.value(),
            'freq_w': self.spin_frq.value(),
            'dot_penalty': self.spin_dot_pen.value(),
            'custom_chars': self.text_custom_chars.toPlainText().replace('\n', ''),
            'custom_penalty': self.spin_custom_pen.value(),            
            'y_tolerance': self.spin_ytol.value(),
            'y_shift_penalty': self.spin_yp_pen.value(),
            'global_y_shift': self.spin_yshi.value(),
            
            'use_roi': self.chk_roi.isChecked(),
            'eye_char_w': self.spin_eye_w.value(),
            'roi_weight': self.spin_roi_w.value(),
            'roi_pha_w': self.spin_roi_pha.value(),
            'roi_den_w': self.spin_roi_den.value(),
            'roi_mis_w': self.spin_roi_mis.value(),
            'roi_frq_w': self.spin_roi_frq.value(),
            
            'bg_mode': self.combo_bg.currentText(),
            'tone_weight': self.spin_tone_w.value(),
            'contrast': self.spin_contrast.value(),
            'brightness': self.spin_bright.value()
        }
        
        self.btn_gen.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.pbar.setValue(0)
        self.text_out.clear()
        
        mask_bin = self.lbl_img1.mask_img.copy() if self.lbl_img1.mask_img is not None else None
        
        self.thread = WorkerThread(self.pipeline, self.loaded_rgb, mask_bin, params)
        self.thread.sig_progress.connect(self.pbar.setValue)
        self.thread.sig_log.connect(self.log)
        
        self.thread.sig_line_done.connect(self.update_img2)
        self.thread.sig_thinned_done.connect(self.update_img3)
        self.thread.sig_finished.connect(self.on_generation_finished)
        self.thread.start()

    def update_img2(self, bin_img):
        h, w = bin_img.shape
        inv_arr = 255 - bin_img
        qimg = QImage(inv_arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.lbl_img2.setPixmap(QPixmap.fromImage(qimg))
        
    def update_img3(self, thinned_img):
        h, w = thinned_img.shape
        inv_arr = 255 - thinned_img
        qimg = QImage(inv_arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.lbl_img3.setPixmap(QPixmap.fromImage(qimg))

    def on_generation_finished(self, text, grid_vis, aa_img_arr):
        self.btn_gen.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if text:
            self.text_out.setPlainText(text)
            
            if grid_vis is not None:
                h, w, ch = grid_vis.shape
                qimg_grid = QImage(grid_vis.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.lbl_img4.setPixmap(QPixmap.fromImage(qimg_grid))
                
            if aa_img_arr is not None:
                h, w, ch = aa_img_arr.shape
                qimg_aa = QImage(aa_img_arr.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.lbl_img5.setPixmap(QPixmap.fromImage(qimg_aa))
        else:
            self.log("❌ 작업 중지 또는 실패.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SJISApp()
    ex.show()
    sys.exit(app.exec_())
