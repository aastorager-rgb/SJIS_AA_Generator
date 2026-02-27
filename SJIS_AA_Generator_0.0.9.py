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
# 1. Global Settings & Typography Helpers
# ==========================================
FONT_SIZE = 16
LINE_SPACING = 2
ROW_HEIGHT = FONT_SIZE + LINE_SPACING

DENSE_KANJI = set(['瀟', '憎', '浄', '占', '李', '斗', '狄', '灘', '濾', '鼎', '撼'])
EYE_UP_LEFT = set(list("だ灯衍行仍了乍仡乞云伝芸茫忙它佗俐仗なｨｪｵｴﾃﾇﾏﾓ"))
EYE_UP_CENTER = set(list("不示宍亦兀亢万迩尓禾乏弌弍弐泛夾赱符≡女乍气旡まみてテチ"))
EYE_UP_RIGHT = set(list("豺犾狄勿下卞抃圷圦坏心沁气汽斥拆仔竹刃刈付以雫爿なうかて刈ﾊ､"))
EYE_LOW_LEFT = set(list("芍弋爪心父戈弌弍弐式汽辷込乂癶廴匕丈叱杙之比仆トヽヾゝゞ㌧､"))
EYE_LOW_RIGHT = set(list("歹万久刋升刈乃汐沙少炒梦斗孑才必瓜欠次亥圦乂ノソルツ八㌧㌢㌃㍗㌣㌻"))
EYE_IDIOMS_ALL = EYE_UP_LEFT | EYE_UP_CENTER | EYE_UP_RIGHT | EYE_LOW_LEFT | EYE_LOW_RIGHT

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
            
            if method == "DoG (Soft Lines)":
                img_blur = cv2.GaussianBlur(255-gray, (0, 0), sigmaX=max(0.5, line_thickness * 0.5))
                lines = cv2.divide(255-gray, 255-img_blur, scale=256)
                _, binary = cv2.threshold(lines, threshold, 255, cv2.THRESH_BINARY_INV)
                binary = 255 - binary
            elif method == "Canny (Hard Edges)": 
                binary = cv2.Canny(gray, threshold // 2, threshold)
            elif method == "Adaptive Threshold":
                c_val = (threshold - 127) / 5.0 + 5.0 
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, c_val)
            else: 
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        scale = target_h / (40.0 * ROW_HEIGHT)
        act_t = max(1, int(round(line_thickness * scale)))
        
        if act_t > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (act_t, act_t))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
        return binary if invert_output else 255 - binary

    def process_thinning(self, binary_img, clean_strength, method):
        binary = binary_img.copy()
        if np.mean(binary) > 127: binary = 255 - binary
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        if clean_strength > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_strength*2+1, clean_strength*2+1))
            binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        
        try:
            if method == "KMM": return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            elif method == "Guo-hall": return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        except: pass
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if method == "KMM" else (cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) if method == "Guo-hall" else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
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
            for i, c in enumerate(chars):
                w = get_char_width(c, font, dummy_draw)
                flags = 0
                if c in EYE_IDIOMS_ALL: flags |= 1
                if c in EYE_UP_LEFT or c in EYE_LOW_LEFT: flags |= 2
                if c in EYE_UP_CENTER: flags |= 4
                if c in EYE_UP_RIGHT or c in EYE_LOW_RIGHT: flags |= 8
                if c in EYE_UP_LEFT or c in EYE_UP_CENTER or c in EYE_UP_RIGHT: flags |= 16
                if c in EYE_LOW_LEFT or c in EYE_LOW_RIGHT: flags |= 32
                
                if c.strip() and w > 0:
                    img_char = Image.new("L", (w, ROW_HEIGHT), 0)
                    ImageDraw.Draw(img_char).text((0, 0), c, font=font, fill=255)
                    char_arr = np.array(img_char).astype(np.uint8)
                    c_cos, c_sin, _ = calculate_orientation_map(char_arr)
                    c_mask = (char_arr > 0).astype(np.float32)
                    char_data_cache.append({'char': c, 'mask': c_mask, 'cos_strict': c_cos * c_mask, 'sin_strict': c_sin * c_mask, 'width': w, 'freq_score': freq_scores[i], 'ink': np.sum(c_mask), 'flags': flags})
                else: 
                    char_data_cache.append({'char': c, 'mask': None, 'width': max(1, w), 'freq_score': freq_scores[i], 'ink': 0, 'flags': flags})

            for idx, data in enumerate(char_data_cache):
                cw = data['width']
                if data['mask'] is None: continue
                if cw not in char_groups:
                    char_groups[cw] = {'indices': [], 'masks': [], 'cos_stricts': [], 'sin_stricts': [], 'inks': [], 'freqs': [], 'flags': [], 'is_dense': []}
                char_groups[cw]['indices'].append(idx)
                char_groups[cw]['masks'].append(torch.from_numpy(data['mask']).unsqueeze(0))
                char_groups[cw]['cos_stricts'].append(torch.from_numpy(data['cos_strict']).unsqueeze(0))
                char_groups[cw]['sin_stricts'].append(torch.from_numpy(data['sin_strict']).unsqueeze(0))
                char_groups[cw]['inks'].append(data['ink'])
                char_groups[cw]['freqs'].append(data['freq_score'])
                char_groups[cw]['flags'].append(data['flags'])
                char_groups[cw]['is_dense'].append(data['char'] in DENSE_KANJI)
            
            for cw in char_groups:
                for k in ['masks', 'cos_stricts', 'sin_stricts']: char_groups[cw][k] = torch.stack(char_groups[cw][k]).to(self._device).to(torch.float32)
                for k in ['inks', 'freqs']: char_groups[cw][k] = torch.tensor(char_groups[cw][k], dtype=torch.float32, device=self._device)
                char_groups[cw]['flags'] = torch.tensor(char_groups[cw]['flags'], dtype=torch.int32, device=self._device)
                char_groups[cw]['is_dense'] = torch.tensor(char_groups[cw]['is_dense'], dtype=torch.bool, device=self._device)
                
            self._char_list_cache = chars
            self._char_groups_cache = char_groups
            self._char_data_cache = char_data_cache
            self._current_font_path = font_path
            self._current_csv_path = char_list_path

        return self._char_data_cache, font

    def _get_gap_string(self, width, is_l):
        if width <= 0: return ""
        fw_count = int(round(width)) // self._fw_width
        rem = int(round(width)) % self._fw_width
        res = ""
        if rem >= self._hw_width + self._dot_width: res = ". "
        elif rem >= self._hw_width: res = " "
        elif rem >= self._dot_width: res = "."
        res += "　" * fw_count
        return res

    def _solve_stripe_sequential(self, scores, char_data, w, spacing, last_ink_x, is_roi_mask):
        cur_x = 0.0; line = ""; placements = []; is_l = True
        while int(round(cur_x)) <= last_ink_x and int(round(cur_x)) < w:
            sx = int(round(cur_x))
            best_idx = np.argmax(scores[sx, :])
            if scores[sx, best_idx] <= 0.0:
                gw = min(self._fw_width, w - sx)
                gs = self._get_gap_string(gw, is_l)
                if not gs: 
                    cur_x += max(1.0, float(gw)); continue
                line += gs
                for gc in gs:
                    gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                    ir = np.any(is_roi_mask[int(cur_x):min(w, int(cur_x+gcw))])
                    placements.append((gc, float(cur_x), gcw, ir))
                    cur_x += gcw
                is_l = False; continue
            char_info = char_data[best_idx]
            cw = char_info['width']
            line += char_info['char']
            ir = np.any(is_roi_mask[sx:min(w, sx+cw)])
            placements.append((char_info['char'], float(cur_x), cw, ir))
            cur_x += max(1.0, float(cw) + spacing)
            is_l = False
        return line, placements

    def _solve_stripe_score_priority(self, score_matrix, char_data_list, w, spacing, last_ink_x, is_roi_mask):
        placements = []
        scores = score_matrix.copy()
        if last_ink_x < w: scores[last_ink_x:, :] = -99999.0
            
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
            placements.append((c, float(bx), cw, ir))
            
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
        line = ""; cx = 0.0; isl = True; ap = []
        for c, x, cw, ir in placements:
            tx = max(cx, float(x))
            gp = tx - cx
            if gp >= self._dot_width:
                gs = self._get_gap_string(gp, isl)
                line += gs
                for gc in gs:
                    gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                    gc_ir = np.any(is_roi_mask[int(cx):min(w, int(cx+gcw))])
                    ap.append((gc, cx, gcw, gc_ir))
                    cx += gcw
            ap.append((c, cx, cw, ir))
            line += c; cx += max(1.0, float(cw) + spacing); isl = False
            
        if cx < w - self._dot_width:
            gs = self._get_gap_string(w - cx, isl)
            line += gs
            for gc in gs:
                gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                gc_ir = np.any(is_roi_mask[int(cx):min(w, int(cx+gcw))])
                ap.append((gc, cx, gcw, gc_ir))
                cx += gcw
        return line, ap

    def solve_stripe_hybrid(self, row_img_bin, row_tone, row_cos, row_sin, row_roi, row_roi_weights, char_data_list, spacing, y_t, params):
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
            return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask)

        with torch.no_grad():
            t_str = torch.from_numpy(row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_cos = torch.from_numpy(row_cos * row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_sin = torch.from_numpy(row_sin * row_img_bin).unsqueeze(0).unsqueeze(0).to(self._device)
            t_tne = torch.from_numpy(tone_map).unsqueeze(0).unsqueeze(0).to(self._device)
            t_roi = torch.from_numpy(row_roi).unsqueeze(0).unsqueeze(0).to(torch.int32).to(self._device)
            t_wgt = torch.from_numpy(row_roi_weights).unsqueeze(0).unsqueeze(0).to(self._device)
            
            Y_out = H - h + 1
            if Y_out <= 0: return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask)
            y_pen = (torch.abs(torch.arange(Y_out, device=self._device).view(Y_out, 1) - y_t) * params['y_shift_penalty']).unsqueeze(0)

            for cw, g in self._char_groups_cache.items():
                N = len(g['indices'])
                ones_k = torch.ones((1, 1, h, cw), dtype=torch.float32, device=self._device)
                
                ov2 = torch.nn.functional.conv2d(t_str, g['masks']).squeeze(0)
                m_cos = torch.nn.functional.conv2d(t_cos, g['cos_stricts']).squeeze(0)
                m_sin = torch.nn.functional.conv2d(t_sin, g['sin_stricts']).squeeze(0)
                pha2 = (ov2 - (m_cos + m_sin)) * 0.5
                t_ink2 = torch.nn.functional.conv2d(t_str, ones_k).squeeze(0)
                
                b_list = [(torch.nn.functional.conv2d((t_roi & (1<<k)).float(), ones_k).squeeze(0) > 0) for k in range(6)]
                b1, b2, b4, b8, b16, b32 = b_list[0], b_list[1], b_list[2], b_list[3], b_list[4], b_list[5]
                
                blob_centrality = torch.nn.functional.conv2d(t_wgt, ones_k).squeeze(0) / (h * cw)
                exc, mis = torch.relu(g['inks'].view(N, 1, 1) - ov2), torch.relu(t_ink2 - ov2)
                
                if params['use_roi']:
                    lerp_w = lambda r, b: b + (r - b) * blob_centrality
                    cur_w_den = lerp_w(params['roi_den_w'], params['density_w'])
                    cur_w_mis = lerp_w(params['roi_mis_w'], params['missing_w'])
                    cur_w_pha = lerp_w(params['roi_pha_w'], params['phase_w'])
                    cur_w_frq = lerp_w(params['roi_frq_w'], params['freq_w'])
                else:
                    cur_w_den, cur_w_mis, cur_w_pha, cur_w_frq = params['density_w'], params['missing_w'], params['phase_w'], params['freq_w']

                calc = ov2 - pha2*cur_w_pha - exc*cur_w_den - mis*cur_w_mis + g['freqs'].view(N,1,1)*cur_w_frq - y_pen
                
                c_req = g['flags'] & ~1
                spatial_match = (~((c_req & 2).bool().view(N,1,1)) | b2) & (~((c_req & 4).bool().view(N,1,1)) | b4) & (~((c_req & 8).bool().view(N,1,1)) | b8) & (~((c_req & 16).bool().view(N,1,1)) | b16) & (~((c_req & 32).bool().view(N,1,1)) | b32)
                
                is_eye = (g['flags'] & 1) > 0
                val_s = b1 & spatial_match & (ov2 > 0)
                
                # --- Eye Character Weight ---
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
            return self._solve_stripe_score_priority(scores, char_data_list, W, spacing, last_x, is_roi_mask)
        return self._solve_stripe_sequential(scores, char_data_list, W, spacing, last_x, is_roi_mask)

    def generate(self, ori_rgb, thinned_bin, mask_bin, params, progress_callback, log_callback):
        log_callback("진행 1/5: 리소스 로드 중...")
        res = self.load_resources(params['char_csv'], params['font_path'], params['char_tone'])
        if not res: return None, None, None
        char_data_list, font = res
        
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
                        roi_weight_map[y:y+h, x:x+w][blob_mask] = dist[blob_mask] / dist.max()
                    hh, hw = h//2, w//2
                    roi_map[y:y+hh, x:x+w][blob_mask[0:hh, :]] |= 16
                    roi_map[y+hh:y+h, x:x+w][blob_mask[hh:h, :]] |= 32
                    roi_map[y:y+h, x:x+hw][blob_mask[:, 0:hw]] |= 2
                    roi_map[y:y+h, x+hw:x+w][blob_mask[:, hw:w]] |= 8
                    
        tasks = []
        m_y = 0 if params['bg_mode'].startswith("1") else params['y_tolerance']
        for r in range(params['text_lines']):
            ys = r * ROW_HEIGHT; ye = min(ys + ROW_HEIGHT, target_h)
            sys, sye = max(0, ys-m_y), min(target_h, ye+m_y)
            pt, pb = max(0, -(ys-m_y)), max(0, (ys+ROW_HEIGHT+m_y)-target_h)
            tasks.append([
                np.pad(img_bin_f32[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_tone[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_cos[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_sin[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_weight_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant')
            ])
            
        res_l = [""] * len(tasks)
        all_p = [[] for _ in range(len(tasks))]
        replaced_boxes = [[] for _ in range(len(tasks))]
        
        log_callback(f"진행 2/5: 글자 배치 진행 중... (총 {len(tasks)}줄)")
        for idx, t in enumerate(tasks):
            res_l[idx], all_p[idx] = self.solve_stripe_hybrid(t[0], t[1], t[2], t[3], t[4], t[5], char_data_list, 0.0, m_y, params)
            progress_callback(int((idx / len(tasks)) * 50))
            if idx % 10 == 0: torch.cuda.empty_cache()

        if params['bg_mode'].startswith("2") and params['tone_weight'] > 0 and self._tone_chars_cache:
            log_callback("진행 3/5: 빈 공간 톤 채우기...")
            tone_chars = self._tone_chars_cache
            REPLACEABLE_CHARS = set(['　', ' ', '.', ',', "'", '．', '，']) 
            
            def get_best_tone_string_dynamic(patch, bg_weight):
                target_width = patch.shape[1]
                if target_width == 0: return None
                dp = {0: (0.0, 0, "")}
                for w in range(1, target_width + 1):
                    best_cost = float('inf'); best_prev = 0; best_char = ""
                    for tc_tone, tc_char, tc_width in tone_chars:
                        if tc_width <= 0 or w < tc_width: continue
                        if (w - tc_width) in dp:
                            char_patch = patch[:, w - tc_width : w]
                            local_tone = (1.0 - (np.mean(char_patch) / 255.0)) * bg_weight if char_patch.size > 0 else 0.0
                            cost = dp[w - tc_width][0] + abs(tc_tone - local_tone)
                            if cost < best_cost: best_cost = cost; best_prev = w - tc_width; best_char = tc_char
                    if best_cost < float('inf'): dp[w] = (best_cost, best_prev, best_char)
                if target_width in dp:
                    res_str = ""; curr = target_width
                    while curr > 0:
                        _, prev, ch = dp[curr]
                        res_str = ch + res_str; curr = prev
                    return res_str
                return None 

            for idx, t in enumerate(tasks):
                row_tone = t[1][m_y:m_y+ROW_HEIGHT, :]
                new_line = ""; chunk_chars = ""; chunk_w = 0.0; chunk_start_x = 0.0
                
                def flush_chunk():
                    nonlocal new_line, chunk_chars, chunk_w, chunk_start_x
                    if chunk_w > 0:
                        ix, icw = int(round(chunk_start_x)), int(round(chunk_w))
                        patch = row_tone[:, ix:ix+icw]
                        if patch.size > 0 and (1.0 - (np.min(patch) / 255.0)) * params['tone_weight'] > 0.05:
                            filled = get_best_tone_string_dynamic(patch, params['tone_weight'])
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

        log_callback("진행 4/5: 우측 공백 트림 및 그리드 생성 중...")
        for i in range(len(res_l)):
            res_l[i] = re.sub(r'[　 \.,\'．，]+$', '', res_l[i])
        
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
# 3. Custom UI Widgets (Upgraded SliderSpinBox)
# ==========================================
class SliderSpinBox(QWidget):
    def __init__(self, label, min_val, max_val, step, default_val, is_float=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(125)
        
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
            
        # 화살표 제거 후 텍스트 가운데 정렬
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setFixedWidth(50)
        self.spin.setAlignment(Qt.AlignCenter)
        
        self.btn_minus = QPushButton("◀")
        self.btn_minus.setFixedWidth(20)
        self.btn_minus.clicked.connect(self.decrement)
        
        self.btn_plus = QPushButton("▶")
        self.btn_plus.setFixedWidth(20)
        self.btn_plus.clicked.connect(self.increment)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.scale), int(max_val * self.scale))
        self.slider.setValue(int(default_val * self.scale))
        
        self.spin.valueChanged.connect(self.update_slider)
        self.slider.valueChanged.connect(self.update_spin)
        
        # [라벨] [◀] [슬라이더] [▶] [숫자입력창] 레이아웃
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
        self.brush_size = 20
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

    def run(self):
        try:
            self.sig_log.emit("▶ [1/3] 선화 추출 시작...")
            binary = self.p.extract_lines(self.img_rgb, self.params['text_lines'], self.params['line_method'], self.params['threshold'], self.params['thickness'], self.params['kmeans_k'], True)
            self.sig_line_done.emit(binary)
            
            self.sig_log.emit("▶ [2/3] 세선화 (Thinning) 시작...")
            thinned = self.p.process_thinning(binary, self.params['clean'], self.params['thin_method'])
            self.sig_thinned_done.emit(thinned)
            
            self.sig_log.emit("▶ [3/3] 아스키 아트 생성 중...")
            aa_text, grid_vis, aa_img = self.p.generate(self.img_rgb, thinned, self.mask_bin, self.params, self.sig_progress.emit, self.sig_log.emit)
            
            self.sig_finished.emit(aa_text, grid_vis, aa_img)
        except Exception as e:
            self.sig_log.emit(f"❌ 오류 발생:\n{str(e)}")
            self.sig_finished.emit("", None, None)

class SJISApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = SJISPipeline()
        self.loaded_rgb = None
        self.initUI()

    def create_load_row(self, label, default_txt, filter_ext):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        lbl = QLabel(label); lbl.setMinimumWidth(50)
        line_edit = QLineEdit(default_txt)
        btn = QPushButton("Load")
        btn.clicked.connect(lambda: self.browse_file(line_edit, filter_ext))
        layout.addWidget(lbl); layout.addWidget(line_edit); layout.addWidget(btn)
        return layout, line_edit

    def browse_file(self, line_edit, filter_ext):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", filter_ext)
        if fname: line_edit.setText(fname)

    def initUI(self):
        self.setWindowTitle("SJIS-Art Generator (Portable V8)")
        self.resize(1600, 1000)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ----------------------------------------------------
        # 좌측 패널 (Left Panel: Inputs & Params & Logs)
        # ----------------------------------------------------
        left_panel = QWidget()
        left_panel.setFixedWidth(730)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Inputs Group
        group_input = QGroupBox("1. Inputs & Process")
        ilay = QVBoxLayout(group_input)
        self.btn_load = QPushButton("Load Main Image")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        self.btn_load.clicked.connect(self.load_image)
        ilay.addWidget(self.btn_load)
        
        r1, self.line_font = self.create_load_row("Font:", "Saitamaar.ttf", "Font Files (*.ttf *.otf)")
        r2, self.line_csv = self.create_load_row("Chars:", "char_list_freq.csv", "CSV Files (*.csv)")
        r3, self.line_tone = self.create_load_row("Tone:", "char_tone.txt", "Text Files (*.txt)")
        w1=QWidget(); w1.setLayout(r1); ilay.addWidget(w1)
        w2=QWidget(); w2.setLayout(r2); ilay.addWidget(w2)
        w3=QWidget(); w3.setLayout(r3); ilay.addWidget(w3)
        
        self.spin_lines = SliderSpinBox("Text Lines:", 10, 200, 1, 40, is_float=False)
        ilay.addWidget(self.spin_lines)
        
        self.btn_gen = QPushButton("Generate ASCII Art")
        self.btn_gen.setMinimumHeight(45)
        self.btn_gen.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_gen.clicked.connect(self.start_generation)
        ilay.addWidget(self.btn_gen)
        
        self.pbar = QProgressBar()
        ilay.addWidget(self.pbar)
        left_layout.addWidget(group_input)

        # 2. Parameters Columns (HBox)
        param_cols = QHBoxLayout()
        
        # Col 1: Line, Thinning, AA Params
        col1_widget = QWidget()
        col1_lay = QVBoxLayout(col1_widget)
        col1_lay.setContentsMargins(0,0,0,0)
        
        group_line = QGroupBox("2. Line Art & Thinning")
        glay = QVBoxLayout(group_line)
        self.combo_line = QComboBox()
        self.combo_line.addItems(["Segmentation (K-means)", "Adaptive Threshold", "Canny (Hard Edges)", "DoG (Soft Lines)", "Simple (Grayscale)"])
        self.combo_line.currentIndexChanged.connect(self.toggle_line_params)
        glay.addWidget(self.combo_line)
        self.spin_kmeans_k = SliderSpinBox("K-means K:", 2, 16, 1, 3, is_float=False)
        self.spin_kmeans_k.setEnabled(True)
        glay.addWidget(self.spin_kmeans_k)
        self.spin_thresh = SliderSpinBox("Threshold:", 0, 255, 1, 127, False)
        self.spin_thick = SliderSpinBox("Thickness:", 0.1, 10.0, 0.1, 2.5)
        self.spin_clean = SliderSpinBox("Clean Strength:", 0, 5, 1, 0, False)
        glay.addWidget(self.spin_thresh); glay.addWidget(self.spin_thick); glay.addWidget(self.spin_clean)
        
        self.combo_thin = QComboBox()
        self.combo_thin.addItems(["K3M", "KMM", "Guo-hall"])
        glay.addWidget(self.combo_thin)
        col1_lay.addWidget(group_line)

        group_aa = QGroupBox("3. Generation Params")
        c2_lay = QVBoxLayout(group_aa)
        
        self.combo_place = QComboBox()
        self.combo_place.addItems(["Score-Priority", "Sequential"])
        c2_lay.addWidget(self.combo_place)
        
        self.spin_pha = SliderSpinBox("Phase Weight:", 0.0, 10.0, 0.1, 1.0)
        self.spin_den = SliderSpinBox("Density Penalty:", 0.0, 10.0, 0.1, 0.6)
        self.spin_mis = SliderSpinBox("Missing Penalty:", 0.0, 10.0, 0.1, 0.4)
        self.spin_frq = SliderSpinBox("Freq Bonus:", 0.0, 20.0, 0.1, 0.5)
        self.spin_ytol = SliderSpinBox("Y-Tolerance:", 0, 10, 1, 2, False)
        self.spin_yp_pen = SliderSpinBox("Y-Shift Penalty:", 0.0, 10.0, 0.1, 0.5)
        self.spin_yshi = SliderSpinBox("Global Y-Shift:", -16, 16, 1, 0, False)
        
        c2_lay.addWidget(self.spin_pha); c2_lay.addWidget(self.spin_den); c2_lay.addWidget(self.spin_mis)
        c2_lay.addWidget(self.spin_frq); c2_lay.addWidget(self.spin_ytol); c2_lay.addWidget(self.spin_yp_pen); c2_lay.addWidget(self.spin_yshi)
        col1_lay.addWidget(group_aa)
        col1_lay.addStretch()

        # Col 2: Eye Mask, Tone
        col2_widget = QWidget()
        col2_lay = QVBoxLayout(col2_widget)
        col2_lay.setContentsMargins(0,0,0,0)

        group_eye = QGroupBox("4. Eye Detailing (ROI Mask)")
        c3_lay = QVBoxLayout(group_eye)
        self.chk_roi = QCheckBox("[Mask] Enable Eye Detailing")
        self.chk_roi.setChecked(True)
        self.chk_roi.toggled.connect(self.toggle_mask_logic)
        c3_lay.addWidget(self.chk_roi)
        
        self.spin_eye_w = SliderSpinBox("Eye Char Weight:", 0.0, 500.0, 1.0, 100.0)
        self.spin_roi_w = SliderSpinBox("Wide Char Weight (≥8px):", 0.0, 50.0, 1.0, 15.0)
        self.spin_roi_pha = SliderSpinBox("ROI Phase Weight:", 0.0, 10.0, 0.1, 0.0)
        self.spin_roi_den = SliderSpinBox("ROI Density Penalty:", 0.0, 10.0, 0.1, 0.0)
        self.spin_roi_mis = SliderSpinBox("ROI Missing Penalty:", 0.0, 10.0, 0.1, 0.3)
        self.spin_roi_frq = SliderSpinBox("ROI Freq Bonus:", 0.0, 10.0, 0.1, 1.0)
        
        c3_lay.addWidget(self.spin_eye_w); c3_lay.addWidget(self.spin_roi_w)
        c3_lay.addWidget(self.spin_roi_pha); c3_lay.addWidget(self.spin_roi_den)
        c3_lay.addWidget(self.spin_roi_mis); c3_lay.addWidget(self.spin_roi_frq)
        
        mask_tools = QHBoxLayout()
        self.btn_undo = QPushButton("Undo Mask")
        self.btn_clear = QPushButton("Clear Mask")
        self.btn_undo.clicked.connect(lambda: self.lbl_img1.undo())
        self.btn_clear.clicked.connect(lambda: self.lbl_img1.clear_mask())
        mask_tools.addWidget(self.btn_undo); mask_tools.addWidget(self.btn_clear)
        c3_lay.addLayout(mask_tools)
        
        self.spin_brush = SliderSpinBox("Brush Size:", 1, 100, 1, 6, False)
        self.spin_brush.slider.valueChanged.connect(lambda: setattr(self.lbl_img1, 'brush_size', self.spin_brush.value()))
        self.spin_brush.spin.valueChanged.connect(lambda: setattr(self.lbl_img1, 'brush_size', self.spin_brush.value()))
        c3_lay.addWidget(self.spin_brush)
        col2_lay.addWidget(group_eye)

        group_tone = QGroupBox("5. Background Tone")
        c4_lay = QVBoxLayout(group_tone)
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(["0: Line-art Only", "1: Full Area Tone", "2: Fill Empty Spaces"])
        self.combo_bg.currentIndexChanged.connect(self.toggle_tone_params)
        c4_lay.addWidget(self.combo_bg)
        self.spin_tone_w = SliderSpinBox("Tone Weight:", 0.0, 10.0, 0.1, 1.0)
        self.spin_contrast = SliderSpinBox("Contrast:", 0.1, 3.0, 0.1, 1.0)
        self.spin_bright = SliderSpinBox("Brightness:", -1.0, 1.0, 0.1, 0.5)
        
        self.spin_tone_w.setEnabled(False)
        self.spin_contrast.setEnabled(False)
        self.spin_bright.setEnabled(False)
        
        c4_lay.addWidget(self.spin_tone_w); c4_lay.addWidget(self.spin_contrast); c4_lay.addWidget(self.spin_bright)
        col2_lay.addWidget(group_tone)
        col2_lay.addStretch()

        param_cols.addWidget(col1_widget)
        param_cols.addWidget(col2_widget)
        left_layout.addLayout(param_cols)

        # 3. Execution Logs 
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
        
        w2_lay = QVBoxLayout(); w2_lay.addWidget(QLabel("2. Line Art (Extracted & Inverted)"))
        self.lbl_img2 = AspectRatioLabel(); self.lbl_img2.setFrameShape(QFrame.Box)
        w2_lay.addWidget(self.lbl_img2)

        w3_lay = QVBoxLayout(); w3_lay.addWidget(QLabel("3. Thinned (Skeleton & Inverted)"))
        self.lbl_img3 = AspectRatioLabel(); self.lbl_img3.setFrameShape(QFrame.Box)
        w3_lay.addWidget(self.lbl_img3)

        w4_lay = QVBoxLayout(); w4_lay.addWidget(QLabel("4. Grid Image (Visual Analysis)"))
        self.lbl_img4 = AspectRatioLabel(); self.lbl_img4.setFrameShape(QFrame.Box)
        w4_lay.addWidget(self.lbl_img4)

        w5_lay = QVBoxLayout(); w5_lay.addWidget(QLabel("5. Output (Selectable Text)"))
        self.text_out = QTextEdit(); self.text_out.setLineWrapMode(QTextEdit.NoWrap)
        font = QFont("Saitamaar", 10); font.setStyleHint(QFont.Monospace); self.text_out.setFont(font)
        w5_lay.addWidget(self.text_out)
        
        w6_lay = QVBoxLayout(); w6_lay.addWidget(QLabel("6. Rendered AA Image"))
        self.lbl_img5 = AspectRatioLabel(); self.lbl_img5.setFrameShape(QFrame.Box)
        w6_lay.addWidget(self.lbl_img5)

        grid_imgs.addLayout(w1_lay, 0, 0)
        grid_imgs.addLayout(w2_lay, 0, 1)
        grid_imgs.addLayout(w3_lay, 0, 2)
        grid_imgs.addLayout(w4_lay, 1, 0)
        grid_imgs.addLayout(w5_lay, 1, 1)
        grid_imgs.addLayout(w6_lay, 1, 2)

        for i in range(3): grid_imgs.setColumnStretch(i, 1)
        for i in range(2): grid_imgs.setRowStretch(i, 1)
        
        main_layout.addLayout(grid_imgs, stretch=1)

    def log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    # --- Parameter Toggle Logic ---
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
        
        # ROI 파라미터 전부 잠금/해제
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
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
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
            self.log("❌ 작업 실패. 파라미터나 로드된 파일을 다시 확인해주세요.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SJISApp()
    ex.show()
    sys.exit(app.exec_())
