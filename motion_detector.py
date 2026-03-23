"""
Quad Fusion Motion Detector
===========================
Определение движущихся объектов (лодок) по 4 независимым сигналам:
  1. Wake Association  — наличие кильватерного следа рядом с лодкой (YOLOv9)
  2. Optical Flow      — оптический поток внутри bbox vs фон (Farneback)
  3. Trajectory Shift  — смещение центра bbox между кадрами (IoU-трекинг)
  4. Water Trail       — детекция белой пены/волн за лодкой по текстуре (CV)

Сигнал 4 критичен когда дрон следит за лодкой (flow≈0, trajectory≈0).

Финальный motion score = w1*wake + w2*flow + w3*trajectory + w4*trail  ∈ [0, 1]

Использование:
  python motion_detector.py --video test.mp4 --weights train10/weights/best.pt
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# ─────────────────────────── Конфигурация ───────────────────────────────
CLASS_BOAT = 0
CLASS_WAKE = 1

# Веса фузии четырёх сигналов (в сумме = 1.0)
W_WAKE       = 0.30   # Wake-детекция модели
W_FLOW       = 0.20   # Optical flow
W_TRAJECTORY = 0.15   # Displacement по трекингу
W_TRAIL      = 0.35   # Текстура водного следа (самый надёжный при слежении дроном)

# Пороги
MOTION_THRESHOLD      = 0.35   # score выше — объект движется
WAKE_ASSOC_IOU_THRESH = 0.0    # минимальный IoU для ассоциации wake↔boat
WAKE_ASSOC_DIST_THRESH = 150   # максимальное расстояние центров wake↔boat (px)
FLOW_MAGNITUDE_THRESH = 2.0    # средний поток ниже — считается нулевым
TRACK_IOU_THRESH      = 0.25   # IoU для матчинга треков между кадрами
TRACK_HISTORY         = 15     # сколько кадров хранить историю трека

# Фильтр слишком больших bbox (доля от площади кадра)
MAX_BBOX_AREA_RATIO   = 0.10   # bbox > 10% кадра — игнорируем
# Фильтр bbox, касающихся края кадра (скорее всего ложные — причал, берег)
EDGE_MARGIN_PX        = 5      # если bbox касается края кадра ближе чем N px — отбрасываем

# Локальная компенсация камеры: отступ вокруг bbox для вычисления фона
FLOW_LOCAL_MARGIN     = 40     # пикселей вокруг bbox для оценки фонового потока

# Детектор водного следа (белая пена/волны за лодкой)
TRAIL_SEARCH_MULT     = 2.5    # зона поиска = bbox * MULT вокруг лодки
TRAIL_BRIGHT_THRESH   = 180    # порог яркости для белой пены (0..255)
TRAIL_MIN_RATIO       = 0.02   # минимальная доля ярких пикселей для score > 0
TRAIL_SAT_RATIO       = 0.15   # доля ярких пикселей при которой score = 1.0

# Гистерезис: сглаживание motion state по треку (без мигания)
EMA_ALPHA             = 0.3    # коэффициент EMA (0..1, меньше = плавнее)
HYSTERESIS_ON         = 0.30   # порог переключения STATIC → MOVING
HYSTERESIS_OFF        = 0.15   # порог переключения MOVING → STATIC
MIN_STATIC_FRAMES     = 8      # сколько кадров подряд score < OFF чтобы стать STATIC

# Визуализация
COLOR_MOVING    = (0, 0, 255)    # красный
COLOR_STATIC    = (0, 255, 0)    # зелёный
COLOR_WAKE      = (255, 165, 0)  # оранжевый
COLOR_FLOW_VIS  = (255, 255, 0)  # жёлтый — стрелки потока


# ─────────────────────────── Утилиты ────────────────────────────────────
def bbox_iou(b1, b2):
    """IoU между двумя bbox [x1,y1,x2,y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bbox_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def center_dist(b1, b2):
    c1, c2 = bbox_center(b1), bbox_center(b2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ─────────────────── 1. Wake Association Score ──────────────────────────
def compute_wake_scores(boats, wakes):
    """
    Для каждой лодки: есть ли рядом wake?
    Возвращает dict {boat_idx: score 0..1}
    Логика: IoU > 0 ИЛИ расстояние центров < порога → score = 1.0
            Иначе score экспоненциально убывает с расстоянием.
    """
    scores = {}
    for i, boat in enumerate(boats):
        best = 0.0
        for wake in wakes:
            iou = bbox_iou(boat, wake)
            dist = center_dist(boat, wake)
            if iou > WAKE_ASSOC_IOU_THRESH:
                best = 1.0
                break
            if dist < WAKE_ASSOC_DIST_THRESH:
                # Экспоненциальный спад: чем ближе wake, тем выше score
                s = np.exp(-dist / (WAKE_ASSOC_DIST_THRESH / 3))
                best = max(best, s)
        scores[i] = best
    return scores


# ─────────────────── 2. Optical Flow Score ──────────────────────────────
def compute_flow_scores(prev_gray, curr_gray, boats):
    """
    Dense Farneback optical flow внутри каждого boat bbox.
    Локальная компенсация камеры: вычисляем медиану потока в кольце ВОКРУГ bbox
    (фон) и вычитаем из потока внутри bbox. Так движение камеры не влияет.
    Возвращает dict {boat_idx: score 0..1}, а также flow_map для визуализации.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    fh, fw = flow.shape[:2]
    scores = {}

    for i, b in enumerate(boats):
        x1, y1, x2, y2 = map(int, b)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            scores[i] = 0.0
            continue

        # Внутренний поток (bbox объекта)
        roi_flow = flow[y1:y2, x1:x2]

        # Внешнее кольцо (фон вокруг bbox) для оценки движения камеры
        m = FLOW_LOCAL_MARGIN
        ox1, oy1 = max(0, x1 - m), max(0, y1 - m)
        ox2, oy2 = min(fw, x2 + m), min(fh, y2 + m)

        # Маска: кольцо = внешний прямоугольник минус внутренний bbox
        ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
        inner_y1, inner_x1 = y1 - oy1, x1 - ox1
        inner_y2, inner_x2 = y2 - oy1, x2 - ox1
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        outer_flow = flow[oy1:oy2, ox1:ox2]
        if ring_mask.any():
            bg_dx = np.median(outer_flow[ring_mask, 0])
            bg_dy = np.median(outer_flow[ring_mask, 1])
        else:
            bg_dx = np.median(flow[..., 0])
            bg_dy = np.median(flow[..., 1])

        # Компенсированный поток внутри bbox
        comp_dx = roi_flow[..., 0] - bg_dx
        comp_dy = roi_flow[..., 1] - bg_dy
        comp_mag = np.sqrt(comp_dx ** 2 + comp_dy ** 2)

        # 75-й перцентиль — устойчивее к шуму
        avg_mag = np.percentile(comp_mag, 75) if comp_mag.size > 0 else 0.0

        # Нормализация
        if avg_mag < FLOW_MAGNITUDE_THRESH:
            scores[i] = 0.0
        else:
            scores[i] = min(1.0, (avg_mag - FLOW_MAGNITUDE_THRESH) / 8.0)

    return scores, flow


# ─────────────────── 3. Water Trail Texture Score ───────────────────────
def compute_trail_scores(frame_gray, frame_bgr, boats):
    """
    Детекция водного следа (белая пена, волны) вокруг лодки по текстуре.
    Ищем яркие, низко-насыщенные пиксели в расширенной зоне вокруг bbox,
    исключая сам bbox лодки. Белая пена на воде = высокая яркость + низкая
    насыщенность (в отличие от зелёной воды).
    Работает даже когда дрон следит за лодкой (flow≈0, trajectory≈0).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    fh, fw = frame_gray.shape[:2]
    scores = {}

    for i, b in enumerate(boats):
        x1, y1, x2, y2 = map(int, b)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            scores[i] = 0.0
            continue

        # Расширенная зона поиска вокруг лодки
        expand_w = int(bw * TRAIL_SEARCH_MULT / 2)
        expand_h = int(bh * TRAIL_SEARCH_MULT / 2)
        sx1 = max(0, x1 - expand_w)
        sy1 = max(0, y1 - expand_h)
        sx2 = min(fw, x2 + expand_w)
        sy2 = min(fh, y2 + expand_h)

        # Маска: расширенная зона МИНУС bbox лодки
        search_h = sy2 - sy1
        search_w = sx2 - sx1
        if search_h <= 0 or search_w <= 0:
            scores[i] = 0.0
            continue

        mask = np.ones((search_h, search_w), dtype=bool)
        inner_y1, inner_x1 = y1 - sy1, x1 - sx1
        inner_y2, inner_x2 = y2 - sy1, x2 - sx1
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        roi_hsv = hsv[sy1:sy2, sx1:sx2]
        roi_v = roi_hsv[..., 2]   # яркость (Value)
        roi_s = roi_hsv[..., 1]   # насыщенность (Saturation)

        # Белая пена: высокая яркость + низкая насыщенность
        bright_mask = (roi_v > TRAIL_BRIGHT_THRESH) & (roi_s < 80) & mask

        total_pixels = mask.sum()
        if total_pixels == 0:
            scores[i] = 0.0
            continue

        bright_ratio = bright_mask.sum() / total_pixels

        # Нормализация: линейная между MIN_RATIO и SAT_RATIO
        if bright_ratio < TRAIL_MIN_RATIO:
            scores[i] = 0.0
        elif bright_ratio >= TRAIL_SAT_RATIO:
            scores[i] = 1.0
        else:
            scores[i] = (bright_ratio - TRAIL_MIN_RATIO) / (TRAIL_SAT_RATIO - TRAIL_MIN_RATIO)

    return scores


# ─────────────────── 4. Trajectory Displacement Score ───────────────────
class SimpleTracker:
    """
    Простой IoU-based трекер для отслеживания лодок между кадрами.
    Хранит историю центров для вычисления displacement.
    """
    def __init__(self):
        self.tracks = {}       # track_id -> {'bbox', 'history', 'ema_score', 'is_moving', 'static_count'}
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2]
        Возвращает dict {det_idx: track_id}
        """
        if not self.tracks:
            mapping = {}
            for i, det in enumerate(detections):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det,
                    'history': [bbox_center(det)],
                    'age': 0,
                    'ema_score': 0.0,
                    'is_moving': False,
                    'static_count': 0,
                }
                mapping[i] = tid
            return mapping

        # Матрица IoU: existing tracks × new detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = bbox_iou(self.tracks[tid]['bbox'], det)

        mapping = {}
        used_tracks = set()
        used_dets = set()

        # Жадный матчинг по убыванию IoU
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[idx]
            if best_iou < TRACK_IOU_THRESH:
                break
            ti, di = idx
            tid = track_ids[ti]
            self.tracks[tid]['bbox'] = detections[di]
            self.tracks[tid]['history'].append(bbox_center(detections[di]))
            if len(self.tracks[tid]['history']) > TRACK_HISTORY:
                self.tracks[tid]['history'] = self.tracks[tid]['history'][-TRACK_HISTORY:]
            self.tracks[tid]['age'] = 0
            mapping[di] = tid
            used_tracks.add(ti)
            used_dets.add(di)
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        # Новые треки для не-сматченных детекций
        for di, det in enumerate(detections):
            if di not in used_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det,
                    'history': [bbox_center(det)],
                    'age': 0,
                    'ema_score': 0.0,
                    'is_moving': False,
                    'static_count': 0,
                }
                mapping[di] = tid

        # Старение не-сматченных треков
        for ti, tid in enumerate(track_ids):
            if ti not in used_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > 5:
                    del self.tracks[tid]

        return mapping

    def update_motion_state(self, track_id, raw_score):
        """
        Обновляет сглаженный motion score (EMA) и состояние с гистерезисом.
        Возвращает (smoothed_score, is_moving).

        Логика:
        - EMA сглаживает raw_score по времени
        - STATIC → MOVING: ema > HYSTERESIS_ON
        - MOVING → STATIC: ema < HYSTERESIS_OFF в течение MIN_STATIC_FRAMES подряд
        """
        track = self.tracks.get(track_id)
        if track is None:
            return raw_score, raw_score >= MOTION_THRESHOLD

        # EMA обновление
        prev_ema = track['ema_score']
        ema = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * prev_ema
        track['ema_score'] = ema

        was_moving = track['is_moving']

        if not was_moving:
            # STATIC → MOVING: порог ON
            if ema >= HYSTERESIS_ON:
                track['is_moving'] = True
                track['static_count'] = 0
        else:
            # MOVING → STATIC: нужно MIN_STATIC_FRAMES подряд ниже порога OFF
            if ema < HYSTERESIS_OFF:
                track['static_count'] += 1
                if track['static_count'] >= MIN_STATIC_FRAMES:
                    track['is_moving'] = False
            else:
                track['static_count'] = 0

        return ema, track['is_moving']

    def get_displacement(self, track_id, n_frames=5):
        """
        Суммарное смещение центра за последние n_frames.
        """
        hist = self.tracks.get(track_id, {}).get('history', [])
        if len(hist) < 2:
            return 0.0
        recent = hist[-min(n_frames, len(hist)):]
        total = 0.0
        for j in range(1, len(recent)):
            dx = recent[j][0] - recent[j - 1][0]
            dy = recent[j][1] - recent[j - 1][1]
            total += np.hypot(dx, dy)
        return total


def compute_trajectory_scores(tracker, mapping, frame_diag):
    """
    Нормализованный displacement score для каждой детекции.
    """
    scores = {}
    for det_idx, tid in mapping.items():
        disp = tracker.get_displacement(tid, n_frames=5)
        # Нормализуем по диагонали кадра
        norm_disp = disp / frame_diag if frame_diag > 0 else 0
        # Sigmoid-масштабирование: движение > 1% диагонали за 5 кадров = сильный сигнал
        scores[det_idx] = min(1.0, norm_disp / 0.03)
    return scores


# ─────────────────── Фузия четырёх сигналов ──────────────────────────────
def fuse_scores(wake_s, flow_s, traj_s, trail_s):
    """
    Взвешенная фузия 4 сигналов.
    Буст если wake или trail высоки (оба — сильные индикаторы движения).
    """
    raw = W_WAKE * wake_s + W_FLOW * flow_s + W_TRAJECTORY * traj_s + W_TRAIL * trail_s
    # Бустим если wake или trail высокие
    if wake_s > 0.7:
        raw = max(raw, 0.80)
    if trail_s > 0.5:
        raw = max(raw, 0.65)
    if wake_s > 0.7 and trail_s > 0.3:
        raw = max(raw, 0.90)
    return min(1.0, raw)


# ─────────────────── Визуализация ───────────────────────────────────────
def draw_flow_arrows(frame, flow, bbox, step=16):
    """Рисует стрелки оптического потока внутри bbox."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    for y in range(y1, y2, step):
        for x in range(x1, x2, step):
            fx, fy = flow[y, x]
            mag = np.hypot(fx, fy)
            if mag > 1.5:
                cv2.arrowedLine(
                    frame, (x, y), (int(x + fx * 2), int(y + fy * 2)),
                    COLOR_FLOW_VIS, 1, tipLength=0.3
                )


def draw_results(frame, boats, wakes, wake_scores, flow_scores, traj_scores,
                 trail_scores, motion_scores, motion_states, flow_map, mapping, tracker):
    """Рисует все аннотации на кадре."""
    overlay = frame.copy()

    # Wake boxes
    for w in wakes:
        x1, y1, x2, y2 = map(int, w)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_WAKE, 2)
        cv2.putText(overlay, "wake", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WAKE, 1)

    # Boat boxes + motion
    for i, boat in enumerate(boats):
        x1, y1, x2, y2 = map(int, boat)
        ms = motion_scores.get(i, 0)
        is_moving = motion_states.get(i, False)
        color = COLOR_MOVING if is_moving else COLOR_STATIC
        label = "MOVING" if is_moving else "STATIC"

        # Бокс
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Шкала motion score — заливка-полоска
        bar_w = x2 - x1
        bar_h = 6
        bar_y = y1 - 28
        cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * ms)
        cv2.rectangle(overlay, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), color, -1)

        # Текст: 4 сигнала
        ws = wake_scores.get(i, 0)
        fs = flow_scores.get(i, 0)
        ts = traj_scores.get(i, 0)
        trs = trail_scores.get(i, 0)
        txt = f"{label} {ms:.2f} [W:{ws:.1f} F:{fs:.1f} T:{ts:.1f} Tr:{trs:.1f}]"
        cv2.putText(overlay, txt, (x1, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Стрелки потока внутри bbox (только для движущихся)
        if is_moving and flow_map is not None:
            draw_flow_arrows(overlay, flow_map, boat, step=20)

        # Траектория
        if i in mapping:
            tid = mapping[i]
            hist = tracker.tracks.get(tid, {}).get('history', [])
            if len(hist) > 1:
                pts = [tuple(map(int, p)) for p in hist]
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(overlay, pts[j - 1], pts[j], c, 2)

    # Инфо-панель
    cv2.putText(overlay, "Triple Fusion Motion Detector", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Weights: Wake={W_WAKE} Flow={W_FLOW} Traj={W_TRAJECTORY}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    n_moving = sum(1 for s in motion_states.values() if s)
    n_total = len(boats)
    cv2.putText(overlay, f"Boats: {n_total}  Moving: {n_moving}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


# ─────────────────── Главный пайплайн ───────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Triple Fusion Motion Detector")
    parser.add_argument("--video", type=str, required=True, help="Путь к видео")
    parser.add_argument("--weights", type=str, required=True, help="Путь к весам YOLOv9")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold для YOLO")
    parser.add_argument("--output", type=str, default=None, help="Путь к выходному видео")
    parser.add_argument("--show", action="store_true", help="Показывать в реальном времени")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входа YOLO")
    args = parser.parse_args()

    # Авто-имя выхода
    if args.output is None:
        p = Path(args.video)
        args.output = str(p.parent / f"{p.stem}_motion{p.suffix}")

    # Загрузка модели
    print(f"[*] Загрузка модели: {args.weights}")
    model = YOLO(args.weights)

    # Открытие видео
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[!] Не удалось открыть видео: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diag = np.hypot(w, h)
    frame_area = w * h

    print(f"[*] Видео: {w}x{h} @ {fps:.1f}fps, {total} кадров")
    print(f"[*] Выход: {args.output}")

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    tracker = SimpleTracker()
    prev_gray = None
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── YOLOv9 inference ──
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = results.boxes

        boats = []
        wakes = []
        boat_confs = []
        for j in range(len(boxes)):
            cls = int(boxes.cls[j])
            xyxy = boxes.xyxy[j].cpu().numpy().tolist()
            conf = float(boxes.conf[j])
            # Фильтр: пропускаем слишком большие bbox
            bbox_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if bbox_area / frame_area > MAX_BBOX_AREA_RATIO:
                continue
            if cls == CLASS_BOAT:
                # Фильтр края — только для boat (причалы, берег)
                if (xyxy[0] < EDGE_MARGIN_PX or xyxy[1] < EDGE_MARGIN_PX or
                        xyxy[2] > w - EDGE_MARGIN_PX or xyxy[3] > h - EDGE_MARGIN_PX):
                    continue
                boats.append(xyxy)
                boat_confs.append(conf)
            elif cls == CLASS_WAKE:
                # Wake НЕ фильтруем по краю — может быть частично за кадром
                wakes.append(xyxy)

        # ── Сигнал 1: Wake Association ──
        wake_scores = compute_wake_scores(boats, wakes)

        # ── Сигнал 2: Optical Flow ──
        flow_scores = {}
        flow_map = None
        if prev_gray is not None and len(boats) > 0:
            flow_scores, flow_map = compute_flow_scores(prev_gray, curr_gray, boats)
        else:
            flow_scores = {i: 0.0 for i in range(len(boats))}

        # ── Сигнал 3: Water Trail Texture ──
        trail_scores = compute_trail_scores(curr_gray, frame, boats)

        # ── Сигнал 4: Trajectory Displacement ──
        mapping = tracker.update(boats)
        traj_scores = compute_trajectory_scores(tracker, mapping, frame_diag)

        # ── Фузия 4 сигналов + гистерезис ──
        motion_scores = {}   # smoothed EMA score
        motion_states = {}   # bool: is_moving (with hysteresis)
        for i in range(len(boats)):
            ws = wake_scores.get(i, 0)
            fs = flow_scores.get(i, 0)
            ts = traj_scores.get(i, 0)
            trs = trail_scores.get(i, 0)
            raw = fuse_scores(ws, fs, ts, trs)
            if i in mapping:
                ema, is_mov = tracker.update_motion_state(mapping[i], raw)
                motion_scores[i] = ema
                motion_states[i] = is_mov
            else:
                motion_scores[i] = raw
                motion_states[i] = raw >= MOTION_THRESHOLD

        # ── Визуализация ──
        vis = draw_results(
            frame, boats, wakes,
            wake_scores, flow_scores, traj_scores, trail_scores,
            motion_scores, motion_states, flow_map, mapping, tracker
        )
        writer.write(vis)

        if args.show:
            cv2.imshow("Motion Detector", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_num % 50 == 0:
            n_mov = sum(1 for v in motion_states.values() if v)
            print(f"  [{frame_num}/{total}] boats={len(boats)} wakes={len(wakes)} moving={n_mov}")

        prev_gray = curr_gray

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[✓] Готово! Результат: {args.output}")


if __name__ == "__main__":
    main()
