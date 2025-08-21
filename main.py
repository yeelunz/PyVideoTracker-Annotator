import sys, os, json
from dataclasses import dataclass
from typing import List, Dict, Optional
import cv2, numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QEvent
from PySide6.QtGui import QAction, QPainter, QPen, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSlider, QLineEdit, QMessageBox, QInputDialog, QProgressDialog, QAbstractItemView
)

# ===================== Data Structures =====================
@dataclass
class Box:
    track_id: int
    category_id: int
    bbox: List[float]  # x,y,w,h
    def to_annotation(self, ann_id: int, image_id: int) -> Dict:
        x,y,w,h = self.bbox
        return {"id": ann_id,"image_id": image_id,"category_id": self.category_id,"track_id": self.track_id,
                "bbox": [round(float(x),2), round(float(y),2), round(float(w),2), round(float(h),2)],
                "area": round(float(w*h),2),"iscrowd":0}

# ===================== Project Core =====================
class AnnotationProject:
    def __init__(self, video_path: str, categories: List[str]):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.categories = categories
        self.category_to_id = {n: i + 1 for i, n in enumerate(categories)}
        self.id_to_category = {i + 1: n for i, n in enumerate(categories)}
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError('無法開啟影片')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.annotations_by_frame = {}
        self.next_track_id = 1
        self.tmp_path = os.path.splitext(video_path)[0] + '.json.tmp'
        self.undo_stack = []
        self._load_tmp()

    # --- 平滑工具：對一段連續幀上的同一 track 的 bbox 做移動平均 ---
    def smooth_track_mavg(self, track_id: int, frame_indices: List[int], kernel_size: int = 5, anchor_first: bool = True, modify_first: bool = False):
        """對指定 track 在給定幀序列上做移動平均平滑。
        - track_id: 要平滑的追蹤 ID
        - frame_indices: 順序化的幀索引列表（例如 [f0, f1, ..., fn]）
        - kernel_size: 奇數，移動平均視窗大小
        - anchor_first: 若為 True，第一個有效幀的 bbox 會作為錨點參與平滑計算，但可選擇是否回寫修改
        - modify_first: 若為 False，第一個有效幀不會被實際修改（僅用於計算）
        """
        if not frame_indices:
            return
        try:
            k = max(1, int(kernel_size))
        except Exception:
            k = 5
        if k % 2 == 0:
            k += 1
        # 收集存在的框
        seq = []  # (fi, Box)
        for fi in frame_indices:
            arr = self.annotations_by_frame.get(fi, [])
            for b in arr:
                if b.track_id == track_id:
                    seq.append((fi, b))
                    break
        if len(seq) <= 1:
            return
        xs = [float(b.bbox[0]) for _, b in seq]
        ys = [float(b.bbox[1]) for _, b in seq]
        ws = [float(b.bbox[2]) for _, b in seq]
        hs = [float(b.bbox[3]) for _, b in seq]

        def _mavg(arr, kk):
            if kk <= 1 or len(arr) <= 1:
                return arr[:]
            pad = kk // 2
            arr_np = np.asarray(arr, dtype=float)
            arr_pad = np.pad(arr_np, (pad, pad), mode='edge')
            kernel = np.ones(kk, dtype=float) / float(kk)
            out = np.convolve(arr_pad, kernel, mode='valid')
            return out.tolist()

        sx = _mavg(xs, k)
        sy = _mavg(ys, k)
        sw = _mavg(ws, k)
        sh = _mavg(hs, k)

        # 邊界保護並寫回
        W, H = self.width, self.height
        for i, (fi, b) in enumerate(seq):
            if i == 0 and anchor_first and not modify_first:
                continue  # 保留第一幀原樣
            nx, ny, nw, nh = float(sx[i]), float(sy[i]), float(sw[i]), float(sh[i])
            # clamp
            nx = max(0.0, min(W - 1.0, nx))
            ny = max(0.0, min(H - 1.0, ny))
            nw = max(1.0, min(W - nx, nw))
            nh = max(1.0, min(H - ny, nh))
            new_bbox = [nx, ny, nw, nh]
            # 使用 update_box 以保留 undo 歷史
            self.update_box(fi, track_id, new_bbox)

    def _load_tmp(self):
        # 優先從最終輸出的 COCO JSON 載入進度
        coco_path = os.path.splitext(self.video_path)[0] + '.json'
        if os.path.exists(coco_path):
            try:
                with open(coco_path,'r',encoding='utf-8') as f:
                    data = json.load(f)
                # 建立 image_id -> frame_index 對照
                img_map = {}
                for img in data.get('images', []):
                    img_map[img['id']] = img.get('frame_index', 0)
                frames: Dict[int, List[Box]] = {}
                max_tid = 0
                for ann in data.get('annotations', []):
                    image_id = ann.get('image_id')
                    frame_index = img_map.get(image_id, 0)
                    track_id = ann.get('track_id', 0)
                    category_id = ann.get('category_id', 1)
                    bbox = ann.get('bbox', [0,0,0,0])
                    frames.setdefault(frame_index, []).append(Box(track_id=track_id, category_id=category_id, bbox=bbox))
                    if track_id > max_tid:
                        max_tid = track_id
                self.annotations_by_frame = frames
                self.next_track_id = max(1, max_tid + 1)
                return
            except Exception as e:
                print('載入 COCO JSON 失敗，嘗試載入暫存檔:', e)
        # 回退：嘗試載入舊版暫存檔（相容舊專案）
        if os.path.exists(self.tmp_path):
            try:
                with open(self.tmp_path,'r',encoding='utf-8') as f:
                    data = json.load(f)
                self.next_track_id = data.get('next_track_id',1)
                for k, arr in data.get('frames',{}).items():
                    fi = int(k)
                    self.annotations_by_frame[fi] = [Box(track_id=b['track_id'], category_id=b['category_id'], bbox=b['bbox']) for b in arr]
            except Exception as e:
                print('載入暫存檔失敗:', e)

    def save_tmp(self):
        # 立即寫入 COCO-VID JSON（取代暫存）
        try:
            self.export_coco_vid()
        except Exception as e:
            print('寫入 JSON 失敗:', e)

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def clone_prev_if_needed(self, frame_index: int):
        # 需求更新：完全取消自動複製標註，改為空操作
        return

    def new_box(self, frame_index: int, category_id: int, bbox: List[float]) -> int:
        if frame_index not in self.annotations_by_frame:
            self.annotations_by_frame[frame_index] = []
        tid = self.next_track_id; self.next_track_id += 1
        box = Box(track_id=tid, category_id=category_id, bbox=bbox)
        self.annotations_by_frame[frame_index].append(box)
        self.undo_stack.append({'type':'add','frame':frame_index,'track_id':tid})
        self.save_tmp(); return tid

    def add_box_with_track_id(self, frame_index: int, track_id: int, category_id: int, bbox: List[float]):
        """在指定幀加入指定 track_id 的框；若該幀已有同 track，則改為更新其 bbox。
        用於插值/延伸保持同一追蹤 ID。
        """
        arr = self.annotations_by_frame.setdefault(frame_index, [])
        # 若已存在同 track，改為更新
        for b in arr:
            if b.track_id == track_id:
                old = b.bbox.copy()
                if old != bbox:
                    b.bbox = bbox
                    self.undo_stack.append({'type':'modify','frame':frame_index,'track_id':track_id,'old_bbox':old,'new_bbox':bbox})
                self.save_tmp(); return track_id
        # 否則新增
        arr.append(Box(track_id=track_id, category_id=category_id, bbox=bbox))
        self.undo_stack.append({'type':'add','frame':frame_index,'track_id':track_id})
        if track_id >= self.next_track_id:
            self.next_track_id = track_id + 1
        self.save_tmp(); return track_id

    def delete_box(self, frame_index: int, track_id: int, push_undo: bool=True):
        arr = self.annotations_by_frame.get(frame_index)
        if not arr: return
        for b in arr:
            if b.track_id == track_id and push_undo:
                self.undo_stack.append({'type':'delete','frame':frame_index,'box':{'track_id':b.track_id,'category_id':b.category_id,'bbox':b.bbox.copy()}})
                break
        self.annotations_by_frame[frame_index] = [b for b in arr if b.track_id != track_id]
        self.save_tmp()

    def update_box(self, frame_index: int, track_id: int, new_bbox: List[float]):
        arr = self.annotations_by_frame.get(frame_index, [])
        for b in arr:
            if b.track_id == track_id:
                old = b.bbox.copy()
                if old != new_bbox:
                    b.bbox = new_bbox
                    self.undo_stack.append({'type':'modify','frame':frame_index,'track_id':track_id,'old_bbox':old,'new_bbox':new_bbox})
                break
        self.save_tmp()

    def undo(self):
        if not self.undo_stack:
            return
        act = self.undo_stack.pop()
        t = act['type']
        if t == 'add':
            self.delete_box(act['frame'], act['track_id'], push_undo=False)
        elif t == 'delete':
            data = act['box']
            self.annotations_by_frame.setdefault(act['frame'], []).append(Box(track_id=data['track_id'], category_id=data['category_id'], bbox=data['bbox']))
            self.save_tmp()
        elif t == 'modify':
            arr = self.annotations_by_frame.get(act['frame'], [])
            for b in arr:
                if b.track_id == act['track_id']:
                    b.bbox = act['old_bbox']
                    break
            self.save_tmp()

    def export_coco_vid(self):
        images, annotations = [], []
        ann_id, image_id = 1, 1
        for fi in sorted(self.annotations_by_frame.keys()):
            boxes = self.annotations_by_frame[fi]
            if not boxes: continue
            images.append({'id':image_id,'video_id':1,'frame_index':fi,'file_name':f"{os.path.splitext(self.video_name)[0]}/{fi+1:08d}.jpg",'height':self.height,'width':self.width})
            for b in boxes:
                annotations.append(b.to_annotation(ann_id, image_id)); ann_id += 1
            image_id += 1
        categories = [{'id':cid,'name':name} for cid,name in self.id_to_category.items()]
        data = {'info':{'description':'PyVideoTracker-Annotator export'},'videos':[{'id':1,'name':self.video_name}], 'images':images,'annotations':annotations,'categories':categories}
        out_path = os.path.splitext(self.video_path)[0] + '.json'
        with open(out_path,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False)
        return out_path

# ===================== Video Canvas =====================
class VideoCanvas(QWidget):
    HANDLE_SIZE = 8
    def __init__(self, project: AnnotationProject, parent=None):
        super().__init__(parent)
        self.project = project
        self.frame: Optional[np.ndarray] = None
        self.frame_index = 0
        self.boxes: List[Box] = []
        self.current_category_id = 1
        self.selected_track_id: Optional[int] = None
        # drawing/edit state
        self.is_drawing = False
        self.start_point: Optional[QPointF] = None
        self.current_rect: Optional[QRectF] = None
        self.dragging = False
        self.resizing = False
        self.resize_handle: Optional[str] = None
        self.drag_offset = QPointF(0,0)
        self.original_bbox: Optional[List[float]] = None
        self.setMouseTracking(True)
    # 關鍵：允許接收鍵盤事件（Delete）
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ghost_boxes: List[Box] = []  # 新增：上一幀提示框 (僅顯示，不可互動)

    # transform helpers
    def _get_transform(self):
        if self.frame is None:
            return 1,0,0
        h,w,_ = self.frame.shape
        ww, wh = self.width(), self.height()
        scale = min(ww/w, wh/h)
        sw, sh = int(w*scale), int(h*scale)
        xo, yo = (ww - sw)//2, (wh - sh)//2
        return scale, xo, yo
    def _frame_to_widget_rect(self, bbox):
        s,xo,yo = self._get_transform(); x,y,w,h = bbox
        return QRectF(xo + x*s, yo + y*s, w*s, h*s)
    def _widget_to_frame(self, rect: QRectF):
        if self.frame is None: return [0,0,0,0]
        h_im,w_im,_ = self.frame.shape
        s,xo,yo = self._get_transform()
        x = (rect.left()-xo)/s; y = (rect.top()-yo)/s; w = rect.width()/s; h = rect.height()/s
        x = max(0,min(x,w_im-1)); y = max(0,min(y,h_im-1))
        w = max(1,min(w,w_im - x)); h = max(1,min(h,h_im - y))
        return [x,y,w,h]

    def load_frame(self, index: int):
        frame = self.project.get_frame(index)
        if frame is None: return
        self.frame_index = index
        # 取消自動複製：不再呼叫 clone_prev_if_needed
        self.boxes = self.project.annotations_by_frame.get(index, [])
        # 一律顯示前一幀為幽靈框（第一幀沒有）
        if index > 0:
            prev = self.project.annotations_by_frame.get(index-1, [])
            self.ghost_boxes = [Box(track_id=b.track_id, category_id=b.category_id, bbox=b.bbox.copy()) for b in prev]
        else:
            self.ghost_boxes = []
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update()

    def paintEvent(self, event):
        if self.frame is None: return
        p = QPainter(self)
        h,w,_ = self.frame.shape
        s,xo,yo = self._get_transform()
        qimg = self._np_to_qimage(self.frame)
        p.drawImage(QRectF(xo,yo,w*s,h*s), qimg)
        def draw_box(b:Box, color:QColor, thick=2):
            x,y,bw,bh = b.bbox
            p.setPen(QPen(color, thick))
            fill = QColor(color); fill.setAlpha(50 if b.track_id != self.selected_track_id else 80)
            p.setBrush(QBrush(fill))
            p.drawRect(xo + x*s, yo + y*s, bw*s, bh*s)
            p.setPen(QPen(QColor(255,255,255),1))
            p.drawText(xo + x*s + 2, yo + y*s + 14, str(b.track_id))
        # 先畫幽靈框 (虛線、淡色)
        if self.ghost_boxes:
            p.setBrush(Qt.BrushStyle.NoBrush)
            s,xo,yo = self._get_transform()
            for gb in self.ghost_boxes:
                x,y,bw,bh = gb.bbox
                color = QColor(255,255,0)
                color.setAlpha(120)
                pen = QPen(color, 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRect(xo + x*s, yo + y*s, bw*s, bh*s)
                p.setPen(QPen(color,1))
                p.drawText(xo + x*s + 2, yo + y*s + 12, str(gb.track_id))
        # 再畫實際框
        for b in self.boxes:
            draw_box(b, QColor(255,0,0) if b.track_id == self.selected_track_id else QColor(0,200,0))
        if self.is_drawing and self.current_rect is not None:
            p.setPen(QPen(QColor(255,255,0),2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.current_rect)
        if self.selected_track_id is not None:
            for b in self.boxes:
                if b.track_id == self.selected_track_id:
                    r = self._frame_to_widget_rect(b.bbox)
                    p.setBrush(QBrush(QColor(255,0,0)))
                    p.setPen(Qt.PenStyle.NoPen)
                    hs = self.HANDLE_SIZE
                    for px,py in [(r.left(),r.top()),(r.right(),r.top()),(r.left(),r.bottom()),(r.right(),r.bottom())]:
                        p.drawRect(px-hs/2, py-hs/2, hs, hs)
                    break

    def _np_to_qimage(self, arr):
        h,w,c = arr.shape; bpl = c*w
        from PySide6.QtGui import QImage
        return QImage(arr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()

    # hit tests
    def _hit_box(self,pos:QPointF)->Optional[int]:
        s,xo,yo = self._get_transform()
        for b in reversed(self.boxes):
            x,y,w,h = b.bbox
            rx,ry = xo + x*s, yo + y*s
            if rx <= pos.x() <= rx + w*s and ry <= pos.y() <= ry + h*s:
                return b.track_id
        return None
    def _hit_handle(self,pos:QPointF)->Optional[str]:
        if self.selected_track_id is None: return None
        for b in self.boxes:
            if b.track_id == self.selected_track_id:
                r = self._frame_to_widget_rect(b.bbox); hs = self.HANDLE_SIZE
                handles = {'tl':QPointF(r.left(),r.top()),'tr':QPointF(r.right(),r.top()),'bl':QPointF(r.left(),r.bottom()),'br':QPointF(r.right(),r.bottom())}
                for name,pt in handles.items():
                    if abs(pos.x()-pt.x())<=hs and abs(pos.y()-pt.y())<=hs:
                        return name
        return None

    def mousePressEvent(self,event):
        if event.button() != Qt.MouseButton.LeftButton or self.frame is None:
            return
        # 確保畫布取得鍵盤焦點，讓 Delete 能作用
        self.setFocus()
        pos = event.position()
        handle = self._hit_handle(pos)
        if handle:
            self.resizing = True
            self.resize_handle = handle
            for b in self.boxes:
                if b.track_id == self.selected_track_id:
                    self.original_bbox = b.bbox.copy()
                    break
            return
        tid = self._hit_box(pos)
        if tid is not None:
            self.selected_track_id = tid
            for b in self.boxes:
                if b.track_id == tid:
                    rect = self._frame_to_widget_rect(b.bbox)
                    self.dragging = True
                    self.original_bbox = b.bbox.copy()
                    self.drag_offset = QPointF(pos.x() - rect.left(), pos.y() - rect.top())
                    break
            self.parent().parent().update_box_list(select_track=tid)
            self.update()
            return
        # start drawing
        self.is_drawing = True
        self.start_point = pos
        self.current_rect = QRectF(pos, pos)
        self.selected_track_id = None
        self.dragging = False
        self.resizing = False
        self.update()

    def mouseMoveEvent(self,event):
        pos = event.position()
        if self.is_drawing and self.start_point:
            self.current_rect = QRectF(self.start_point, pos).normalized(); self.update(); return
        if self.dragging and self.selected_track_id and self.original_bbox is not None:
            s,xo,yo = self._get_transform(); nx = (pos.x()-self.drag_offset.x()-xo)/s; ny = (pos.y()-self.drag_offset.y()-yo)/s
            w0,h0 = self.original_bbox[2], self.original_bbox[3]
            self._apply_temp_bbox([nx,ny,w0,h0]); return
        if self.resizing and self.selected_track_id and self.original_bbox is not None:
            s,xo,yo = self._get_transform(); ox,oy,ow,oh = self.original_bbox
            px,py = (pos.x()-xo)/s,(pos.y()-yo)/s
            nx,ny,nw,nh = ox,oy,ow,oh
            if 't' in self.resize_handle: nh = (oy+oh)-py; ny = py
            if 'b' in self.resize_handle: nh = py - oy
            if 'l' in self.resize_handle: nw = (ox+ow)-px; nx = px
            if 'r' in self.resize_handle: nw = px - ox
            if nw < 1: nx = ox+ow-1; nw = 1
            if nh < 1: ny = oy+oh-1; nh = 1
            self._apply_temp_bbox([nx,ny,nw,nh]); return
        self.setCursor(Qt.CursorShape.SizeAllCursor if self._hit_handle(pos) else Qt.CursorShape.ArrowCursor)

    def _apply_temp_bbox(self,new_bbox):
        for b in self.boxes:
            if b.track_id == self.selected_track_id:
                b.bbox = new_bbox; break
        self.update()

    def mouseReleaseEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton: return
        if self.is_drawing:
            self.is_drawing = False
            if self.current_rect and self.frame is not None:
                bbox = self._widget_to_frame(self.current_rect)
                if bbox[2] > 5 and bbox[3] > 5:
                    self.project.new_box(self.frame_index, self.current_category_id, bbox)
                    self.boxes = self.project.annotations_by_frame.get(self.frame_index, [])
            self.current_rect = None
            self.parent().parent().update_box_list(); self.update()
        elif self.dragging and self.selected_track_id and self.original_bbox is not None:
            for b in self.boxes:
                if b.track_id == self.selected_track_id and b.bbox != self.original_bbox:
                    self.project.update_box(self.frame_index, b.track_id, b.bbox)
                    break
            self.dragging = False; self.original_bbox = None
        elif self.resizing and self.selected_track_id and self.original_bbox is not None:
            for b in self.boxes:
                if b.track_id == self.selected_track_id and b.bbox != self.original_bbox:
                    self.project.update_box(self.frame_index, b.track_id, b.bbox)
                    break
            self.resizing = False; self.original_bbox = None

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Delete and self.selected_track_id is not None:
            self.project.delete_box(self.frame_index, self.selected_track_id)
            self.boxes = self.project.annotations_by_frame.get(self.frame_index, [])
            self.selected_track_id = None
            self.update(); self.parent().parent().update_box_list()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self,event):
        tid = self._hit_box(event.position())
        if tid is not None:
            self.selected_track_id = tid
            self.update(); self.parent().parent().update_box_list(select_track=tid)

# ===================== Main Window =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyVideoTracker-Annotator')
        self.resize(1200, 800)
        # state
        self.project = None
        self.root_dir = None
        self.categories = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False

        # layout root
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left area
        self.canvas_container = QVBoxLayout()
        self.video_canvas_placeholder = QLabel('請開啟資料夾')
        self.video_canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_widget = None
        self.frame_info_label = QLabel('Frame: -/-')
        self.canvas_container.addWidget(self.video_canvas_placeholder, 1)
        self.canvas_container.addWidget(self.frame_info_label)

        controls = QHBoxLayout()
        self.btn_prev = QPushButton('上一幀')
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_play = QPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton('下一幀')
        self.btn_next.clicked.connect(self.next_frame)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText('跳轉幀 (Enter)')
        self.jump_edit.returnPressed.connect(self.jump_to_frame)
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_next)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.jump_edit)
        self.canvas_container.addLayout(controls)
        root.addLayout(self.canvas_container, 3)

        # right area
        side = QVBoxLayout()
        side.addWidget(QLabel('影片'))
        self.list_videos = QListWidget()
        self.list_videos.itemClicked.connect(self.on_select_video)
        side.addWidget(self.list_videos, 2)

        side.addWidget(QLabel('類別'))
        self.list_categories = QListWidget()
        self.list_categories.itemClicked.connect(self.on_select_category)
        side.addWidget(self.list_categories, 1)

        side.addWidget(QLabel('本幀標註'))
        self.list_boxes = QListWidget()
        self.list_boxes.itemClicked.connect(self.on_select_box)
        side.addWidget(self.list_boxes, 2)

        # 插值按鈕
        self.btn_interpolate = QPushButton('插值')
        self.btn_interpolate.setToolTip('以本幀唯一標註使用 CSRT 追蹤器往後插值/延伸 N 幀（保持同一 track_id），並自動以視窗=5做平滑')
        self.btn_interpolate.clicked.connect(self.do_interpolate)
        side.addWidget(self.btn_interpolate)

        # 已標註幀清單（支援多選）
        side.addWidget(QLabel('已標註幀'))
        self.list_annotated_frames = QListWidget()
        self.list_annotated_frames.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_annotated_frames.installEventFilter(self)
        self.list_annotated_frames.itemClicked.connect(self.on_select_annotated_frame)
        side.addWidget(self.list_annotated_frames, 2)

        root.addLayout(side, 1)

        # 功能表與快捷鍵
        self._build_menu()
        self._bind_shortcuts()

    # menu / shortcuts
    def _build_menu(self):
        m = self.menuBar().addMenu('檔案')
        act_open = QAction('開啟資料夾', self)
        act_open.triggered.connect(self.open_folder)
        act_undo = QAction('復原', self)
        act_undo.setShortcut(QKeySequence('Ctrl+Z'))
        act_undo.triggered.connect(self.undo)
        m.addAction(act_open)
        m.addAction(act_undo)

    def _mk_action(self, text, slot, keys):
        if not isinstance(keys,list): keys=[keys]
        act = QAction(text,self); act.triggered.connect(slot)
        if keys: act.setShortcut(QKeySequence(keys[0]))
        # 不再生成多個相同 shortcut 避免 Ambiguous
        return act

    def _bind_shortcuts(self):
        self.addAction(self._mk_action('下一幀', self.next_frame, Qt.Key.Key_Right))
        self.addAction(self._mk_action('下一幀2', self.next_frame, Qt.Key.Key_D))
        self.addAction(self._mk_action('上一幀', self.prev_frame, Qt.Key.Key_Left))
        self.addAction(self._mk_action('上一幀2', self.prev_frame, Qt.Key.Key_A))
        self.addAction(self._mk_action('播放暫停', self.toggle_play, Qt.Key.Key_Space))
        self.addAction(self._mk_action('上一類別', self.prev_category, Qt.Key.Key_W))
        self.addAction(self._mk_action('下一類別', self.next_category, Qt.Key.Key_S))
        # Undo 已於功能表設定，不再重複

    # project actions (folder-first workflow)
    def open_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, '選擇資料夾', '.')
        if not dir_path:
            return
        self.root_dir = dir_path
        labels_path = os.path.join(self.root_dir, 'labels.txt')
        if not os.path.exists(labels_path):
            QMessageBox.information(self,'類別定義','未找到 labels.txt，建立預設 median_nerve')
            try:
                with open(labels_path,'w',encoding='utf-8') as f: f.write('median_nerve\n')
            except Exception as e:
                QMessageBox.warning(self,'錯誤', f'無法建立 labels.txt: {e}')
        try:
            with open(labels_path,'r',encoding='utf-8') as f:
                self.categories = [l.strip() for l in f if l.strip()]
        except Exception:
            self.categories = []
        if not self.categories:
            self.categories = ['median_nerve']
        self.populate_categories()
        self.populate_video_list()
        if self.list_videos.count() > 0:
            self.list_videos.setCurrentRow(0)
            self.on_select_video(self.list_videos.currentItem())

    def populate_video_list(self):
        self.list_videos.clear()
        if not self.root_dir:
            return
        # 掃描資料夾下的影片檔（不遞迴）
        exts = {'.mp4','.avi','.mov','.mkv'}
        try:
            names = sorted([n for n in os.listdir(self.root_dir) if os.path.splitext(n)[1].lower() in exts])
        except Exception as e:
            QMessageBox.warning(self,'錯誤', f'讀取資料夾失敗: {e}')
            return
        for n in names:
            vp = os.path.join(self.root_dir, n)
            count = self.get_annotation_count_for_video(vp)
            item = QListWidgetItem(f'{n}  (標註: {count})')
            item.setData(Qt.ItemDataRole.UserRole, vp)
            self.list_videos.addItem(item)

    def get_annotation_count_for_video(self, video_path: str) -> int:
    # 讀取即時輸出的 COCO-VID JSON
        out_path = os.path.splitext(video_path)[0] + '.json'
        try:
            if os.path.exists(out_path):
                with open(out_path,'r',encoding='utf-8') as f:
                    data = json.load(f)
                anns = data.get('annotations', [])
                return len(anns)
        except Exception:
            pass
        return 0

    def on_select_video(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str):
        # 先保存當前進度
        if self.project:
            try:
                self.project.save_tmp()
            except Exception:
                pass
        # 初始化專案
        self.project = AnnotationProject(path, self.categories)
        self.slider.setMaximum(max(0, self.project.total_frames - 1))
        if self.canvas_widget is not None:
            self.canvas_widget.setParent(None)
        self.canvas_widget = VideoCanvas(self.project)
        self.canvas_container.insertWidget(0, self.canvas_widget, 1)
        self.video_canvas_placeholder.hide()
    # 預設將焦點設置到畫布，以便接收鍵盤事件（Delete）
        self.canvas_widget.setFocus()
        self.load_frame(0)
        self.update_annotated_frames()

    def update_current_video_item_count(self):
        # 以記憶體中專案為準，彙總當前影片標註數
        if not self.project or not self.root_dir:
            return
        total = 0
        for arr in self.project.annotations_by_frame.values():
            total += len(arr or [])
        # 找到目前選取的 item 並改文字
        cur_item = self.list_videos.currentItem()
        if cur_item:
            name = os.path.basename(self.project.video_path)
            cur_item.setText(f'{name}  (標註: {total})')

    def populate_categories(self):
        self.list_categories.clear()
        # 類別來自資料夾層級
        if not self.categories:
            return
        for name in self.categories:
            self.list_categories.addItem(QListWidgetItem(name))
        if self.list_categories.count()>0:
            self.list_categories.setCurrentRow(0)

    def load_frame(self,index:int):
        if not self.project: return
        index = max(0,min(index,self.project.total_frames-1))
        self.canvas_widget.load_frame(index)
        self.slider.blockSignals(True); self.slider.setValue(index); self.slider.blockSignals(False)
        self.frame_info_label.setText(f'Frame: {index+1} / {self.project.total_frames}')
        self.update_box_list()

    def update_box_list(self, select_track: Optional[int]=None):
        self.list_boxes.clear()
        if not self.project or not self.canvas_widget: return
        for b in self.project.annotations_by_frame.get(self.canvas_widget.frame_index, []):
            name = self.project.id_to_category.get(b.category_id,str(b.category_id))
            item = QListWidgetItem(f'{name} - track_{b.track_id}')
            item.setData(Qt.ItemDataRole.UserRole, b.track_id)
            self.list_boxes.addItem(item)
            if select_track is not None and b.track_id == select_track:
                self.list_boxes.setCurrentItem(item)
        # 更新影片清單的計數顯示
        self.update_current_video_item_count()
        # 同步更新已標註幀清單
        self.update_annotated_frames()

    def on_select_category(self,item: QListWidgetItem):
        if not self.project or not self.canvas_widget: return
        self.canvas_widget.current_category_id = self.project.category_to_id[item.text()]

    def on_select_box(self,item: QListWidgetItem):
        if not self.canvas_widget: return
        tid = item.data(Qt.ItemDataRole.UserRole)
        self.canvas_widget.selected_track_id = tid
        self.canvas_widget.update()

    def prev_frame(self):
        if not self.project: return
        idx = self.canvas_widget.frame_index - 1
        if idx >= 0: self.load_frame(idx)

    def next_frame(self):
        if not self.project: return
        idx = self.canvas_widget.frame_index + 1
        if idx < self.project.total_frames:
            self.load_frame(idx)
        else:
            self.is_playing = False; self.timer.stop(); self.btn_play.setText('播放')

    def toggle_play(self):
        if not self.project: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText('暫停')
            self.timer.start(int(1000/max(1,self.project.fps)))
        else:
            self.btn_play.setText('播放'); self.timer.stop()

    def on_slider(self,val:int): self.load_frame(val)

    def jump_to_frame(self):
        if not self.project: return
        try:
            idx = int(self.jump_edit.text())-1; self.load_frame(idx)
        except: pass

    def save_tmp(self):
        if self.project:
            self.project.save_tmp();
            self.update_current_video_item_count()
            QMessageBox.information(self,'保存','暫存已保存')

    def export_json(self):
        if not self.project: return
        out = self.project.export_coco_vid(); QMessageBox.information(self,'導出完成', f'已輸出: {out}')

    def undo(self):
        if not self.project:
            return
        self.project.undo()
        if self.canvas_widget:
            self.canvas_widget.boxes = self.project.annotations_by_frame.get(self.canvas_widget.frame_index, [])
            self.canvas_widget.update()
        self.update_box_list()

    # ============== 新增：已標註幀清單 ==============
    def update_annotated_frames(self):
        if not self.project:
            self.list_annotated_frames.clear(); return
        # 目前所有有標註的幀（非空）
        items_map = {}
        self.list_annotated_frames.clear()
        for fi in sorted(k for k,v in self.project.annotations_by_frame.items() if v):
            count = len(self.project.annotations_by_frame.get(fi, []))
            text = f'第 {fi+1} 幀（{count}）'
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, fi)
            self.list_annotated_frames.addItem(it)
            items_map[fi] = it
        # 高亮目前幀（若有在清單中）
        if self.canvas_widget:
            cur = self.canvas_widget.frame_index
            if cur in items_map:
                self.list_annotated_frames.setCurrentItem(items_map[cur])

    def on_select_annotated_frame(self, item: QListWidgetItem):
        fi = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(fi, int):
            self.load_frame(fi)

    def eventFilter(self, watched, event):
        # 在已標註幀清單上按 Delete 時觸發批量刪除
        if watched is getattr(self, 'list_annotated_frames', None) and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete:
                self._delete_selected_annotated_frames()
                return True
        return super().eventFilter(watched, event)

    def _delete_selected_annotated_frames(self):
        if not self.project:
            return
        items = self.list_annotated_frames.selectedItems()
        if not items:
            return
        frame_indices = []
        for it in items:
            fi = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(fi, int):
                frame_indices.append(fi)
        if not frame_indices:
            return
        frame_indices = sorted(set(frame_indices))
        resp = QMessageBox.question(self, '刪除標註', f'確定刪除 {len(frame_indices)} 個幀的全部標註？此操作可用復原(Ctrl+Z)。', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return
        for fi in frame_indices:
            arr = list(self.project.annotations_by_frame.get(fi, []))
            for b in arr:
                self.project.delete_box(fi, b.track_id)
        if self.canvas_widget:
            self.canvas_widget.boxes = self.project.annotations_by_frame.get(self.canvas_widget.frame_index, [])
            self.canvas_widget.update()
        self.update_box_list()

    def keyPressEvent(self, event):
        # Delete: 若焦點在「已標註幀」清單，刪除所選幀的所有標註
        if event.key() == Qt.Key.Key_Delete and hasattr(self, 'list_annotated_frames') and self.list_annotated_frames.hasFocus():
            if not self.project:
                return
            items = self.list_annotated_frames.selectedItems()
            if not items:
                return
            # 取得幀索引
            frame_indices = []
            for it in items:
                fi = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(fi, int):
                    frame_indices.append(fi)
            if not frame_indices:
                return
            frame_indices = sorted(set(frame_indices))
            # 確認
            resp = QMessageBox.question(self, '刪除標註', f'確定刪除 {len(frame_indices)} 個幀的全部標註？此操作可用復原(Ctrl+Z)。', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if resp != QMessageBox.StandardButton.Yes:
                return
            # 執行刪除（逐幀推入 undo）
            for fi in frame_indices:
                arr = self.project.annotations_by_frame.get(fi, [])
                # 將當幀所有框以 delete 操作放入 undo（便於還原）
                for b in list(arr):
                    self.project.delete_box(fi, b.track_id)
            # 若當前幀被刪空，更新畫面
            if self.canvas_widget:
                self.canvas_widget.boxes = self.project.annotations_by_frame.get(self.canvas_widget.frame_index, [])
                self.canvas_widget.update()
            self.update_box_list()
            return
        # 交給原本邏輯處理（含刪除當前幀選取框）
        return super().keyPressEvent(event)

    # ============== 新增：插值功能 ==============
    def do_interpolate(self):
        """使用 CSRT 追蹤器以當前幀的唯一標註往後插值 N 幀，並顯示可取消的進度條。"""
        if not self.project or not self.canvas_widget:
            return
        fi = self.canvas_widget.frame_index
        boxes = self.project.annotations_by_frame.get(fi, [])
        if len(boxes) != 1:
            QMessageBox.warning(self, '插值', '當前幀必須有且只有一個標註才能插值。')
            return
        b = boxes[0]
        max_forward = max(0, self.project.total_frames - fi - 1)
        if max_forward <= 0:
            QMessageBox.information(self, '插值', '已到影片尾端，無法往後插值。')
            return
        n, ok = QInputDialog.getInt(self, '插值', '往後插值幾幀？', 5, 1, max_forward)
        if not ok or n <= 0:
            return

        # 取得第一幀與初始化 CSRT 追蹤器
        frame0 = self.project.get_frame(fi)
        if frame0 is None:
            QMessageBox.warning(self, '插值', '無法讀取當前影格。')
            return
        H, W = frame0.shape[:2]

        x, y, w, h = [float(v) for v in b.bbox]
        # 邊界與最小尺寸保護
        x = max(0.0, min(W - 1.0, x))
        y = max(0.0, min(H - 1.0, y))
        w = max(1.0, min(W - x, w))
        h = max(1.0, min(H - y, h))
        init_rect = (x, y, w, h)

        # 建立 CSRT（相容不同 OpenCV 版本/命名）
        def _create_csrt():
            tracker = None
            # OpenCV-contrib 4.x 通常為 cv2.legacy.TrackerCSRT_create
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                tracker = cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerCSRT_create'):
                tracker = cv2.TrackerCSRT_create()
            else:
                # 部分新版使用 MultiTracker API 的新風格類別
                try:
                    tracker = cv2.legacy.TrackerCSRT.create()
                except Exception:
                    tracker = None
            return tracker

        tracker = _create_csrt()
        if tracker is None:
            QMessageBox.critical(self, '插值', '此環境的 OpenCV 未提供 CSRT 追蹤器。請確認已安裝 opencv-contrib-python。')
            return
        ok = tracker.init(frame0, tuple(map(float, init_rect)))
        if not ok:
            QMessageBox.warning(self, '插值', 'CSRT 初始化失敗，請調整框位置或大小後重試。')
            return

        # 進度條
        prog = QProgressDialog('插值中，請稍候…', '取消', 0, n, self)
        prog.setWindowTitle('插值')
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(200)

        last_bbox = [x, y, w, h]
        produced_frames = []  # 紀錄實際插值成功的幀索引
        for step in range(1, n + 1):
            if prog.wasCanceled():
                break
            tfi = fi + step
            frame = self.project.get_frame(tfi)
            if frame is None:
                break
            ok, rect = tracker.update(frame)
            if not ok or rect is None:
                # 追蹤失敗就中止插值
                QMessageBox.information(self, '插值', f'追蹤在第 {tfi + 1} 幀失敗，已停止插值。')
                break
            rx, ry, rw, rh = [float(v) for v in rect]
            # 邊界保護
            rx = max(0.0, min(W - 1.0, rx))
            ry = max(0.0, min(H - 1.0, ry))
            rw = max(1.0, min(W - rx, rw))
            rh = max(1.0, min(H - ry, rh))
            cur_bbox = [rx, ry, rw, rh]
            last_bbox = cur_bbox
            self.project.add_box_with_track_id(tfi, b.track_id, b.category_id, cur_bbox)
            produced_frames.append(tfi)

            prog.setValue(step)
            QApplication.processEvents()

        prog.setValue(n)
        # 平滑插值結果（預設啟用，視窗=5）
        if produced_frames:
            win = 5
            if win % 2 == 0:
                win += 1
            # 使用第一幀作為錨點，不改動其 bbox，只平滑後續插值幀
            seq_frames = [fi] + produced_frames
            self.project.smooth_track_mavg(b.track_id, seq_frames, kernel_size=win, anchor_first=True, modify_first=False)

        # 保持在當前幀，更新右側清單
        self.update_box_list()

    def prev_category(self):
        row = self.list_categories.currentRow()
        if row > 0:
            self.list_categories.setCurrentRow(row-1)
            self.on_select_category(self.list_categories.currentItem())
    def next_category(self):
        row = self.list_categories.currentRow()
        if row < self.list_categories.count()-1:
            self.list_categories.setCurrentRow(row+1)
            self.on_select_category(self.list_categories.currentItem())

    

# ===================== Entry =====================
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
