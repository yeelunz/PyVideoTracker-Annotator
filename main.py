import sys, os, json
from dataclasses import dataclass
from typing import List, Dict, Optional
import cv2, numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QAction, QPainter, QPen, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSlider, QLineEdit, QMessageBox
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
        self.category_to_id = {n:i+1 for i,n in enumerate(categories)}
        self.id_to_category = {i+1:n for i,n in enumerate(categories)}
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError('無法開啟影片')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.annotations_by_frame: Dict[int,List[Box]] = {}
        self.next_track_id = 1
        self.tmp_path = os.path.splitext(video_path)[0] + '.json.tmp'
        self.undo_stack: List[Dict] = []
        self._load_tmp()

    def _load_tmp(self):
        if not os.path.exists(self.tmp_path):
            return
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
        data = { 'next_track_id': self.next_track_id,
                 'frames': {str(fi): [ {'track_id':b.track_id,'category_id':b.category_id,'bbox':b.bbox} for b in boxes ] for fi, boxes in self.annotations_by_frame.items()}}
        try:
            with open(self.tmp_path,'w',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False)
        except Exception as e:
            print('寫入暫存檔失敗:', e)

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
        # 若本幀尚無標註且不是第一幀，建立幽靈框供參考
        if not self.boxes and index > 0:
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
        if event.button()!=Qt.MouseButton.LeftButton or self.frame is None: return
        pos = event.position()
        handle = self._hit_handle(pos)
        if handle:
            self.resizing = True; self.resize_handle = handle
            for b in self.boxes:
                if b.track_id == self.selected_track_id:
                    self.original_bbox = b.bbox.copy(); break
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
            self.update(); return
        # start drawing
        self.is_drawing = True; self.start_point = pos; self.current_rect = QRectF(pos,pos)
        self.selected_track_id = None; self.dragging = self.resizing = False
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
        self.resize(1200,800)
        self.project: Optional[AnnotationProject] = None
        self.timer = QTimer(self); self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        # left
        self.canvas_container = QVBoxLayout()
        self.video_canvas_placeholder = QLabel('請開啟影片'); self.video_canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_widget: Optional[VideoCanvas] = None
        self.frame_info_label = QLabel('Frame: -/-')
        self.canvas_container.addWidget(self.video_canvas_placeholder,1)
        self.canvas_container.addWidget(self.frame_info_label)
        controls = QHBoxLayout()
        self.btn_prev = QPushButton('上一幀'); self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_play = QPushButton('播放'); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton('下一幀'); self.btn_next.clicked.connect(self.next_frame)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.valueChanged.connect(self.on_slider)
        self.jump_edit = QLineEdit(); self.jump_edit.setPlaceholderText('跳轉幀 (Enter)'); self.jump_edit.returnPressed.connect(self.jump_to_frame)
        controls.addWidget(self.btn_prev); controls.addWidget(self.btn_play); controls.addWidget(self.btn_next); controls.addWidget(self.slider,1); controls.addWidget(self.jump_edit)
        self.canvas_container.addLayout(controls)
        root.addLayout(self.canvas_container,3)
        # right
        side = QVBoxLayout(); side.addWidget(QLabel('類別'))
        self.list_categories = QListWidget(); self.list_categories.itemClicked.connect(self.on_select_category)
        side.addWidget(self.list_categories,1); side.addWidget(QLabel('本幀標註'))
        self.list_boxes = QListWidget(); self.list_boxes.itemClicked.connect(self.on_select_box)
        side.addWidget(self.list_boxes,2)
        root.addLayout(side,1)
        self._build_menu(); self._bind_shortcuts()

    # menu / shortcuts
    def _build_menu(self):
        m = self.menuBar().addMenu('檔案')
        act_open = QAction('開啟影片', self); act_open.triggered.connect(self.open_video)
        act_save = QAction('儲存進度', self); act_save.triggered.connect(self.save_tmp)
        act_export = QAction('導出 COCO-VID', self); act_export.triggered.connect(self.export_json)
        act_undo = QAction('復原', self); act_undo.setShortcut(QKeySequence('Ctrl+Z')); act_undo.triggered.connect(self.undo)
        m.addAction(act_open); m.addAction(act_save); m.addAction(act_export); m.addAction(act_undo)

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
        self.addAction(self._mk_action('儲存', self.save_tmp, QKeySequence('Ctrl+S')))
        self.addAction(self._mk_action('上一類別', self.prev_category, Qt.Key.Key_W))
        self.addAction(self._mk_action('下一類別', self.next_category, Qt.Key.Key_S))
        # Undo 已於功能表設定，不再重複

    # project actions
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self,'選擇影片','.', 'Video Files (*.mp4 *.avi *.mov)')
        if not path: return
        labels_path = os.path.join(os.path.dirname(path),'labels.txt')
        if not os.path.exists(labels_path):
            QMessageBox.information(self,'類別定義','未找到 labels.txt，建立預設 median_nerve')
            with open(labels_path,'w',encoding='utf-8') as f: f.write('median_nerve\n')
        with open(labels_path,'r',encoding='utf-8') as f:
            categories = [l.strip() for l in f if l.strip()]
        if not categories:
            categories = ['median_nerve']
        # 若多於一個也允許? 需求為預設只有一個, 這裡只保證無時給預設
        self.project = AnnotationProject(path, categories)
        self.slider.setMaximum(self.project.total_frames - 1)
        if self.canvas_widget is not None: self.canvas_widget.setParent(None)
        self.canvas_widget = VideoCanvas(self.project)
        self.canvas_container.insertWidget(0, self.canvas_widget, 1)
        self.video_canvas_placeholder.hide()
        self.load_frame(0)
        self.populate_categories()

    def populate_categories(self):
        self.list_categories.clear()
        if not self.project: return
        for name in self.project.categories:
            self.list_categories.addItem(QListWidgetItem(name))
        if self.list_categories.count()>0:
            self.list_categories.setCurrentRow(0)

    def load_frame(self,index:int):
        if not self.project: return
        index = max(0,min(index,self.project.total_frames-1))
        self.project.save_tmp()
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
            self.project.save_tmp(); QMessageBox.information(self,'保存','暫存已保存')

    def export_json(self):
        if not self.project: return
        out = self.project.export_coco_vid(); QMessageBox.information(self,'導出完成', f'已輸出: {out}')

    def undo(self):
        if not self.project: return
        self.project.undo()
        if self.canvas_widget:
            self.canvas_widget.boxes = self.project.annotations_by_frame.get(self.canvas_widget.frame_index, [])
            self.canvas_widget.update()
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

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Delete and self.canvas_widget and self.canvas_widget.selected_track_id is not None:
            self.canvas_widget.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

# ===================== Entry =====================
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
