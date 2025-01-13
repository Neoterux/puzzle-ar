import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, 
    QWidget, QLabel, QHBoxLayout, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from puzzle_game.core.processing.segmenting import SimpleSegmenter
from puzzle_game.utils import load_image

# Constants
BOARD_SIZE = 8
PATTERN_SIZE = (7, 7)

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    camera_ready = pyqtSignal(bool)
    progress_update = pyqtSignal(int)  # Nueva señal para actualizar progreso
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.mutex = QMutex()
        self.camera = None
    
    def run(self):
        # Simulamos pasos de inicialización
        steps = [
            "Inicializando cámara...",
            "Configurando parámetros...",
            "Preparando buffer...",
            "Estableciendo conexión...",
            "Ajustando resolución..."
        ]
        
        for i, step in enumerate(steps):
            self.msleep(300)  # Simulamos trabajo
            self.progress_update.emit((i + 1) * 20)  # 20% por paso
        
        self.mutex.lock()
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.running = True
            self.camera_ready.emit(True)
        else:
            self.camera_ready.emit(False)
        self.mutex.unlock()
        
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(30)  # ~33 fps
    
    def stop(self):
        self.mutex.lock()
        self.running = False
        if self.camera is not None:
            self.camera.release()
        self.mutex.unlock()

class ProcessingThread(QThread):
    processed_frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = []
        self.mutex = QMutex()
        self.running = False
    
    def add_frame(self, frame):
        self.mutex.lock()
        self.frame_queue = [frame]  # Keep only the latest frame
        self.mutex.unlock()
    
    def run(self):
        self.running = True
        while self.running:
            self.mutex.lock()
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                self.mutex.unlock()
                
                # Process frame
                processed_frame = self.main_window.detect_chessboard(frame)
                self.processed_frame_ready.emit(processed_frame)
            else:
                self.mutex.unlock()
            self.msleep(10)
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Puzzle AR - Chess Detection")
        self.setGeometry(100, 100, 1024, 768)
        
        # Initialize selection variables
        self.first_selection = None
        self.waiting_second_selection = False
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create top layout for camera feed
        top_layout = QVBoxLayout()
        self.main_layout.addLayout(top_layout)
        
        # Create buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Empezar", self)
        self.exit_button = QPushButton("Salir", self)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.exit_button)
        top_layout.addLayout(button_layout)
        
        # Create loading progress bar and label
        self.loading_container = QWidget()
        loading_layout = QVBoxLayout(self.loading_container)
        
        self.loading_label = QLabel("Iniciando cámara...", self)
        self.loading_label.setStyleSheet("font-size: 16px; color: #666;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        loading_layout.addWidget(self.progress_bar)
        
        self.loading_container.hide()
        top_layout.addWidget(self.loading_container)
        
        # Create label for camera feed
        self.camera_label = QLabel(self)
        top_layout.addWidget(self.camera_label)
        self.camera_label.hide()
        
        # Create bottom layout for minimap
        bottom_layout = QHBoxLayout()
        self.main_layout.addLayout(bottom_layout)
        
        # Create label for minimap
        self.minimap_label = QLabel(self)
        self.minimap_label.setFixedSize(200, 200)
        bottom_layout.addWidget(self.minimap_label)
        bottom_layout.addStretch()
        
        # Initialize threads
        self.camera_thread = CameraThread()
        self.processing_thread = ProcessingThread(self)
        
        # Connect signals
        self.camera_thread.frame_ready.connect(self.processing_thread.add_frame)
        self.processing_thread.processed_frame_ready.connect(self.update_frame)
        self.camera_thread.camera_ready.connect(self.on_camera_ready)
        self.camera_thread.progress_update.connect(self.update_progress)
        
        # Connect buttons to functions
        self.start_button.clicked.connect(self.start_camera)
        self.exit_button.clicked.connect(self.close)
        
        # Load and segment the test image
        try:
            image_path = "test.jpeg"
            self.original_image = load_image(image_path)
            if self.original_image is None or isinstance(self.original_image, str):
                raise ValueError(f"Error loading image, the image is not valid {self.original_image}")
            self.segmenter = SimpleSegmenter()
            self.segments = self.segmenter.segment_image(self.original_image, BOARD_SIZE, BOARD_SIZE)
            # Convert segments to numpy arrays for display
            self.segment_images = [segment.image for segment in self.segments]
            # Shuffle segments
            self.segments = self.segmenter.shuffle_segments(self.segments)
            # Create and display minimap
            self.update_minimap()
        except Exception as e:
            print(f"Error loading image at initialization: {e}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            self.segments = None
            self.segment_images = None

    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)

    def on_camera_ready(self, success):
        """Handle camera initialization result"""
        self.loading_container.hide()
        if success:
            self.camera_label.show()
            self.processing_thread.start()
        else:
            QMessageBox.critical(
                self,
                "Error",
                "No se pudo inicializar la cámara. Por favor, verifica que esté conectada correctamente."
            )
            self.start_button.show()
    
    def start_camera(self):
        """Start camera with loading animation"""
        self.start_button.hide()
        self.progress_bar.setValue(0)
        self.loading_container.show()
        self.camera_thread.start()

    def update_frame(self, frame):
        """Update the camera feed display"""
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.width(), 
                self.camera_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            ))

    def detect_laser(self, frame):
        """Detect red laser pointer in frame"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define red color range (laser pointer)
        lower_red1 = np.array([0, 200, 200])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 200, 200])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours of the laser point
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (likely the laser point)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5:  # Minimum area threshold
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None

    def get_cell_from_point(self, point, corners, pattern_size):
        """Get the cell coordinates (i,j) from a point in the frame"""
        if not point:
            return None
            
        px, py = point
        top_left = corners[0][0]
        top_right = corners[pattern_size[0]-1][0]
        bottom_left = corners[-pattern_size[0]][0]
        
        # Calculate cell dimensions
        cell_width = (top_right[0] - top_left[0]) / BOARD_SIZE
        cell_height = (bottom_left[1] - top_left[1]) / BOARD_SIZE
        
        # Calculate relative position
        rel_x = px - top_left[0]
        rel_y = py - top_left[1]
        
        # Get cell coordinates
        j = int(rel_x / cell_width)
        i = int(rel_y / cell_height)
        
        # Check if point is within bounds
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            return (i, j)
        return None

    def swap_segments(self, pos1, pos2):
        """Swap two segments in the puzzle"""
        if pos1 and pos2 and self.segments is not None:
            i1, j1 = pos1
            i2, j2 = pos2
            idx1 = i1 * BOARD_SIZE + j1
            idx2 = i2 * BOARD_SIZE + j2
            
            if idx1 < len(self.segments) and idx2 < len(self.segments):
                # Swap current positions
                self.segments[idx1].current_position, self.segments[idx2].current_position = \
                    self.segments[idx2].current_position, self.segments[idx1].current_position
                
                # Swap segments in list
                self.segments[idx1], self.segments[idx2] = self.segments[idx2], self.segments[idx1]
                
                # Update minimap
                self.update_minimap()
                
                return True
        return False
    
    def update_minimap(self):
        """Update the minimap with current segment positions"""
        if self.segments is None or self.original_image is None:
            return
            
        # Create a blank image for the minimap
        minimap = np.zeros_like(self.original_image)
        h, w = self.original_image.shape[:2]
        cell_h, cell_w = h // BOARD_SIZE, w // BOARD_SIZE
        
        # Place segments in their current positions
        for segment in self.segments:
            current_pos = segment.current_position
            y1 = current_pos.x * cell_h
            y2 = (current_pos.x + 1) * cell_h
            x1 = current_pos.y * cell_w
            x2 = (current_pos.y + 1) * cell_w
            
            minimap[y1:y2, x1:x2] = segment.image
        
        # Resize minimap for display
        minimap = cv2.resize(minimap, (200, 200))
        
        # Convert to Qt format and display
        rgb_minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_minimap.shape
        bytes_per_line = ch * w
        qt_minimap = QImage(rgb_minimap.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.minimap_label.setPixmap(QPixmap.fromImage(qt_minimap))

    def place_segment_on_cell(self, frame, segment_idx, corners, i, j, pattern_size):
        """Place a segment on a specific cell of the chessboard"""
        if self.segment_images is None or segment_idx >= len(self.segment_images):
            return frame
            
        segment = self.segment_images[segment_idx]
        
        # Get the corners for the cell
        top_left = corners[0][0]
        top_right = corners[pattern_size[0]-1][0]
        bottom_left = corners[-pattern_size[0]][0]
        
        # Calculate cell dimensions
        cell_width = (top_right[0] - top_left[0]) / BOARD_SIZE
        cell_height = (bottom_left[1] - top_left[1]) / BOARD_SIZE
        
        # Calculate cell corners
        x1 = int(top_left[0] + cell_width * j)
        y1 = int(top_left[1] + cell_height * i)
        x2 = int(top_left[0] + cell_width * (j + 1))
        y2 = int(top_left[1] + cell_height * (j + 1))
        
        # Define destination points for perspective transform
        dst_points = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # Define source points (original segment corners)
        h, w = segment.shape[:2]
        src_points = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp segment
        warped_segment = cv2.warpPerspective(segment, matrix, (frame.shape[1], frame.shape[0]))
        
        # Create a mask for the warped segment
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [dst_points.astype(np.int32)], (255, 255, 255))
        
        # Blend the warped segment with the frame
        frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        frame_fg = cv2.bitwise_and(warped_segment, mask)
        frame = cv2.add(frame_bg, frame_fg)
        
        # If this cell is selected, highlight it
        if self.first_selection == (i, j):
            cv2.polylines(frame, [dst_points.astype(np.int32)], True, (0, 0, 255), 2)
        
        return frame

    def detect_chessboard(self, frame):
        if self.segments is None:
            return frame
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
        
        if ret:
            # Refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw the corners on the frame
            cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, ret)
            
            # Detect laser pointer
            laser_point = self.detect_laser(frame)
            if laser_point:
                # Draw laser point
                cv2.circle(frame, laser_point, 5, (0, 0, 255), -1)
                
                # Get cell coordinates for laser point
                cell = self.get_cell_from_point(laser_point, corners, PATTERN_SIZE)
                if cell:
                    if not self.waiting_second_selection:
                        # First selection
                        self.first_selection = cell
                        self.waiting_second_selection = True
                    else:
                        # Second selection - try to swap
                        if self.swap_segments(self.first_selection, cell):
                            print(f"Swapped segments at {self.first_selection} and {cell}")
                        self.first_selection = None
                        self.waiting_second_selection = False
            
            # Get the four corners of the chessboard
            if len(corners) == PATTERN_SIZE[0] * PATTERN_SIZE[1]:
                # Place segments on cells
                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        segment_idx = i * BOARD_SIZE + j
                        if segment_idx < len(self.segments):
                            # Get the shuffled segment index
                            shuffled_segment = self.segments[segment_idx]
                            original_idx = shuffled_segment.original_position.x * BOARD_SIZE + shuffled_segment.original_position.y
                            frame = self.place_segment_on_cell(
                                frame, 
                                original_idx,
                                corners,
                                i, j,
                                PATTERN_SIZE
                            )
                
                # Get the outer corners
                top_left = corners[0][0]
                top_right = corners[PATTERN_SIZE[0]-1][0]
                bottom_left = corners[-PATTERN_SIZE[0]][0]
                bottom_right = corners[-1][0]
                
                # Draw the chessboard boundary
                pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        return frame
    
    def closeEvent(self, a0):
        self.camera_thread.stop()
        self.processing_thread.stop()
        self.camera_thread.wait()
        self.processing_thread.wait()
        a0.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
