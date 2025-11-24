import os
import time
import math
import random

import numpy as np
import threading
from collections import deque
import pygame
import pygame.gfxdraw
import librosa                       
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
                           
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None

          
WIDTH, HEIGHT = 1280, 720
AXIS_DEBUG = True
FPS = 60
MUSIC_FILE = "Piano Hub - Weight of the World.mp3"

                 
PALETTE = [
    (255, 105, 180),        
    (0, 191, 255),                        
    (144, 238, 144),                    
    (255, 255, 255),                       
]

BG_COLOR = (0, 0, 0)
GLOW_ALPHA = 180             

        
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

                                                              
                        
SEGMENTS = 512
                            
GLOW_ALPHA_BASE = 40
                                                 
RING_ALPHA_BASE = 60
                                                          
FADE_START = 1.5
FADE_END = 3.0
                                                                       
                                 
SPHERE_ROT_PERIOD_SEC = 60.0

                                      
                               
                       
MIC_SENS_LOW = 0.08
MIC_SENS_MID = 0.12
MIC_SENS_HIGH = 0.10

                                     
VOICE_BAND = (80, 5000)
            
NOISE_LEARN_SECONDS = 2.0
NOISE_EST_ALPHA = 0.995                        
NOISE_SUBTRACT_FACTOR = 1.0        

          
FREQ_BANDS = {
    'low': (20, 250),
    'mid': (250, 2000),
    'high': (2000, 8000)
}

      
INITIAL_WAKE_DURATION = 30.0                
MAX_RIPPLES = 240
                  
COLOR_BRIGHTNESS_FACTOR = 0.5
             
NOISE_ALPHA = 12                       
CHROMA_OFFSET_PX = 1              
                                 
RINGS_HIGHLIGHT_FACTOR = 0.18
                                  
PAINT_STRIPES_STRIPE_WIDTH_FACTOR = 0.9
                                                                                            
PAINT_STRIPES_STROKE_THICKNESS = 1
                                                                          
PAINT_STRIPES_FILL_SCREEN = True

         
UI_X = 12
UI_Y = 12
BTN_W = 28
BTN_H = 20
BTN_SPACING = 8
FONT_SIZE = 16
SLIDER_LERP = 0.18
HANDLE_COLOR = (180, 180, 180)
HANDLE_HIGHLIGHT_COLOR = (255, 220, 120)
ENERGY_BAR_H = 6
ENERGY_COLORS = {
    'low': (180, 60, 60),
    'mid': (60, 180, 120),
    'high': (80, 140, 220),
}

        
VISUAL_MODES = ['RINGS', 'SONIC_SPHERE', 'PAINT_STRIPES']
PAINT_STRIPES_STATE = {}

                      
INPUT_MODE = 'MIC'
                   
LOADED_AUDIO = {'path': None, 'features': None, 'duration': 0.0}


def choose_audio_file_via_dialog():
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select audio file', filetypes=[('Audio', '*.mp3 *.wav *.mp4 *.m4a'), ('All files', '*.*')])
    root.destroy()
    return file_path


def load_and_analyze_file(path):
    """Load audio file (librosa) and compute per-band energies over time. Returns features dict as analyze_audio does."""
    try:
        feats = analyze_audio(path)
                                
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
            duration = len(y) / float(sr)
        except Exception:
            duration = 0.0
        LOADED_AUDIO['path'] = path
        LOADED_AUDIO['features'] = feats
        LOADED_AUDIO['duration'] = duration
        print(f"Loaded and analyzed file: {path} duration={duration:.2f}s")
        return True
    except Exception as e:
        print('Failed to load/analyze file:', e)
        return False


def play_loaded_file():
    """Play the loaded audio file using pygame.mixer if available."""
    path = LOADED_AUDIO.get('path')
    if not path:
        return False
    try:
        pygame.mixer.init(frequency=SR)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        print('Failed to play file via pygame.mixer:', e)
        return False

                                        
VISUAL_MODE = 0                    
MODE_BTN_X = UI_X + 220
MODE_BTN_Y = UI_Y
MODE_BTN_W = 48
MODE_BTN_H = BTN_H
MODE_BTN_SPACING = 6

                                                      
class LightRipple:
    """
    表示一个从中心向外扩展的圆形/螺旋波纹。
    通过两层绘制实现：半透明宽体 + 细亮高光边。
    """
    def __init__(self, color, start_time, radius=6.0, thickness=1.2, speed=80.0, spiral=False, center=None):
                 
        self.color = color
        self.start_time = start_time
                            
        if center is None:
            self.center = (WIDTH // 2, HEIGHT // 2)
        else:
            self.center = center

                     
        self.phase = random.uniform(0, 2 * math.pi)
        self.twist = random.uniform(-1.2, 1.2) if spiral else 0.0

                        
        choices = [c for c in PALETTE if c != color]
        self.grad_target = random.choice(choices) if choices else color
        f = 0.35 + (math.sin(self.phase) + 1) * 0.15
        self.base_col = (
            int(self.color[0] * (1 - f) + self.grad_target[0] * f),
            int(self.color[1] * (1 - f) + self.grad_target[1] * f),
            int(self.color[2] * (1 - f) + self.grad_target[2] * f),
        )

        self.radius = radius
        self.thickness = thickness
        self.speed = speed          
        self.spiral = spiral
        self.age = 0.0
        self.alive = True
        self.base_alpha = 220

    def update(self, dt, volume_factor, stable_mode):
        self.age += dt
        speed = self.speed * (1.0 + volume_factor * 2.5)
        self.radius += speed * dt
        self.current_thickness = max(0.6, self.thickness * (1.0 + volume_factor * 3.0))
        life_norm = max(0.0, 1.0 - (self.age / 10.0))
        alpha = int(self.base_alpha * life_norm * (1.0 + volume_factor * 1.8))
        self.current_alpha = max(0, min(255, alpha))
        if self.radius - self.current_thickness > max(WIDTH, HEIGHT) * 1.2:
            self.alive = False

    def draw(self, surf):
                                
        cx, cy = int(self.center[0]), int(self.center[1])

        def mix(a, b, f):
            return (int(a[0] * (1 - f) + b[0] * f), int(a[1] * (1 - f) + b[1] * f), int(a[2] * (1 - f) + b[2] * f))

                                     
        base_col = getattr(self, 'base_col', self.color)

                           
        layers = 4
        for layer in range(layers):
            frac = layer / max(1, layers - 1)
                         
            layer_amp = self.current_thickness * (0.6 + frac * 1.2)
            layer_alpha = int(self.current_alpha * (0.35 + 0.5 * (1 - frac)))
                  
            blended = (
                int(base_col[0] * (1 - frac) + self.color[0] * frac),
                int(base_col[1] * (1 - frac) + self.color[1] * frac),
                int(base_col[2] * (1 - frac) + self.color[2] * frac),
            )
            col = (*blended, max(8, layer_alpha))

                                               
            segments = SEGMENTS
            points = []
            for i in range(segments + 1):
                t = i / segments
                              
                r = self.radius
                                                                   
                r += math.sin(t * math.pi * 4 + self.phase) * (layer_amp * 0.45)
                r += 0.6 * math.sin(t * math.pi * 7 + self.phase * 1.3) * (layer_amp * 0.25)
                      
                angle = self.phase + t * (2 * math.pi) * (1.0 + self.twist * 0.2)
                if self.spiral:
                    angle += t * self.twist * 0.9
                    r += t * (self.current_thickness * 2.0)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                points.append((int(x), int(y)))

            tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                                            
            try:
                pygame.draw.aalines(tmp, col, True, points)
            except Exception:
                             
                for j in range(len(points) - 1):
                    pygame.draw.aaline(tmp, col, points[j], points[j + 1])
            for off in (-1, 1):
                shifted = [(p[0] + off, p[1] + off) for p in points]
                try:
                    pygame.draw.aalines(tmp, (*col[:3], max(2, int(col[3] * 0.35))), True, shifted)
                except Exception:
                    for j in range(len(shifted) - 1):
                        pygame.draw.aaline(tmp, (*col[:3], max(2, int(col[3] * 0.35))), shifted[j], shifted[j + 1])

                                      
            if layer == layers - 1:
                                                                             
                try:
                    mixed_edge = (
                        int(self.color[0] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
                        int(self.color[1] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
                        int(self.color[2] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
                    )
                    edge_col = (*mixed_edge, int(self.current_alpha))
                    pygame.draw.aalines(tmp, edge_col, True, points)
                                                                                          
                    try:
                        pygame.draw.aalines(tmp, (255,255,255, max(2, int(self.current_alpha * 0.12))), True, points)
                    except Exception:
                        pass
                except Exception:
                                                      
                    edge_col = (*self.color, int(self.current_alpha))
                    try:
                        pygame.draw.aalines(tmp, edge_col, True, points)
                    except Exception:
                        for j in range(len(points) - 1):
                            pygame.draw.aaline(tmp, edge_col, points[j], points[j + 1])

            surf.blit(tmp, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

def draw_polar_waves(surf, t, energies=None, center=None, layers=None):
    """Clean multi-layer polar waves for background texture."""
    if center is None:
        cx, cy = WIDTH // 2, HEIGHT // 2
    else:
        cx, cy = center

    low_e = float(energies.get('low', 0.0)) if energies is not None else 0.0
    mid_e = float(energies.get('mid', 0.0)) if energies is not None else 0.0
    high_e = float(energies.get('high', 0.0)) if energies is not None else 0.0

    if layers is None:
        base = min(WIDTH, HEIGHT)
        layers = [
            (base * 0.30, (12.0 * 0.55) * (1.0 + low_e * 2.5), 6, 0.6 * (1.0 + low_e * 1.2), 0.0 + mid_e * 1.0, (255, 105, 180), 60 + int(low_e * 120)),
            (base * 0.32, (8.5 * 0.55) * (1.0 + mid_e * 2.0), 9, 1.0 * (1.0 + mid_e * 1.0), 1.2 + high_e * 1.6, (135, 206, 250), 44 + int(mid_e * 100)),
            (base * 0.34, (6.0 * 0.55) * (1.0 + high_e * 3.0), 12, 1.4 * (1.0 + high_e * 1.6), 2.5 + low_e * 0.7, (255, 0, 255), 36 + int(high_e * 88)),
            (base * 0.36, (3.6 * 0.55) * (1.0 + (low_e + mid_e) * 1.2), 18, 1.8 * (1.0 + (mid_e + high_e) * 0.8), 0.7 + random.uniform(0.0, 1.0), (255, 180, 200), 28 + int((low_e + mid_e) * 60)),
        ]

    segments = SEGMENTS
    two_pi = 2 * math.pi
    max_screen_r = math.hypot(WIDTH / 2, HEIGHT / 2)

    for layer in layers:
        base_R0, A_base, k, omega_base, phase_base, color_rgb, alpha_base = layer
        num_rings = 3 + int(4 * low_e)
        ring_spacing = max(8.0, min(WIDTH, HEIGHT) * 0.03)
        growth_speed = 12.0 * (1.0 + low_e * 3.0)

        for ri in range(num_rings):
            R0 = base_R0 + ri * ring_spacing + (growth_speed * t * (0.6 + ri * 0.08))
            A = A_base * (1.0 + 0.6 * mid_e) * (1.0 + 0.15 * ri)
            omega = omega_base * (1.0 + 0.4 * high_e)
            phase = phase_base + ri * 0.2

            dist_norm = max(0.0, min(1.0, R0 / (max_screen_r * 1.2)))
            alpha = int(alpha_base * max(0.0, 1.0 - dist_norm))
            if alpha <= 2:
                continue

            points = []
            for i in range(segments + 1):
                theta = (i / segments) * two_pi
                r = R0 + A * math.sin(k * theta + omega * t + phase)
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)
                points.append((int(x), int(y)))

            tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            col = (*color_rgb, max(2, int(alpha)))
            try:
                pygame.draw.aalines(tmp, col, True, points)
            except Exception:
                for j in range(len(points) - 1):
                    pygame.draw.aaline(tmp, col, points[j], points[j + 1])

            for off in (-1, 1):
                shifted = [(p[0] + off, p[1] + off) for p in points]
                try:
                    pygame.draw.aalines(tmp, (*color_rgb, max(2, int(alpha * 0.45))), True, shifted)
                except Exception:
                    for j in range(len(shifted) - 1):
                        pygame.draw.aaline(tmp, (*color_rgb, max(2, int(alpha * 0.45))), shifted[j], shifted[j + 1])

            surf.blit(tmp, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


                                         


def draw_single_ring(surf, center, R, A, k, omega, phase, color_rgb, alpha, segments=SEGMENTS):
    """绘制单个极坐标环（闭合），简洁实现，避免创建全局 glow surface。"""
    cx, cy = int(center[0]), int(center[1])
    two_pi = 2 * math.pi
    points = []
    for i in range(segments + 1):
        theta = (i / segments) * two_pi
        r = R + A * math.sin(k * theta + omega + phase)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        points.append((int(x), int(y)))

    tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    col = (*color_rgb, max(2, int(alpha)))
    try:
        pygame.draw.aalines(tmp, col, True, points)
    except Exception:
        for j in range(len(points) - 1):
            pygame.draw.aaline(tmp, col, points[j], points[j + 1])

                                                                       
    try:
        rshift = CHROMA_OFFSET_PX
        bshift = -CHROMA_OFFSET_PX
        red_pts = [(p[0] + rshift, p[1]) for p in points]
        blue_pts = [(p[0] + bshift, p[1]) for p in points]
                             
        pygame.draw.aalines(tmp, (255, int(color_rgb[1] * 0.6), int(color_rgb[2] * 0.6), max(2, int(alpha * 0.35))), True, red_pts)
        pygame.draw.aalines(tmp, (int(color_rgb[0] * 0.6), int(color_rgb[1] * 0.6), 255, max(2, int(alpha * 0.35))), True, blue_pts)
                                                                            
        try:
            highlight_alpha = max(2, int(alpha * 0.28))
                                                                    
            mixed = (
                int(color_rgb[0] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
                int(color_rgb[1] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
                int(color_rgb[2] * (1.0 - RINGS_HIGHLIGHT_FACTOR) + 255 * RINGS_HIGHLIGHT_FACTOR),
            )
            pygame.draw.aalines(tmp, (*mixed, highlight_alpha), True, points)
        except Exception:
            pass
    except Exception:
        pass

    surf.blit(tmp, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


def draw_spectrum_bars(surf, center, energies, t, radius=60, bars=64):
                                  
    pass


                    
def lerp_color(c0, c1, f):
    return (
        int(c0[0] * (1 - f) + c1[0] * f),
        int(c0[1] * (1 - f) + c1[1] * f),
        int(c0[2] * (1 - f) + c1[2] * f),
    )


                                                             
def freq_to_color(freq):
            
    purple = (120, 60, 200)
    orange = (255, 150, 80)
    red = (255, 80, 80)
    if freq <= 250.0:
        return purple
    if freq <= 2000.0:
                                          
        f = (freq - 250.0) / (2000.0 - 250.0)
        return lerp_color(purple, orange, f)
                                
    f = min(1.0, (freq - 2000.0) / (8000.0 - 2000.0))
    return lerp_color(orange, red, f)


def draw_particles_and_connections(surf, center, particles, t, life=3.0, fog_alpha=18, axis_mode=False):
    """绘制粒子与连接，更新并移除过期粒子。

    particles: list of dicts with keys: birth, life, x,y,z,size,color,amp
    """
    cx, cy = center
              
    cam_pos = np.array([0.0, 0.0, -200.0])               
                                             
    cam_base_y = math.radians(45.0)
    cam_rot_y = cam_base_y + math.sin(t * 0.18) * 0.12                    
    cam_rot_x = math.sin(t * 0.07) * 0.06             
    cam_z_speed = 60.0                         
    focal_length = 800.0

                             
    axis_half = int(min(300, WIDTH * 0.35))
                                    
    axis_z_base = cam_z_speed * 2.5
    world_origin = np.array([center[0], center[1] + 80, axis_z_base])
                                     
    x_start = world_origin + np.array([-axis_half, 0.0, 0.0])
    x_end = world_origin + np.array([ axis_half, 0.0, 0.0])
    y_end = world_origin + np.array([0.0, -axis_half, 0.0])
    z_end = world_origin + np.array([axis_half * 0.6, -axis_half * 0.6, axis_half * 0.6])
                     
    def rot_matrix(rx, ry):
        Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
        return Ry.dot(Rx)
    R = rot_matrix(cam_rot_x, cam_rot_y)

    def world_to_screen(pt):
                                 
                                            
        p = pt - world_origin
            
        rp = R.dot(p)
                  
        cp = rp - cam_pos
        zc = max(1e-3, cp[2] + 300.0)
        sx = int(center[0] + (cp[0] * focal_length) / zc)
        sy = int(center[1] + (cp[1] * focal_length) / zc)
        depth_scale = focal_length / zc
        return sx, sy, depth_scale

                                                           
    def draw_3d_axis_local(num_xticks=6, num_yticks=4, grid_alpha=20):
        try:
                  
            sx0, sy0, _ = world_to_screen(x_start)
            sx1, sy1, _ = world_to_screen(x_end)
            syx, syy, _ = world_to_screen(y_end)
            szx, szy, _ = world_to_screen(z_end)
                                                                     
            if AXIS_DEBUG:
                print('DEBUG AXIS: world_origin=', world_origin)
                print('DEBUG AXIS: x_start=', x_start, 'x_end=', x_end)
                print('DEBUG AXIS: y_end=', y_end, 'z_end=', z_end)
                print('DEBUG AXIS: proj x_start->', (sx0, sy0), 'x_end->', (sx1, sy1))

                                                                                     
            tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

                       
            pygame.draw.line(tmp, (255,255,255,255), (sx0, sy0), (sx1, sy1), 2)
                   
            top_c = world_origin + np.array([0.0, 180.0, 0.0])
            bot_c = world_origin + np.array([0.0, -40.0, 0.0])
            tcx, tcy, _ = world_to_screen(top_c)
            bcx, bcy, _ = world_to_screen(bot_c)
            pygame.draw.line(tmp, (255,255,255,255), (tcx, tcy), (bcx, bcy), 2)
            pygame.draw.line(tmp, (255,255,255,255), (sx1, sy1), (szx, szy), 2)

            font = pygame.font.SysFont(None, 14)
                                
            for i in range(num_xticks + 1):
                t = i / float(num_xticks)
                world_pt = x_start + (x_end - x_start) * t
                                   
                for j in range(0, 3):
                    offset = (j / 2.0) * (y_end - world_origin)
                    gp = world_pt + offset
                    gx, gy, _ = world_to_screen(gp)
                    try:
                        pygame.draw.line(tmp, (200,200,200, grid_alpha), (gx, gy), (gx+6, gy), 1)
                    except Exception:
                        pass
                                               
                tx, ty, _ = world_to_screen(world_pt)
                label = f"{int((t*100))}%"
                txt_s = font.render(label, True, (230,230,230))
                surf.blit(txt_s, (tx + 6, ty - 8))

                    
            for i in range(num_yticks + 1):
                t = i / float(num_yticks)
                world_pt = y_end + (world_origin - y_end) * t
                tx, ty, _ = world_to_screen(world_pt)
                label = f"{int((1.0 - t)*90 - 45)}°"
                txt_s = font.render(label, True, (230,230,230))
                surf.blit(txt_s, (tx + 6, ty - 8))

                         
            for i in range(4):
                t = i / 3.0
                world_pt = x_start + (z_end - x_start) * t
                tx, ty, _ = world_to_screen(world_pt)
                label = f"{int(t*3)}s"
                txt_s = font.render(label, True, (230,230,230))
                surf.blit(txt_s, (tx + 6, ty - 8))

                                
            try:
                plane_alpha = max(6, int(grid_alpha * 0.25))
                for xi in range(-2, 3):
                    for zi in range(0, 5):
                        wx = world_origin[0] - axis_half + (axis_half * 2) * ((xi + 2) / 4.0)
                        wy = world_origin[1] - 40 - (axis_half * 0.25) * (zi / 4.0)
                        wp = np.array([wx, wy, zi * (axis_half * 0.15)])
                        gx, gy, _ = world_to_screen(wp)
                        try:
                            pygame.draw.circle(tmp, (255,255,255, plane_alpha), (gx, gy), 1)
                        except Exception:
                            pass
            except Exception:
                pass

                                     
            try:
                surf.blit(tmp, (0,0))
            except Exception:
                pass

        except Exception:
                                  
            pass

               
    draw_3d_axis_local()

                           
    proj = []
                                          
    for p in particles:
        age = t - p['birth']
        pz = age * cam_z_speed
        if axis_mode:
            fx = float(p.get('fx', 400.0))
            x_norm = math.log10(max(20.0, fx)) / math.log10(8000.0)
                                        
            world_x = world_origin[0] - axis_half + x_norm * (axis_half * 2)
            amp = float(p.get('amp', 0.0))
                                                              
            world_y = world_origin[1] + 40 - amp * 240
            world_z = pz
            world_pt = np.array([world_x, world_y, world_z])
            sx, sy, depth_scale = world_to_screen(world_pt)
        proj.append((p, sx, sy, depth_scale, world_pt))

                             
    for i in range(len(particles) - 1):
        p1 = particles[i]
        p2 = particles[i + 1]
                
        _, x1, y1, ds1, wp1 = proj[i]
        _, x2, y2, ds2, wp2 = proj[i+1]
        age1 = t - p1['birth']
        age2 = t - p2['birth']
        if age1 < 0 or age2 < 0:
            continue
        a1 = max(0.0, min(1.0, age1 / p1['life']))
        a2 = max(0.0, min(1.0, age2 / p2['life']))
                                                             
        fade_in = 0.18
        fade_out = 0.6
        def alpha_for(age, life):
            if age < fade_in:
                return age / fade_in
            if age > life - fade_out:
                return max(0.0, (life - age) / fade_out)
            return 1.0
        alpha1 = alpha_for(age1, p1['life'])
        alpha2 = alpha_for(age2, p2['life'])
        line_alpha = int(255 * min(alpha1, alpha2) * 0.9)
                    
        col = ( (p1['color'][0] + p2['color'][0]) // 2, (p1['color'][1] + p2['color'][1]) // 2, (p1['color'][2] + p2['color'][2]) // 2 )
        try:
            pygame.draw.aaline(surf, (*col, line_alpha), (x1, y1), (x2, y2))
        except Exception:
            pygame.draw.line(surf, (*col, line_alpha), (x1, y1), (x2, y2))

                               
    remove_idxs = []
    for i, (p, sx, sy, depth_scale, world_pt) in enumerate(proj):
        age = t - p['birth']
        if age >= p['life']:
            remove_idxs.append(i)
            continue
                                                    
        fade_in = 0.18
        fade_out = 0.6
        if age < fade_in:
            fade_a = age / fade_in
        elif age > p['life'] - fade_out:
            fade_a = max(0.0, (p['life'] - age) / fade_out)
        else:
            fade_a = 1.0
        base_alpha = float(p.get('base_alpha', 0.75))
        alpha_val = max(0.0, min(1.0, base_alpha * fade_a))
                                            
        size = max(2, int(p['size'] * (0.6 + 1.8 * depth_scale)))
        surf_col = (p['color'][0], p['color'][1], p['color'][2], int(255 * alpha_val))
        try:
            glow_surf = pygame.Surface((size * 6 + 4, size * 6 + 4), pygame.SRCALPHA)
            gx = glow_surf.get_width() // 2
            gy = glow_surf.get_height() // 2
            for r in range(max(2, size * 3), 0, -2):
                rad_alpha = int((int(surf_col[3]) * (r / max(2, size * 3))) * 0.35)
                pygame.draw.circle(glow_surf, (surf_col[0], surf_col[1], surf_col[2], rad_alpha), (gx, gy), r)
            glow_x = int(sx - gx)
            glow_y = int(sy - gy)
            surf.blit(glow_surf, (glow_x, glow_y), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.circle(surf, surf_col, (int(sx), int(sy)), max(1, size))
        except Exception:
            pygame.draw.circle(surf, surf_col, (int(sx), int(sy)), max(1, size))

                           
    for idx in reversed(remove_idxs):
        particles.pop(idx)



def draw_spoke_lines(surf, center, energies, t, spokes=18, max_len=420):
    """
    粒子垂直频谱（替代原有放射线）：
    - 在水平范围内绘制若干竖直条带（columns），每条由大量小粒子组成。
    - 每列的高度由合成的频带能量决定（低/中/高分布到不同区间），粒子密度和亮度沿高度渐变。
    - 边缘带轻微散射以产生模糊和深度感，主色调为蓝白发光。
    """
    cx, cy = center
    low = float(energies.get('low', 0.0))
    mid = float(energies.get('mid', 0.0))
    high = float(energies.get('high', 0.0))

                    
    cols = 40                     
    total_w = min(540, WIDTH * 0.7)
    spacing = total_w / max(1, cols - 1)
    left = cx - total_w / 2.0
    base_bottom = cy + int(min(200, HEIGHT * 0.22))           
    max_height = min(520, HEIGHT * 0.78)
    particle_spacing = 2.6          
    global_brightness = 1.0

                                      
    for i in range(cols):
        u = i / float(max(1, cols - 1))
        if u < 0.33:
            band_f = u / 0.33
            e = low * (0.6 + 0.4 * band_f)
        elif u < 0.66:
            band_f = (u - 0.33) / 0.33
            e = mid * (0.6 + 0.5 * band_f)
        else:
            band_f = (u - 0.66) / 0.34
            e = high * (0.5 + 0.5 * band_f)

                         
        e = max(0.0, min(1.0, e + 0.06 * math.sin(t * 6.0 + i * 0.4) + 0.02 * (random.random() - 0.5)))

        col_x = left + i * spacing
        col_height = e * max_height
        if col_height < 1.0:
            continue

                             
                          
    n_particles = max(6, int(col_height / particle_spacing * (0.6 + 0.8 * e)))
    for j in range(n_particles):
                                 
            f_h = (j + 1) / float(max(1, n_particles))
                
            y = base_bottom - f_h * col_height
                               
            xj = col_x + random.uniform(-1.2, 1.2) + math.sin((t * 3.0 + i * 0.7 + j * 0.3)) * 0.9

                                          
            base_blue = (50, 170, 255)
            col_rgb = (
                int(base_blue[0] * (1.0 - f_h) + 255 * f_h),
                int(base_blue[1] * (1.0 - f_h) + 255 * f_h),
                int(base_blue[2] * (1.0 - f_h) + 255 * f_h),
            )
                             
            alpha = int(240 * (0.35 + 0.65 * f_h) * (0.75 + 0.5 * e) * global_brightness)
            alpha = max(4, min(255, alpha))

                                     
            edge_spread = max(0.6, (1.0 - e) * 2.0)
            try:
                               
                pygame.draw.circle(surf, (col_rgb[0], col_rgb[1], col_rgb[2], alpha), (int(xj), int(y)), 1)
                if alpha > 18:
                                    
                    try:
                        glow_c = pygame.Surface((6,6), pygame.SRCALPHA)
                        pygame.draw.circle(glow_c, (col_rgb[0], col_rgb[1], col_rgb[2], max(6, int(alpha*0.18))), (3,3), 3)
                        surf.blit(glow_c, (int(xj)-3, int(y)-3), special_flags=pygame.BLEND_RGBA_ADD)
                    except Exception:
                        pass
                                   
                if random.random() < 0.12:
                    sx = xj + random.uniform(-spacing * 0.8, spacing * 0.8) * edge_spread
                    sy = y + random.uniform(-3.0, 3.0)
                    pygame.draw.circle(surf, (col_rgb[0], col_rgb[1], col_rgb[2], max(2, int(alpha * 0.18))), (int(sx), int(sy)), 1)
            except Exception:
                pass

                                 
                 
    try:
        fog = pygame.Surface((int(total_w) + 16, int(max_height) + 24), pygame.SRCALPHA)
        for i_h in range(6):
            a = int(18 * (1.0 - i_h * 0.14))
            pygame.draw.rect(fog, (24, 100, 200, a), (0, int(i_h * 3), fog.get_width(), fog.get_height() - int(i_h * 3)), border_radius=4)
        surf.blit(fog, (int(left - 8), int(base_bottom - max_height - 12)), special_flags=pygame.BLEND_RGBA_ADD)
    except Exception:
        pass


                                             


                                             


                                                                 
def _rand_on_sphere(radius):
    """返回均匀分布在球体表面的点（x,y,z）。"""
    u = random.random()
    v = random.random()
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    return np.array([x, y, z])


def draw_sonic_sphere(surf, center, energies, t, sonic_state):
    """绘制透明的多层环形球体、辐射线与发光节点网络。

    sonic_state 是一个字典，包含运行时的节点列表与参数：
      - 'nodes': list of dict {pos: np.array([x,y,z]), birth, life, col}
      - 'radius_base': 基础半径
    """
    cx, cy = center
    low = energies.get('low', 0.0)
    mid = energies.get('mid', 0.0)
    high = energies.get('high', 0.0)

            
    amp = (low * 0.6 + mid * 0.3 + high * 0.1)
    freq_density = 1.0 + high * 6.0 + mid * 2.5            
    brightness = min(1.0, 0.15 + amp * 3.5)

                  
    base_r = sonic_state.get('radius_base', min(WIDTH, HEIGHT) * 0.18)
                
    spread = amp * min(WIDTH, HEIGHT) * 0.12
    r_outer = base_r + spread

                                               
                                    
    R_world = base_r * 2.0 * 1.5
    focal = 800.0
    cam_pos = np.array([0.0, 0.0, -R_world * 3.0])
    cam_base_y = math.radians(20.0)
    cam_rot_y = cam_base_y + math.sin(t * 0.12) * 0.06
    cam_rot_x = math.sin(t * 0.05) * 0.04

                
    Rx = np.array([[1,0,0],[0,math.cos(cam_rot_x),-math.sin(cam_rot_x)],[0,math.sin(cam_rot_x),math.cos(cam_rot_x)]])
    Ry = np.array([[math.cos(cam_rot_y),0,math.sin(cam_rot_y)],[0,1,0],[-math.sin(cam_rot_y),0,math.cos(cam_rot_y)]])
    Rm = Ry.dot(Rx)

                                                                         
    obj_rot_x = math.radians(45.0)
    obj_rot_y = 0.0
    obj_rot_z = math.radians(20.0)
    Ox = np.array([[1, 0, 0],[0, math.cos(obj_rot_x), -math.sin(obj_rot_x)],[0, math.sin(obj_rot_x), math.cos(obj_rot_x)]])
    Oy = np.array([[math.cos(obj_rot_y), 0, math.sin(obj_rot_y)],[0, 1, 0],[-math.sin(obj_rot_y), 0, math.cos(obj_rot_y)]])
    Oz = np.array([[math.cos(obj_rot_z), -math.sin(obj_rot_z), 0],[math.sin(obj_rot_z), math.cos(obj_rot_z), 0],[0, 0, 1]])
                                                  
    ObjR = Ox.dot(Oy).dot(Oz)

                                                               
    try:
        rot_progress = ((t % SPHERE_ROT_PERIOD_SEC) / SPHERE_ROT_PERIOD_SEC)
    except Exception:
        rot_progress = 0.0
    extra_y = rot_progress * math.pi           
    ExtraY = np.array([[math.cos(extra_y), 0, math.sin(extra_y)],[0,1,0],[-math.sin(extra_y),0,math.cos(extra_y)]])
    ObjR = ExtraY.dot(ObjR)

    def project_pt(p3):
        rp = Rm.dot(p3)
        cp = rp - cam_pos
        zc = cp[2] + 300.0
        if zc <= 1e-3:
            return None
        sx = int(cx + (cp[0] * focal) / zc)
        sy = int(cy + (cp[1] * focal) / zc)
        depth_scale = float(focal) / float(zc)
        return sx, sy, depth_scale

    tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

                                      
    trail = sonic_state.setdefault('trail_surf', None)
    if trail is None:
        trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        sonic_state['trail_surf'] = trail
                                  
    try:
        trail_fade_alpha = 18
        fade_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fade_surf.fill((0, 0, 0, trail_fade_alpha))
        trail.blit(fade_surf, (0, 0))
    except Exception:
        pass
                                    
    try:
        tmp.blit(trail, (0, 0))
    except Exception:
        pass

    
                 
                            
    lat_lines = 8
    lon_segs = max(48, int(72 * (1.0 + high * 0.6)))
    for li in range(lat_lines + 1):
        phi = -math.pi/2 + (li / float(lat_lines)) * math.pi
        ring_r = R_world * math.cos(phi)
        z = R_world * math.sin(phi)
        pts = []
        for i in range(lon_segs + 1):
            theta = (i / float(lon_segs)) * 2 * math.pi
            x3 = ring_r * math.cos(theta)
            y3 = ring_r * math.sin(theta)
            p3 = np.array([x3, y3, z])
                                                           
            p3 = ObjR.dot(p3)
            proj = project_pt(p3)
            if proj is None:
                continue
            sx, sy, ds = proj
            pts.append((sx, sy))
        if len(pts) > 2:
                            
            alpha_line = int((12 + brightness * 160 * (0.25 + (li / float(lat_lines)) * 0.6)) * 0.6)
                       
            col = (30, min(255, int(200 + amp * 40)), 90, alpha_line)
            try:
                pygame.draw.aalines(tmp, col, True, pts)
            except Exception:
                for j in range(len(pts)-1):
                    pygame.draw.aaline(tmp, col, pts[j], pts[j+1])
                                       
            try:
                pygame.draw.aalines(trail, (*col[:3], max(6, alpha_line//6)), True, pts)
            except Exception:
                for j in range(len(pts)-1):
                    pygame.draw.aaline(trail, (*col[:3], max(6, alpha_line//6)), pts[j], pts[j+1])
                                             
                                       
                                                                                     
            glow_alpha = max(4, int(alpha_line * 0.30))
            glow_col = (40 + int(amp * 30), min(255, int(220 + amp * 40)), 100, glow_alpha)
                               

                               
    circles = sonic_state.setdefault('audio_circles', [])
    total_amp = amp
                
    spawn_chance = min(1.0, total_amp * 5.0)
    if random.random() < spawn_chance:
                              
        band = random.choices(['low','mid','high'], weights=[low+0.01, mid+0.01, high+0.01])[0]
                                              
        center3 = _rand_on_sphere(R_world)
                
        life_small = 2.0 + total_amp * 1.6
        small_boxes = []
        for _ in range(2 + int(total_amp * 6)):
            ao = random.uniform(0.0, 1.0)
            ar = ao * (R_world * 0.04)
            ang = random.random() * 2 * math.pi
            small_boxes.append({'off_r': ar, 'off_ang': ang, 'size': random.randint(4,8)})
        circles.append({'pos_local': center3, 'birth': t, 'life': life_small, 'amp': total_amp, 'band': band, 'boxes': small_boxes, 'phase': random.uniform(0, 2*math.pi)})
        print(f"DEBUG: spawned small circle band={band} amp={total_amp:.3f} life={life_small:.2f}")
                               
        life_large = life_small * 1.6
        large_boxes = []
        for _ in range(max(0, 1 + int(total_amp * 2))):
            ao = random.uniform(0.0, 1.0)
            ar = ao * (R_world * 0.06)
            ang = random.random() * 2 * math.pi
            large_boxes.append({'off_r': ar, 'off_ang': ang, 'size': random.randint(6,10)})
        circles.append({'pos_local': center3, 'birth': t, 'life': life_large, 'amp': total_amp, 'band': band, 'boxes': large_boxes, 'phase': random.uniform(0, 2*math.pi)})
        print(f"DEBUG: spawned large circle band={band} amp={total_amp:.3f} life={life_large:.2f}")

             
    new_circles = []
    for c in circles:
        age = t - c['birth']
        if age >= c['life']:
            continue
        life_frac = age / c['life']
                                 
        pos_world = ObjR.dot(c.get('pos_local', c.get('pos', np.array([0.0,0.0,0.0]))))
                
        n = pos_world / (np.linalg.norm(pos_world) + 1e-9)
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(n, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        u = np.cross(up, n)
        u = u / (np.linalg.norm(u) + 1e-9)
        v = np.cross(n, u)

                            
        r0 = R_world * 0.008
        r_world = r0 + life_frac * (R_world * (0.16 + c['amp'] * 0.28))
        segs_c = max(36, int(48 + 32 * (0.5 + (1.0 - life_frac))))
        pts_screen = []
                                      
        phase = c.get('phase', 0.0)
        wave_amp = max(0.3, r_world * 0.02 * (0.4 + c['amp']))
        wave_k = 3.0 + int(2.0 * c['amp'])
        for i in range(segs_c + 1):
            ang = (i / float(segs_c)) * 2 * math.pi
            r_mod = r_world + wave_amp * math.sin(wave_k * ang + phase)
            offset = u * math.cos(ang) * r_mod + v * math.sin(ang) * r_mod
            p3 = pos_world + offset
            proj = project_pt(p3)
            if proj is None:
                continue
            sx, sy, ds = proj
            pts_screen.append((sx, sy))
        if len(pts_screen) > 2:
                                      
            circ_tmp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                                                                   
            alpha = int(220 * (1.0 - life_frac) * (0.45 + c['amp'] * 1.2) * 0.6)
            alpha = max(12, min(255, alpha))
            core_col = (255, 255, 255, alpha)
            try:
                pygame.draw.aalines(circ_tmp, core_col, True, pts_screen)
            except Exception:
                for j in range(len(pts_screen)-1):
                    pygame.draw.aaline(circ_tmp, core_col, pts_screen[j], pts_screen[j+1])
                                    
            try:
                pygame.draw.aalines(circ_tmp, (*core_col[:3], max(4, int(core_col[3]*0.55))), True, pts_screen)
            except Exception:
                for j in range(len(pts_screen)-1):
                    pygame.draw.aaline(circ_tmp, (*core_col[:3], max(4, int(core_col[3]*0.55))), pts_screen[j], pts_screen[j+1])
                                            
            try:
                dot_alpha = max(6, int(alpha * 0.20))
                for p in pts_screen[::max(1, len(pts_screen)//int(24))]:
                    try:
                        pygame.draw.circle(circ_tmp, (255,255,255,dot_alpha), (int(p[0]), int(p[1])), 1)
                    except Exception:
                        pass
            except Exception:
                pass
                                       
            try:
                tmp.blit(circ_tmp, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
            except Exception:
                surf.blit(circ_tmp, (0,0))
                               
            try:
                trail.blit(circ_tmp, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
            except Exception:
                pass
                                      
            try:
                xs = [p[0] for p in pts_screen]
                ys = [p[1] for p in pts_screen]
                cxp = sum(xs) / len(xs)
                cyp = sum(ys) / len(ys)
                pygame.draw.circle(tmp, (255, 200, 180, max(12, core_col[3])), (int(cxp), int(cyp)), 1)
            except Exception:
                pass
                                          
            for b in c.get('boxes', []):
                angb = b['off_ang'] + t * 0.2 * (0.5 + c['amp'])
                off = u * math.cos(angb) * b['off_r'] + v * math.sin(angb) * b['off_r']
                pb = pos_world + off
                projb = project_pt(pb)
                if projb is None:
                    continue
                sbx, sby, sds = projb
                box_alpha = int(200 * (1.0 - life_frac))
                try:
                    pygame.draw.circle(tmp, (40, 220, 120, box_alpha), (int(sbx), int(sby)), max(1, b['size']//2))
                except Exception:
                    try:
                        pygame.draw.circle(tmp, (40, 220, 120, box_alpha), (int(sbx), int(sby)), 1)
                    except Exception:
                        pass
                                
            new_circles.append(c)
    sonic_state['audio_circles'] = new_circles

                           
    surf.blit(tmp, (0, 0))



                                                                         
def project_3d(*args, **kwargs):
    """Stubbed project_3d after SCATTER3D removal: returns center coords and scale=1.0 to keep callers safe."""
                                                               
    center = args[1] if len(args) > 1 else kwargs.get('center', (WIDTH//2, HEIGHT//2))
    return int(center[0]), int(center[1]), 1.0


def draw_grid3d(*args, **kwargs):
    """No-op grid drawer kept for compatibility after SCATTER3D removal."""
    return


def draw_3d_scatter(*args, **kwargs):
    """No-op scatter drawer kept for compatibility after SCATTER3D removal."""
    return



                                                      
def analyze_audio(path):
    print("开始分析音频...")
    y, sr = librosa.load(path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    features = {}
    for band, (low, high) in FREQ_BANDS.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            features[band] = np.zeros(S.shape[1])
            continue
        energy = np.mean(S[idx, :], axis=0)
        energy = librosa.util.normalize(energy)
        features[band] = energy
    print("音频分析完成")
    return features


class MicrophoneAnalyzer:
    """实时麦克风分析：保留最近 N_FFT 的样本，基于 rFFT 计算频带能量并自适应归一化。"""
    def __init__(self, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, freq_bands=FREQ_BANDS):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bands = freq_bands
        self.buffer = deque(maxlen=self.n_fft)
        self.lock = threading.Lock()
        self.stream = None
        self.window = np.hanning(self.n_fft)
                  
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self.band_idx = {}
        for band, (low, high) in self.freq_bands.items():
            idx = np.where((self.freqs >= low) & (self.freqs <= high))[0]
            self.band_idx[band] = idx
                           
        self.running_max = {b: 1e-6 for b in self.freq_bands}
                                    
        self.noise_spec = np.zeros_like(self.freqs)
        self.noise_learn_frames = max(1, int(NOISE_LEARN_SECONDS * (self.sr / self.hop_length)))
        self.frames_seen = 0

    def start(self):
        if sd is None:
            raise RuntimeError('sounddevice 未安装或不可用，请 pip install sounddevice')

        def callback(indata, frames, time_info, status):
            if status:
                                
                pass
                                          
            with self.lock:
                        
                self.buffer.extend(indata[:, 0].tolist())

        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', blocksize=self.hop_length, callback=callback)
        self.stream.start()

    def get_energies(self):
                                       
        with self.lock:
            if len(self.buffer) < self.n_fft:
                return {'low': 0.0, 'mid': 0.0, 'high': 0.0}
                                                     
            buf = np.array(list(self.buffer)[-self.n_fft:], dtype=np.float32)

                   
        buf = buf * self.window
        spec = np.abs(np.fft.rfft(buf))
                                             
        self.frames_seen += 1
        if self.frames_seen <= self.noise_learn_frames:
                            
            self.noise_spec = (self.noise_spec * (self.frames_seen - 1) + spec) / (self.frames_seen)
        else:
                                                          
            total_energy = float(np.mean(spec) + 1e-12)
            noise_level = float(np.mean(self.noise_spec) + 1e-12)
            if total_energy < noise_level * 1.2:
                                
                self.noise_spec = NOISE_EST_ALPHA * self.noise_spec + (1.0 - NOISE_EST_ALPHA) * spec
            else:
                                         
                self.noise_spec = NOISE_EST_ALPHA * self.noise_spec + (1.0 - NOISE_EST_ALPHA) * spec * 0.0

                          
        clean_spec = spec - NOISE_SUBTRACT_FACTOR * self.noise_spec
        clean_spec = np.maximum(clean_spec, 0.0)
        energies = {}
        for band, idx in self.band_idx.items():
            if len(idx) == 0:
                energies[band] = 0.0
                continue
                           
            e = float(np.mean(clean_spec[idx]))
                        
            self.running_max[band] = max(e, self.running_max[band] * 0.995)
            norm = e / (self.running_max[band] + 1e-9)
                         
            if band == 'low':
                sens = MIC_SENS_LOW
            elif band == 'mid':
                sens = MIC_SENS_MID
            else:
                sens = MIC_SENS_HIGH
            energies[band] = float(min(1.0, norm * sens))
        return energies

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass


class SimulatedMicrophoneAnalyzer:
    """Fallback mic when sounddevice is not available. Produces smooth oscillating energies."""
    def __init__(self):
        self.start_time = time.time()

    def start(self):
               
        self.start_time = time.time()

    def stop(self):
        return

    def get_energies(self):
        t = time.time() - self.start_time
        low = max(0.0, 0.5 + 0.5 * math.sin(t * 0.9)) * 0.6
        mid = max(0.0, 0.5 + 0.5 * math.sin(t * 1.6 + 1.2)) * 0.5
        high = max(0.0, 0.5 + 0.5 * math.sin(t * 3.1 + 2.6)) * 0.4
                                 
        low += (random.random() - 0.5) * 0.05
        mid += (random.random() - 0.5) * 0.05
        high += (random.random() - 0.5) * 0.03
        return {'low': max(0.0, low), 'mid': max(0.0, mid), 'high': max(0.0, high)}


                                                     
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Light Ripple Visualizer")
    clock = pygame.time.Clock()

    glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

                          
    color_cycle = []
    def next_color():
        nonlocal color_cycle
        if not color_cycle:
            color_cycle = PALETTE.copy()
            random.shuffle(color_cycle)
        return color_cycle.pop()

              
                                                                               
    try:
        mic = MicrophoneAnalyzer()
        mic.start()
    except Exception as e:
        print("无法启动麦克风分析：", e)
        print("使用模拟麦克风以便离线测试视觉效果。")
        mic = SimulatedMicrophoneAnalyzer()
        mic.start()
    start_time = time.time()
    last_dbg = 0.0
                                           
    global VISUAL_MODE, INPUT_MODE

    snap_saved = False
    dragging = None
    mic_paused = False
    file_paused = False
                               
    smoothed_handles = {'low': MIC_SENS_LOW, 'mid': MIC_SENS_MID, 'high': MIC_SENS_HIGH}

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                            
                for mi, m in enumerate(VISUAL_MODES):
                    bx = MODE_BTN_X
                    by = MODE_BTN_Y + mi * (MODE_BTN_H + MODE_BTN_SPACING)
                    if bx <= mx <= bx + MODE_BTN_W and by <= my <= by + MODE_BTN_H:
                        VISUAL_MODE = mi
                        break
                                                               
                upx = MODE_BTN_X + MODE_BTN_W + 8
                upy = MODE_BTN_Y
                upw = 80
                uph = MODE_BTN_H
                if upx <= mx <= upx + upw and upy <= my <= upy + uph:
                                         
                    file_path = choose_audio_file_via_dialog()
                    if file_path:
                        ok = load_and_analyze_file(file_path)
                        if ok:
                            played = play_loaded_file()
                            INPUT_MODE = 'FILE'
                            try:
                                mic.stop()
                                mic_paused = True
                            except Exception:
                                pass
                                                              
                ctrl_x = MODE_BTN_X + MODE_BTN_W + 8
                ctrl_y = MODE_BTN_Y + MODE_BTN_H + 8
                btn_w = 24
                btn_h = MODE_BTN_H
                      
                if ctrl_x <= mx <= ctrl_x + btn_w and ctrl_y <= my <= ctrl_y + btn_h:
                    if LOADED_AUDIO.get('path'):
                        try:
                            if file_paused:
                                pygame.mixer.music.unpause()
                                file_paused = False
                            else:
                                pygame.mixer.music.play()
                                file_paused = False
                            mic.stop()
                            mic_paused = True
                        except Exception:
                            pass
                       
                px2 = ctrl_x + btn_w + 6
                if px2 <= mx <= px2 + btn_w and ctrl_y <= my <= ctrl_y + btn_h:
                    if LOADED_AUDIO.get('path'):
                        try:
                            pygame.mixer.music.pause()
                            file_paused = True
                        except Exception:
                            pass
                      
                px3 = px2 + btn_w + 6
                if px3 <= mx <= px3 + btn_w and ctrl_y <= my <= ctrl_y + btn_h:
                    if LOADED_AUDIO.get('path'):
                        try:
                            pygame.mixer.music.stop()
                            file_paused = False
                                                                                               
                            pass
                        except Exception:
                            pass
                                                               
                micbtn_x = MODE_BTN_X + MODE_BTN_W + 8
                micbtn_y = MODE_BTN_Y + MODE_BTN_H + 8 + MODE_BTN_H + 12
                micbtn_w = 80
                micbtn_h = MODE_BTN_H
                if micbtn_x <= mx <= micbtn_x + micbtn_w and micbtn_y <= my <= micbtn_y + micbtn_h:
                    try:
                        INPUT_MODE = 'MIC'
                        if mic_paused:
                            mic.start()
                            mic_paused = False
                        try:
                            pygame.mixer.music.stop()
                        except Exception:
                            pass
                    except Exception:
                        pass
                                  
                for i, band in enumerate(('low', 'mid', 'high')):
                    sx = UI_X
                    sy = UI_Y + i * (BTN_H + BTN_SPACING)
                    slider_w = 200
                    slider_h = BTN_H
                    slot_rect = pygame.Rect(sx, sy, slider_w, slider_h)
                    if slot_rect.collidepoint(mx, my):
                                      
                        rel = (mx - sx) / float(slider_w)
                        rel = max(0.0, min(1.0, rel))
                        if band == 'low':
                            globals()['MIC_SENS_LOW'] = rel
                        elif band == 'mid':
                            globals()['MIC_SENS_MID'] = rel
                        else:
                            globals()['MIC_SENS_HIGH'] = rel
                                
                        dragging = (band, sx, sy, slider_w)
                        break
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    band, sx, sy, slider_w = dragging
                    mx, my = event.pos
                    rel = (mx - sx) / float(slider_w)
                    rel = max(0.0, min(1.0, rel))
                    if band == 'low':
                        globals()['MIC_SENS_LOW'] = rel
                    elif band == 'mid':
                        globals()['MIC_SENS_MID'] = rel
                    else:
                        globals()['MIC_SENS_HIGH'] = rel
            elif event.type == pygame.KEYDOWN:
                                                     
                if event.key == pygame.K_1:
                    VISUAL_MODE = 0
                elif event.key == pygame.K_2:
                    VISUAL_MODE = 1
                elif event.key == pygame.K_3:
                    VISUAL_MODE = 2
                                                    
                elif event.key == pygame.K_i:
                            
                    if INPUT_MODE == 'MIC':
                        if LOADED_AUDIO.get('path'):
                            INPUT_MODE = 'FILE'
                            try:
                                                          
                                mic.stop()
                                mic_paused = True
                            except Exception:
                                pass
                            try:
                                started = play_loaded_file()
                                if not started:
                                    print('Playback failed; using analyzed file data if available.')
                            except Exception:
                                pass
                        else:
                                                           
                            INPUT_MODE = 'MIC'
                    else:
                                       
                        INPUT_MODE = 'MIC'
                        try:
                                                       
                            try:
                                pygame.mixer.music.stop()
                            except Exception:
                                pass
                            if mic_paused:
                                mic.start()
                                mic_paused = False
                        except Exception:
                            pass
                elif event.key == pygame.K_f:
                                                               
                    file_path = choose_audio_file_via_dialog()
                    if file_path:
                        ok = load_and_analyze_file(file_path)
                        if ok:
                                                
                            played = play_loaded_file()
                            INPUT_MODE = 'FILE'
                            try:
                                mic.stop()
                                mic_paused = True
                            except Exception:
                                pass
                            if not played:
                                print('Note: playback unavailable; visuals will still use analyzed data.')
                elif event.key == pygame.K_u:
                                                                 
                    try:
                        INPUT_MODE = 'MIC'
                        if mic_paused:
                            mic.start()
                            mic_paused = False
                        try:
                            pygame.mixer.music.stop()
                        except Exception:
                            pass
                    except Exception:
                        pass
                                               
                if event.key == pygame.K_l:
                    globals()['MIC_SENS_LOW'] = min(1.0, MIC_SENS_LOW + 0.01)
                elif event.key == pygame.K_k:
                    globals()['MIC_SENS_LOW'] = max(0.0, MIC_SENS_LOW - 0.01)
                elif event.key == pygame.K_m:
                    globals()['MIC_SENS_MID'] = min(1.0, MIC_SENS_MID + 0.01)
                elif event.key == pygame.K_n:
                    globals()['MIC_SENS_MID'] = max(0.0, MIC_SENS_MID - 0.01)
                elif event.key == pygame.K_h:
                    globals()['MIC_SENS_HIGH'] = min(1.0, MIC_SENS_HIGH + 0.01)
                elif event.key == pygame.K_j:
                    globals()['MIC_SENS_HIGH'] = max(0.0, MIC_SENS_HIGH - 0.01)

                                           
        if INPUT_MODE == 'FILE' and LOADED_AUDIO.get('features') is not None:
                                                              
            try:
                                                        
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms == -1:
                                                                       
                    pos = 0.0
                else:
                    pos = pos_ms / 1000.0
                                      
                feats = LOADED_AUDIO['features']
                                                   
                band = next(iter(feats))
                length = feats[band].shape[0]
                                                                                  
                sec_per_frame = float(HOP_LENGTH) / float(SR)
                idx = int(pos / sec_per_frame)
                idx = max(0, min(length - 1, idx))
                energies = {b: float(feats[b][idx]) for b in feats}
            except Exception:
                energies = mic.get_energies()
        else:
            energies = mic.get_energies()
                                                                                               
        visual_paused = False
        try:
            if INPUT_MODE == 'FILE' and LOADED_AUDIO.get('path'):
                busy = False
                try:
                    busy = pygame.mixer.music.get_busy()
                except Exception:
                    busy = False
                                                                                              
                if file_paused or not busy:
                    visual_paused = True
        except Exception:
            visual_paused = False

                                                                                    
        if visual_paused:
            low = mid = high = 0.0
        else:
            low = float(energies.get('low', 0.0))
            mid = float(energies.get('mid', 0.0))
            high = float(energies.get('high', 0.0))
                                                          
        if 'last_t' not in locals():
            last_t = 0.0
        if visual_paused:
            t = last_t
        else:
            t = time.time() - start_time
            last_t = t
                                                  
        globals()['VISUAL_PAUSED'] = visual_paused
        volume = (low * 0.6 + mid * 0.3 + high * 0.1)

        stable = (time.time() - start_time) >= INITIAL_WAKE_DURATION

        spawn_chance = min(1.0, volume * (6.0 if stable else 2.0))
        if visual_paused:
            spawn_chance = 0.0
                                
        target_max = int(40 + volume * 300)
        if target_max > MAX_RIPPLES:
            target_max = MAX_RIPPLES

                                 
        cx, cy = WIDTH // 2, HEIGHT // 2
        ring_R = min(WIDTH, HEIGHT) * 0.38

                                                         

        screen.fill(BG_COLOR)
                                 
        glow.fill((0, 0, 0, 0))

                                
        try:
            centers
        except NameError:
            cx, cy = WIDTH // 2, HEIGHT // 2
            ring_r = min(WIDTH, HEIGHT) * 0.18
            centers = []
            for i in range(5):
                a = i * (2 * math.pi / 5)
                x = int(cx + ring_r * math.cos(a))
                y = int(cy + ring_r * math.sin(a))
                centers.append((x, y))

                                 
        energies = {'low': low, 'mid': mid, 'high': high}
                               
        try:
            centers_rings
        except NameError:
            centers_rings = {c: [] for c in centers}
                                   
        try:
            centers_particles
        except NameError:
            centers_particles = {c: [] for c in centers}
                                              
        try:
            sonic_state
        except NameError:
            sonic_state = {'nodes': [], 'radius_base': int(min(WIDTH, HEIGHT) * 0.18)}

        for idx, c in enumerate(centers):
            toff = t + idx * 0.08        
                                                                          
                                                                      
            px = c[0]
            py = c[1]
            local_energies = energies

                                           
                                                                             

                                      
            low_e = float(local_energies.get('low', 0.0))
            mid_e = float(local_energies.get('mid', 0.0))
            high_e = float(local_energies.get('high', 0.0))
            spawn_chance = min(1.0, volume * (3.5 if stable else 1.2))
            if globals().get('VISUAL_PAUSED', False):
                spawn_chance = 0.0
            if random.random() < spawn_chance:
                              
                base_A = (6.0 + mid_e * 18.0) * 0.55
                base_k = random.choice([6, 9, 12, 18])
                base_omega = 0.8 + low_e * 1.8 + high_e * 0.6
                base_phase = random.uniform(0, 2 * math.pi)
                base_color = next_color()
                               
                base_color = (
                    int(base_color[0] * COLOR_BRIGHTNESS_FACTOR),
                    int(base_color[1] * COLOR_BRIGHTNESS_FACTOR),
                    int(base_color[2] * COLOR_BRIGHTNESS_FACTOR),
                )
                base_alpha = int(RING_ALPHA_BASE + volume * 200)
                growth_speed = 40.0 + low_e * 260.0 + volume * 300.0
                life_time = 2.0 + (1.0 + volume) * 3.0
                ring = {
                    'R': 6.0,
                    'A': base_A,
                    'k': base_k,
                    'omega': base_omega,
                    'phase': base_phase,
                    'color': base_color,
                    'alpha': base_alpha,
                                                   
                    'glow_color': (min(255, int(base_color[0] * 1.2)), min(255, int(base_color[1] * 1.2)), min(255, int(base_color[2] * 1.2))),
                    'speed': growth_speed,
                    'birth': t,
                    'life': life_time,
                }
                centers_rings[c].append(ring)

                              
            max_screen_r = math.hypot(WIDTH / 2, HEIGHT / 2)
            if VISUAL_MODES[VISUAL_MODE] == 'RINGS':
                new_list = []
                for ring in centers_rings[c]:
                    age = t - ring['birth']
                                                                      
                    if not globals().get('VISUAL_PAUSED', False):
                        ring['R'] += ring['speed'] * (dt)
                                                                                                
                    if age <= FADE_START:
                        cur_alpha = int(ring['alpha'])
                    else:
                        fade_t = (age - FADE_START) / max(0.0001, (FADE_END - FADE_START))
                        fade_t = max(0.0, min(1.0, fade_t))
                        cur_alpha = int(ring['alpha'] * (1.0 - fade_t))

                                                          
                    if cur_alpha <= 1:
                        continue

                    if ring['R'] < max_screen_r * 1.2 and cur_alpha > 1:
                        draw_single_ring(glow, c, ring['R'], ring['A'], ring['k'], ring['omega'] * t, ring['phase'], ring['color'], cur_alpha)
                        new_list.append(ring)
                centers_rings[c] = new_list
            elif VISUAL_MODES[VISUAL_MODE] == 'SONIC_SPHERE':
                                
                draw_sonic_sphere(glow, (WIDTH // 2, HEIGHT // 2), energies, t, sonic_state)
            elif VISUAL_MODES[VISUAL_MODE] == 'PAINT_STRIPES':
                try:
                    draw_painted_stripes(glow, (WIDTH // 2, HEIGHT // 2), energies, t)
                except Exception:
                    pass
                                                    

                                         

        screen.blit(glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

                            
        if time.time() - last_dbg > 1.0:
            last_dbg = time.time()
            try:
                print(f"DBG: mode={VISUAL_MODES[VISUAL_MODE]} energies={energies}")
            except Exception:
                pass

                                                   
            if VISUAL_MODES[VISUAL_MODE] not in ('BARS', 'PAINT_STRIPES'):
                try:
                    noise = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                                    
                    for _ in range(int(WIDTH * HEIGHT * 0.0006)):
                        x = random.randrange(0, WIDTH)
                        y = random.randrange(0, HEIGHT)
                        a = random.randint(0, NOISE_ALPHA)
                        noise.set_at((x, y), (255, 255, 255, a))
                    screen.blit(noise, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                except Exception:
                    pass

        if not snap_saved:
            try:
                pygame.image.save(glow, "screenshot.png")
                print("Saved screenshot.png")
            except Exception as e:
                print("Failed to save screenshot:", e)
            snap_saved = True

        pygame.display.flip()

                              
        try:
            font = pygame.font.SysFont(None, FONT_SIZE)
            for i, band in enumerate(('low', 'mid', 'high')):
                sx = UI_X
                sy = UI_Y + i * (BTN_H + BTN_SPACING)
                slider_w = 200
                           
                pygame.draw.rect(screen, (60, 60, 60), (sx, sy, slider_w, BTN_H))
                         
                if band == 'low':
                    target_val = MIC_SENS_LOW
                elif band == 'mid':
                    target_val = MIC_SENS_MID
                else:
                    target_val = MIC_SENS_HIGH
                                         
                sh = smoothed_handles.get(band, target_val)
                sh = sh + (target_val - sh) * SLIDER_LERP
                smoothed_handles[band] = sh
                        
                hx = int(sx + sh * slider_w) - 6
                             
                is_dragging = (dragging is not None and dragging[0] == band)
                handle_color = HANDLE_HIGHLIGHT_COLOR if is_dragging else HANDLE_COLOR
                pygame.draw.rect(screen, handle_color, (hx, sy - 2, 12, BTN_H + 4))
                            
                energy_val = float(energies.get(band, 0.0))
                eb_w = int(slider_w * max(0.0, min(1.0, energy_val)))
                eb_rect = (sx, sy + BTN_H + 4, eb_w, ENERGY_BAR_H)
                pygame.draw.rect(screen, ENERGY_COLORS.get(band, (200,200,200)), eb_rect)
                      
                label = font.render(f"{band.upper()}: {target_val:.2f}", True, (200, 200, 200))
                screen.blit(label, (sx + slider_w + 12, sy))
                      
            for mi, m in enumerate(VISUAL_MODES):
                bx = MODE_BTN_X
                by = MODE_BTN_Y + mi * (MODE_BTN_H + MODE_BTN_SPACING)
                btn_rect = (bx, by, MODE_BTN_W, MODE_BTN_H)
                if mi == VISUAL_MODE:
                    pygame.draw.rect(screen, (200, 160, 80), btn_rect)
                else:
                    pygame.draw.rect(screen, (80, 80, 80), btn_rect)
                lbl = font.render(m[0], True, (20, 20, 20))
                screen.blit(lbl, (bx + 8, by))
                           
            upx = MODE_BTN_X + MODE_BTN_W + 8
            upy = MODE_BTN_Y
            upw = 80
            uph = MODE_BTN_H
            pygame.draw.rect(screen, (70, 120, 200), (upx, upy, upw, uph))
            up_lbl = font.render('Upload', True, (255,255,255))
            screen.blit(up_lbl, (upx + 8, upy + 2))
                                                   
            ctrl_x = MODE_BTN_X + MODE_BTN_W + 8
            ctrl_y = MODE_BTN_Y + MODE_BTN_H + 8
            btn_w = 24
            btn_h = MODE_BTN_H
                         
            pygame.draw.rect(screen, (80,200,120), (ctrl_x, ctrl_y, btn_w, btn_h))
            pygame.draw.polygon(screen, (20,20,20), [(ctrl_x+6, ctrl_y+6), (ctrl_x+6, ctrl_y+btn_h-6), (ctrl_x+btn_w-6, ctrl_y+btn_h//2)])
                          
            pygame.draw.rect(screen, (200,180,80), (ctrl_x+btn_w+6, ctrl_y, btn_w, btn_h))
            pygame.draw.rect(screen, (20,20,20), (ctrl_x+btn_w+10, ctrl_y+6, 4, btn_h-12))
            pygame.draw.rect(screen, (20,20,20), (ctrl_x+2*btn_w+10, ctrl_y+6, 4, btn_h-12))
                         
            pygame.draw.rect(screen, (200,80,80), (ctrl_x+2*(btn_w+6), ctrl_y, btn_w, btn_h))
            pygame.draw.rect(screen, (20,20,20), (ctrl_x+2*(btn_w+6)+6, ctrl_y+6, btn_w-12, btn_h-12))
                            
            micbtn_x = MODE_BTN_X + MODE_BTN_W + 8
            micbtn_y = MODE_BTN_Y + MODE_BTN_H + 8 + MODE_BTN_H + 12
            micbtn_w = 80
            micbtn_h = MODE_BTN_H
            pygame.draw.rect(screen, (120,120,200), (micbtn_x, micbtn_y, micbtn_w, micbtn_h))
            mic_lbl = font.render('Use MIC', True, (255,255,255))
            screen.blit(mic_lbl, (micbtn_x + 8, micbtn_y + 2))
                                           
            if LOADED_AUDIO.get('path'):
                try:
                    pbar_x = MODE_BTN_X + MODE_BTN_W + 8
                    pbar_y = ctrl_y + btn_h + 6
                    pbar_w = 160
                    pbar_h = 8
                    pygame.draw.rect(screen, (40,40,40), (pbar_x, pbar_y, pbar_w, pbar_h))
                    pos_ms = -1
                    try:
                        pos_ms = pygame.mixer.music.get_pos()
                    except Exception:
                        pos_ms = -1
                    if pos_ms == -1:
                        frac = 0.0
                    else:
                        pos_s = pos_ms / 1000.0
                        dur = LOADED_AUDIO.get('duration') or 0.0
                        frac = min(1.0, pos_s / dur) if dur > 0 else 0.0
                    pygame.draw.rect(screen, (100,200,240), (pbar_x, pbar_y, int(pbar_w*frac), pbar_h))
                except Exception:
                    pass
                    
            mode_label = font.render(VISUAL_MODES[VISUAL_MODE], True, (200, 200, 200))
            screen.blit(mode_label, (MODE_BTN_X, MODE_BTN_Y + (len(VISUAL_MODES)) * (MODE_BTN_H + MODE_BTN_SPACING) + 4))
                       
            input_label = font.render(f"INPUT: {INPUT_MODE}", True, (200,200,200))
            screen.blit(input_label, (MODE_BTN_X, MODE_BTN_Y + (len(VISUAL_MODES)) * (MODE_BTN_H + MODE_BTN_SPACING) + 28))
            if LOADED_AUDIO.get('path'):
                short = os.path.basename(LOADED_AUDIO.get('path'))
                f_label = font.render(short, True, (170,170,170))
                screen.blit(f_label, (MODE_BTN_X, MODE_BTN_Y + (len(VISUAL_MODES)) * (MODE_BTN_H + MODE_BTN_SPACING) + 48))
            pygame.display.update()
        except Exception:
            pass

                                            
                                    

    pygame.quit()

                
    try:
        mic.stop()
    except Exception:
        try:
                                                   
            pass
        except Exception:
            pass

def draw_painted_stripes(surf, center, energies, t, state=None):
    """
    Top-level painterly stripes mode (correctly placed at top level).
    """
    cx, cy = center
    low = float(energies.get('low', 0.0))
    mid = float(energies.get('mid', 0.0))
    high = float(energies.get('high', 0.0))

    st = PAINT_STRIPES_STATE.setdefault('s', {})
                                               
    target_fps = st.get('target_fps', FPS if 'FPS' in globals() else 60)
                                                    
    max_bands = st.get('max_bands', int(3.0 * target_fps))
    if 'bands' not in st:
        st['bands'] = deque(maxlen=max_bands)
    bands = st['bands']

                                                                                         
    if not globals().get('VISUAL_PAUSED', False):
        bands.append({'low': low, 'mid': mid, 'high': high, 't': t, 'seed': random.random(), 'amp': low*0.5 + mid*0.35 + high*0.15})

                                                                                
    if PAINT_STRIPES_FILL_SCREEN:
        W = WIDTH
        left = 0
    else:
        W = int(WIDTH * 0.9 * PAINT_STRIPES_STRIPE_WIDTH_FACTOR)
    H = int(HEIGHT * 0.8)
    left = int(cx - W//2)
    top = int(cy - H//2)
    band_h = max(6, int(H / max(1, len(bands))))

    def mix_palette(b):
        lowv, midv, highv = b['low'], b['mid'], b['high']
        lowc = np.array([34, 70, 64]) * (0.5 + 0.8*lowv)
        midc = np.array([200, 120, 50]) * (0.5 + 1.0*midv)
        highc = np.array([255, 170, 120]) * (0.3 + 1.6*highv)
        c = lowc + midc + highc
        c = np.clip(c, 0, 255).astype(int)
        return (int(c[0]), int(c[1]), int(c[2]))

    block = pygame.Surface((W, H), pygame.SRCALPHA)

    for idx, b in enumerate(reversed(bands)):
        i = idx
        y = int(i * band_h)
        age = i / max(1, len(bands))
        color = mix_palette(b)
        amp = b['amp']

        bs = pygame.Surface((W, band_h+8), pygame.SRCALPHA)
                                                                      
        strokes = 4 + int(min(6, 6 * (0.6 * b['mid'] + 0.4 * b['high'])))
        for s in range(strokes):
                                                                 
            a = int(110 * (1.0 - age) * (0.28 + 0.7*amp) * (1.0 - s/float(strokes)))
            wlen = int(W * (0.55 + 0.45 * math.sin(b['seed']*6.28 + s)))
            for x in range(0, wlen, 10):
                jitter = int(math.sin((x*0.02) + t*0.45 + b['seed']*10 + s) * (4 + 12*amp + 6*b['high']))
                col = (*color, max(8, min(255, a)))
                try:
                                                                                            
                    sign = 1 if math.sin(b['seed'] * 12.345 + s * 1.73) >= 0 else -1
                    length = int(12 + 8 * amp)
                    end_x = x + jitter + sign * length
                                                                       
                    half = max(0, PAINT_STRIPES_STROKE_THICKNESS // 2)
                    for off in range(-half, half + 1):
                        try:
                                                                          
                            pygame.draw.aaline(bs, col, (x + jitter, 2 + s + off), (end_x, band_h - 2 - s + off))
                        except Exception:
                            pass
                except Exception:
                    pass

        try:
            small = pygame.transform.smoothscale(bs, (max(2, W//14), max(2, (band_h+8)//14)))
            blurred = pygame.transform.smoothscale(small, (W, band_h+8))
            try:
                                                              
                mul_alpha = int(160 * (0.2 + 0.7 * b['mid']))
                blurred.fill((255,255,255, mul_alpha), special_flags=pygame.BLEND_RGBA_MULT)
            except Exception:
                pass

        
                                                            
            try:
                                                                     
                add_strength = 1.0 + 0.9 * b['high']
                if add_strength > 1.02:
                                                                    
                    tmpb = blurred.copy()
                    try:
                        tmpb.fill((255,255,255, int(180 * (0.5 + b['high']*0.5))), special_flags=pygame.BLEND_RGBA_MULT)
                    except Exception:
                        pass
                    block.blit(tmpb, (0, y), special_flags=pygame.BLEND_RGBA_ADD)
                else:
                    block.blit(blurred, (0, y))
            except Exception:
                block.blit(blurred, (0, y))
        except Exception:
            block.blit(bs, (0, y))

    total_amp = max([bb['amp'] for bb in bands]) if bands else 0.0
    try:
        drip = pygame.Surface((W, H), pygame.SRCALPHA)
                                                                                      
        high_factor = min(1.0, total_amp * 2.5)
        for i in range(3 + int(5 * high_factor)):
            w = int(2 + (i * (6 + total_amp*18)))
            x = int(W//2 + math.sin(t*0.6 + i*0.9) * (W*0.22)) - w//2
            h = int(H * (0.18 + total_amp * (0.55 + i*0.05) + 0.6 * b['high']))
                                                
            alpha = int(28 * (1.0 - i/(6.0 + 1e-6)) * (0.45 + total_amp*0.9))
            col = (int(220 * (0.6 + total_amp)), int(120 * (0.38 + total_amp*0.5)), int(90 * (0.3 + total_amp)))
            pygame.draw.ellipse(drip, (*col, max(6, alpha)), (x - 8, int(H*0.5 - h//2), w+16, h))
        small = pygame.transform.smoothscale(drip, (max(2, W//14), max(2, H//14)))
        drip = pygame.transform.smoothscale(small, (W, H))
        block.blit(drip, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
    except Exception:
        pass

    try:
        try:
                                                                           
            block.fill((255,255,255,210), special_flags=pygame.BLEND_RGBA_MULT)
        except Exception:
            pass
                                                              
        tmp_block = block.copy()
        try:
            tmp_block.fill((255,255,255,200), special_flags=pygame.BLEND_RGBA_MULT)
        except Exception:
            pass
        try:
            surf.blit(tmp_block, (left, top), special_flags=pygame.BLEND_RGBA_ADD)
        except Exception:
            surf.blit(tmp_block, (left, top))
    except Exception:
        surf.blit(block, (left, top))


if __name__ == '__main__':
    main()
