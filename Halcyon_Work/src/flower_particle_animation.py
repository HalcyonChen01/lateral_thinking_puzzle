import pygame
import random
import math
import sys

# 配置参数
PARTICLE_RADIUS = 1             # 粒子半径（可控）
PARTICLE_LIFETIME = 450         # 粒子生命周期（帧，可控，影响消散/聚拢速度）
WIDTH, HEIGHT = 800, 600         # 画布宽高
FPS = 30                        # 动画帧率
PARTICLE_NUM = 8000             # 粒子数量
SHOW_IMAGE_TIME = 1.0           # 静止展示原图时间（秒）
DISINTEGRATE_TIME = 15.0         # 消散动画时长（秒）
REINTEGRATE_TIME = 8.0          # 聚拢动画时长（秒）
CYCLE_TIME = SHOW_IMAGE_TIME + DISINTEGRATE_TIME + REINTEGRATE_TIME  # 总循环时长

# 载入图片
IMAGE_PATH = 'H:/Polyu Code/Halcyon/Halcyon_Work/AI_TEST_FLOWER.png'  # 推荐使用斜杠，避免转义问题

class Particle:
    def __init__(self, x, y, color):
        self.orig_x = x
        self.orig_y = y
        self.x = x
        self.y = y
        self.color = color
        self.radius = PARTICLE_RADIUS           # 粒子半径（由参数区控制）
        self.state = 'normal'
        self.progress = 0.0
        self.target_x = x
        self.target_y = y
        # 全屏随机方向，力度最大，确保粒子消散出屏幕
        angle = random.uniform(0, 2 * math.pi)
        diag = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
        dist = diag * 1.2
        self.exit_x = self.orig_x + math.cos(angle) * dist
        self.exit_y = self.orig_y + math.sin(angle) * dist
        self.enter_x = self.orig_x
        self.enter_y = self.orig_y
        self.lifetime = PARTICLE_LIFETIME      # 粒子生命周期（帧），由参数区控制
        self.age = 0

    def update(self, t, phase):
        # phase: 'show', 'disintegrate', 'reintegrate'
        if phase == 'show':
            self.x = self.orig_x
            self.y = self.orig_y
            self.age = 0
        elif phase == 'disintegrate':
            self.age += 1
            progress = min(1.0, self.age / self.lifetime)
            self.x = self.orig_x + (self.exit_x - self.orig_x) * progress
            self.y = self.orig_y + (self.exit_y - self.orig_y) * progress
        elif phase == 'reintegrate':
            # 聚拢动画，5秒内从消散位置回到原始位置
            progress = min(1.0, (t / REINTEGRATE_TIME))
            self.x = self.exit_x + (self.orig_x - self.exit_x) * progress
            self.y = self.exit_y + (self.orig_y - self.exit_y) * progress

    def draw(self, surface, alpha=255):
        # 支持可选alpha参数，alpha为0-255
        color = self.color
        if len(color) == 3:
            color = (*color, alpha)
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (int(self.radius), int(self.radius)), self.radius)
        surface.blit(s, (int(self.x-self.radius), int(self.y-self.radius)))


def get_particles_from_image(image, num_particles):
    img = pygame.image.load(image).convert_alpha()
    img = pygame.transform.smoothscale(img, (400, 400))
    img_rect = img.get_rect(center=(WIDTH//2, HEIGHT//2))
    pixels = []
    for x in range(img_rect.left, img_rect.right, 1):
        for y in range(img_rect.top, img_rect.bottom, 1):
            color = img.get_at((x - img_rect.left, y - img_rect.top))
            if color.a > 0:
                pixels.append((x, y, (color.r, color.g, color.b)))
    random.shuffle(pixels)
    particles = []
    for i in range(min(num_particles, len(pixels))):
        x, y, color = pixels[i]
        particles.append(Particle(x, y, color))
    return particles


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Flower Particle Disintegration')
    clock = pygame.time.Clock()
    particles = get_particles_from_image(IMAGE_PATH, PARTICLE_NUM)
    flower_img = pygame.image.load(IMAGE_PATH).convert_alpha()
    flower_img = pygame.transform.smoothscale(flower_img, (400, 400))
    flower_rect = flower_img.get_rect(center=(WIDTH//2, HEIGHT//2))
    start_time = pygame.time.get_ticks()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        now = (pygame.time.get_ticks() - start_time) / 1000.0
        t = now % CYCLE_TIME

        if t < SHOW_IMAGE_TIME:
            phase = 'show'
        elif t < SHOW_IMAGE_TIME + DISINTEGRATE_TIME:
            phase = 'disintegrate'
        else:
            phase = 'reintegrate'

        screen.fill((30, 30, 30))
        if phase == 'show':
            screen.blit(flower_img, flower_rect)
            for p in particles:
                p.update(t, phase)
        elif phase == 'disintegrate':
            # 消散动画，粒子铺满全屏并彻底消失，无透明度变化
            for p in particles:
                if phase == 'disintegrate' and p.age >= p.lifetime:
                    p.age = 0
                if p.age == 0:
                    p.age = 1
                p.update(t - SHOW_IMAGE_TIME, phase)
                progress = min(1.0, p.age / p.lifetime)
                if progress < 1.0:
                    p.draw(screen, 255)
        elif phase == 'reintegrate':
            # 聚拢动画，粒子始终为不透明
            for p in particles:
                if p.age >= p.lifetime:
                    p.age = 0
                if p.age == 0:
                    p.age = 1
                p.update(t - SHOW_IMAGE_TIME - DISINTEGRATE_TIME, phase)
                p.draw(screen, 255)

        # 粒子循环
        if phase == 'show':
            for p in particles:
                p.age = 0

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == '__main__':
    main()
