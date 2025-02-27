import pygame
import random
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Инициализация Pygame
pygame.init()

# Настройки экрана
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Создание динозаврика
dino_img = pygame.image.load('C:/Users/ChilmanovA/Desktop/AI/dino/dino.png')  # Путь к изображению динозаврика
dino_img = pygame.transform.scale(dino_img, (50, 50))  # Масштабируем изображение
dino_rect = pygame.Rect(100, HEIGHT - dino_img.get_height(), 50, 50) 

# Параметры движения
dino_gravity = 0
obstacle_speed = 12  # Ускорили скорость
score = 0  # Счет

# MediaPipe для детекции руки
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Модель для ИИ
model = Sequential([
    Flatten(input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])

# Режим игры: 1 - игрок, 2 - ИИ, 3 - рука
mode = 3  # Меняй здесь режим

def detect_two_hands():
    ret, frame = cap.read()
    if not ret:
        return False
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:  # Две руки
        cv2.imshow("Hand Tracking", frame)
        return True
    return False

def detect_hand():
    ret, frame = cap.read()
    if not ret:
        return False
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    cv2.imshow("Hand Tracking", frame)
    return bool(result.multi_hand_landmarks)

def generate_obstacle():
    width = random.randint(20, 70) 
    height = random.randint(30, 120) 
    y_position = random.randint(HEIGHT - height - 100, HEIGHT - 100) 
    return pygame.Rect(WIDTH, y_position, width, height) 


def generate_air_obstacle():
    width = random.randint(20, 70)
    height = random.randint(30, 120)
    y_pos = random.randint(50, HEIGHT - 200)  # Разместим препятствие на высоте
    return pygame.Rect(WIDTH, y_pos, width, height)

obstacle_rect = generate_obstacle()
air_obstacle = generate_air_obstacle()

game_data = []  # Хранилище данных игры

running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if mode == 1:  # Игрок управляет вручную
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15

    elif mode == 2:  # ИИ управляет
        input_data = np.array([[dino_rect.bottom, obstacle_rect.left]])
        action = model.predict(input_data)
        if action > 0.5 and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15

    elif mode == 3:  # Управление через камеру
        if detect_two_hands() and dino_rect.bottom >= HEIGHT - 100:  # Дабл прыжок при 2 руках
            dino_gravity = -20  # Сила второго прыжка

        elif detect_hand() and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15

    game_data.append([dino_rect.bottom, obstacle_rect.left, int(dino_gravity < 0)])

    dino_gravity += 1
    dino_rect.y += dino_gravity
    if dino_rect.bottom >= HEIGHT - 100:
        dino_rect.bottom = HEIGHT - 100

    obstacle_rect.x -= obstacle_speed
    if obstacle_rect.right < 0:
        obstacle_rect = generate_obstacle()
        score += 1
        obstacle_speed += 0.5  # Ускорение препятствий с каждым прыжком

    air_obstacle.x -= obstacle_speed
    if air_obstacle.right < 0:
        air_obstacle = generate_air_obstacle()

    # Проверка столкновения
    if dino_rect.colliderect(obstacle_rect) or dino_rect.colliderect(air_obstacle):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Game Over! Score: {score} R - Restart, Q - Quit", True, BLACK)
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.delay(2000)

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        dino_rect = pygame.Rect(100, HEIGHT - 100, 50, 50)
                        obstacle_rect = generate_obstacle()
                        air_obstacle = generate_air_obstacle()
                        obstacle_speed = 9
                        score = 0
                        waiting = False
                    elif event.key == pygame.K_q:
                        running = False
                        waiting = False

    # Отображение счета
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    screen.blit(dino_img, dino_rect.topleft)  
    pygame.draw.rect(screen, RED, obstacle_rect)  
    pygame.draw.rect(screen, BLUE, air_obstacle)  
    pygame.display.flip()
    clock.tick(30)

# Сохранение данных игры
np.save
