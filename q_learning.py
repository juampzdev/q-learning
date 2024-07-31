import pygame
import random
import numpy as np

pygame.init()

# Configuración de colores
white = (255, 255, 255)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)

# Configuración de la pantalla
dis_width = 100
dis_height = 100
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game')

# Configuración de la serpiente
snake_block = 8
snake_speed = 20

clock = pygame.time.Clock()
font_style = pygame.font.SysFont(None, 50)

# Definir acciones posibles
actions = ["LEFT", "RIGHT", "UP", "DOWN"]

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])

def get_state(snake_list, foodx, foody):
    if not snake_list:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # Estado inicial vacío
    head_x, head_y = snake_list[-1]
    state = [
        head_x,
        head_y,
        foodx,
        foody,
        1 if head_x > foodx else 0,
        1 if head_x < foodx else 0,
        1 if head_y > foody else 0,
        1 if head_y < foody else 0,
        1 if (head_x, head_y - snake_block) in snake_list else 0,
        1 if (head_x, head_y + snake_block) in snake_list else 0,
        1 if (head_x - snake_block, head_y) in snake_list else 0,
        1 if (head_x + snake_block, head_y) in snake_list else 0,
    ]
    return tuple(state)

def take_action(action, x1_change, y1_change):
    if action == "LEFT":
        x1_change = -snake_block
        y1_change = 0
    elif action == "RIGHT":
        x1_change = snake_block
        y1_change = 0
    elif action == "UP":
        y1_change = -snake_block
        x1_change = 0
    elif action == "DOWN":
        y1_change = snake_block
        x1_change = 0
    return x1_change, y1_change

# Parámetros de Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05

# Inicializar la tabla Q
Q = {}

def get_Q(state, action):
    return Q.get((state, action), 0.0)

def train_snake():
    global epsilon  # Asegúrate de declarar que usarás la variable global epsilon
    game_over = False
    game_close = False

    x1 = dis_width / 2
    y1 = dis_height / 2

    x1_change = 0
    y1_change = 0

    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    snake_list = [[x1, y1]]  # Asegurando que la lista de la serpiente tenga un valor inicial válido
    length_of_snake = 1

    state = get_state(snake_list, foodx, foody)

    while not game_over:

        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            print('random.choice')
            q_values = [get_Q(state, a) for a in actions]
            max_q = max(q_values)
            action = actions[q_values.index(max_q)]

        x1_change, y1_change = take_action(action, x1_change, y1_change)

        x1 += x1_change
        y1 += y1_change

        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            reward = -10
            game_close = True
        else:
            reward = -1

        dis.fill(black)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = [x1, y1]
        snake_list.append(snake_Head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_Head:
                reward = -10
                game_close = True

        for x in snake_list:
            pygame.draw.rect(dis, white, [x[0], x[1], snake_block, snake_block])

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            length_of_snake += 1
            reward = 10

        new_state = get_state(snake_list, foodx, foody)

        old_q_value = get_Q(state, action)
        q_values_new_state = [get_Q(new_state, a) for a in actions]
        best_q_value_new_state = max(q_values_new_state)
        new_q_value = old_q_value + alpha * (reward + gamma * best_q_value_new_state - old_q_value)
        Q[(state, action)] = new_q_value

        state = new_state

        if game_close:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            game_close = False
            x1 = dis_width / 2
            y1 = dis_height / 2
            x1_change = 0
            y1_change = 0
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            snake_list = [[x1, y1]]
            length_of_snake = 1
            state = get_state(snake_list, foodx, foody)

        clock.tick(snake_speed)

    pygame.quit()
    quit()

train_snake()

