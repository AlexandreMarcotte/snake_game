import pygame
import random

# Define some constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
CELL_SIZE = 20
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


class Snake:
    def __init__(self):
        self.positions = [(100, 100), (80, 100), (60, 100)]  # Initial position of the snake (3 segments)
        self.direction = pygame.K_RIGHT  # Initial direction
        self.grow = False  # To keep track if the snake needs to grow

    def change_direction(self, key):
        if (key == pygame.K_UP and self.direction != pygame.K_DOWN):
            self.direction = pygame.K_UP
        elif (key == pygame.K_DOWN and self.direction != pygame.K_UP):
            self.direction = pygame.K_DOWN
        elif (key == pygame.K_LEFT and self.direction != pygame.K_RIGHT):
            self.direction = pygame.K_LEFT
        elif (key == pygame.K_RIGHT and self.direction != pygame.K_LEFT):
            self.direction = pygame.K_RIGHT

    def move(self):
        x, y = self.positions[0]
        if self.direction == pygame.K_UP:
            y -= CELL_SIZE
        elif self.direction == pygame.K_DOWN:
            y += CELL_SIZE
        elif self.direction == pygame.K_LEFT:
            x -= CELL_SIZE
        elif self.direction == pygame.K_RIGHT:
            x += CELL_SIZE

        # Insert the new position at the front of the snake
        new_head = (x, y)
        self.positions.insert(0, new_head)

        if not self.grow:
            self.positions.pop()  # Remove the last segment if not growing
        self.grow = False

    def grow_snake(self):
        self.grow = True

    def check_collision(self):
        head = self.positions[0]
        # Check if the snake hits the boundaries
        if head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT:
            return True
        # Check if the snake collides with itself
        if head in self.positions[1:]:
            return True
        return False

    def draw(self, screen):
        for pos in self.positions:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))


class Food:
    def __init__(self):
        self.position = (random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
                         random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE)

    def spawn(self):
        self.position = (random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
                         random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE))


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False

    def check_food_collision(self):
        if self.snake.positions[0] == self.food.position:
            self.snake.grow_snake()
            self.food.spawn()
            self.score += 1

    def display_score(self):
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

    def run(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                elif event.type == pygame.KEYDOWN:
                    self.snake.change_direction(event.key)

            self.snake.move()
            if self.snake.check_collision():
                self.game_over = True

            self.check_food_collision()

            self.screen.fill(BLACK)
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            self.display_score()

            pygame.display.flip()
            self.clock.tick(10)  # Control the frame rate

        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
