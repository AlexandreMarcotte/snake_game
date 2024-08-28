import pygame
import random
import numpy as np
import math

# Define constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
CELL_SIZE = 20
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize random weights for a single-layer neural network
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        # Simple feed-forward neural network with sigmoid activation
        hidden = np.dot(inputs, self.weights_input_hidden)
        hidden = np.tanh(hidden)  # activation
        output = np.dot(hidden, self.weights_hidden_output)
        return np.tanh(output)  # activation for output layer


# Snake class
class Snake:
    def __init__(self):
        self.positions = [(100, 100), (80, 100), (60, 100)]  # Initial position
        self.direction = (1, 0)  # Initial direction (moving right)
        self.grow = False
        self.alive = True
        self.brain = NeuralNetwork(6, 10, 4)  # Neural network with 6 inputs, 10 hidden neurons, and 4 outputs

    def move(self):
        if self.alive:
            x, y = self.positions[0]
            dx, dy = self.direction
            new_head = (x + dx * CELL_SIZE, y + dy * CELL_SIZE)
            self.positions.insert(0, new_head)

            if not self.grow:
                self.positions.pop()  # Remove the tail
            else:
                self.grow = False

    def set_direction(self, direction):
        self.direction = direction

    def check_collision(self):
        head = self.positions[0]
        # Check if the snake hits the boundaries
        if head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT:
            self.alive = False
        # Check if the snake collides with itself
        if head in self.positions[1:]:
            self.alive = False

    def grow_snake(self):
        self.grow = True

    def decide(self, food_position):
        # Use the neural network to decide the direction based on the environment
        head_x, head_y = self.positions[0]
        food_x, food_y = food_position
        inputs = np.array([
            food_x - head_x,
            food_y - head_y,
            self.direction[0],
            self.direction[1],
            SCREEN_WIDTH - head_x,
            SCREEN_HEIGHT - head_y
        ])

        output = self.brain.forward(inputs)
        decision = np.argmax(output)  # Choose the action with the highest output value

        if decision == 0 and self.direction != (0, 1):  # Up
            self.set_direction((0, -1))
        elif decision == 1 and self.direction != (0, -1):  # Down
            self.set_direction((0, 1))
        elif decision == 2 and self.direction != (1, 0):  # Left
            self.set_direction((-1, 0))
        elif decision == 3 and self.direction != (-1, 0):  # Right
            self.set_direction((1, 0))


class Food:
    def __init__(self):
        self.position = self.random_position()

    def random_position(self):
        return (random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
                random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE)

    def spawn(self):
        self.position = self.random_position()

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE))


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game with Genetic Algorithm")
        self.clock = pygame.time.Clock()
        self.food = Food()

    def run(self, snakes):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.screen.fill(BLACK)
            self.food.draw(self.screen)

            for snake in snakes:
                if snake.alive:
                    snake.decide(self.food.position)
                    snake.move()
                    snake.check_collision()
                    if snake.positions[0] == self.food.position:
                        snake.grow_snake()
                        self.food.spawn()

                    for pos in snake.positions:
                        pygame.draw.rect(self.screen, GREEN, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))

            pygame.display.flip()
            self.clock.tick(10)


# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.snakes = [Snake() for _ in range(population_size)]

    def evaluate_fitness(self, snake):
        # Fitness function: simple measure based on score and survival time
        return len(snake.positions)

    def select(self):
        # Select snakes with high fitness for breeding (simple tournament selection)
        self.snakes.sort(key=self.evaluate_fitness, reverse=True)
        return self.snakes[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        # Crossover of neural network weights between two parents
        child = Snake()
        child.brain.weights_input_hidden = (parent1.brain.weights_input_hidden + parent2.brain.weights_input_hidden) / 2
        child.brain.weights_hidden_output = (
                                                        parent1.brain.weights_hidden_output + parent2.brain.weights_hidden_output) / 2
        return child

    def mutate(self, snake, mutation_rate=0.01):
        # Mutation to introduce variations in the snake's brain
        mutation = np.random.randn(*snake.brain.weights_input_hidden.shape) * mutation_rate
        snake.brain.weights_input_hidden += mutation
        mutation = np.random.randn(*snake.brain.weights_hidden_output.shape) * mutation_rate
        snake.brain.weights_hidden_output += mutation

    def create_new_generation(self):
        # Select and breed new generation
        parents = self.select()
        next_generation = []
        for _ in range(self.population_size):
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            next_generation.append(child)
        self.snakes = next_generation


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=10)
    game = Game()

    # Run the game for multiple generations
    for generation in range(100):  # 100 generations
        print(f"Generation {generation + 1}")
        game.run(ga.snakes)
        ga.create_new_generation()
