import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# Define constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
CELL_SIZE = 20
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Matrix dimensions based on screen size and cell size
MATRIX_WIDTH = SCREEN_WIDTH // CELL_SIZE
MATRIX_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        hidden = np.dot(inputs, self.weights_input_hidden)
        hidden = np.tanh(hidden)
        output = np.dot(hidden, self.weights_hidden_output)
        return np.tanh(output)

# Snake class
class Snake:
    def __init__(self):
        self.positions = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.grow = False
        self.alive = True
        self.brain = NeuralNetwork(8, 10, 4)
        self.frames_since_last_food = 0
        self.total_frames_alive = 0

    def move(self, game_matrix):
        if self.alive:
            self.total_frames_alive += 1
            head_x, head_y = self.positions[0]
            dx, dy = self.direction
            new_head = (head_x + dx, head_y + dy)
            self.positions.insert(0, new_head)

            if not self.grow:
                tail = self.positions.pop()
                game_matrix[tail[1]][tail[0]] = 0  # Remove tail from the matrix
            else:
                self.grow = False

            self.check_collision(game_matrix)

    def set_direction(self, direction):
        self.direction = direction

    def check_collision(self, game_matrix):
        head_x, head_y = self.positions[0]
        if game_matrix[head_y][head_x] == 3 or (head_x, head_y) in self.positions[1:]:
            self.alive = False
        else:
            game_matrix[head_y][head_x] = 1  # Update matrix with snake head

    def grow_snake(self):
        self.grow = True
        self.frames_since_last_food = 0

    def decide(self, food_position):
        head_x, head_y = self.positions[0]
        food_x, food_y = food_position

        inputs = np.array([
            food_x - head_x,
            food_y - head_y,
            self.direction[0],
            self.direction[1],
            head_x,  # Distance to left wall
            MATRIX_WIDTH - head_x,  # Distance to right wall
            head_y,  # Distance to top wall
            MATRIX_HEIGHT - head_y  # Distance to bottom wall
        ])

        output = self.brain.forward(inputs)
        decision = np.argmax(output)

        if decision == 0 and self.direction != (0, 1):  # Up
            self.set_direction((0, -1))
        elif decision == 1 and self.direction != (0, -1):  # Down
            self.set_direction((0, 1))
        elif decision == 2 and self.direction != (1, 0):  # Left
            self.set_direction((-1, 0))
        elif decision == 3 and self.direction != (-1, 0):  # Right
            self.set_direction((1, 0))

# Food class
class Food:
    def __init__(self):
        self.position = self.random_position()

    def random_position(self):
        return (random.randint(1, MATRIX_WIDTH - 2), random.randint(1, MATRIX_HEIGHT - 2))

    def spawn(self, game_matrix):
        self.position = self.random_position()
        game_matrix[self.position[1]][self.position[0]] = 2

# Game class
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game with Matrix")
        self.clock = pygame.time.Clock()
        self.food = Food()
        self.game_matrix = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH), dtype=int)
        self.initialize_fence()

    def initialize_fence(self):
        # Create the boundary (fence) in the matrix
        self.game_matrix[0, :] = 3
        self.game_matrix[-1, :] = 3
        self.game_matrix[:, 0] = 3
        self.game_matrix[:, -1] = 3

    def print_matrix(self):
        for row in self.game_matrix:
            print(" ".join(map(str, row)))
        print()

    def run(self, snakes, max_no_food_frames=200):
        self.food.spawn(self.game_matrix)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            all_dead = True

            self.screen.fill(BLACK)

            for snake in snakes:
                if snake.alive:
                    all_dead = False
                    snake.decide(self.food.position)
                    snake.move(self.game_matrix)

                    # Check if the snake eats food
                    if snake.positions[0] == self.food.position:
                        snake.grow_snake()
                        self.food.spawn(self.game_matrix)

                    # Check for no food for too long
                    snake.frames_since_last_food += 1
                    if snake.frames_since_last_food > max_no_food_frames:
                        snake.alive = False

            self.draw_matrix()

            pygame.display.flip()
            self.clock.tick(10)

            self.print_matrix()  # Print the matrix at each step

            if all_dead:
                break

    def draw_matrix(self):
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                if self.game_matrix[y][x] == 1:
                    color = GREEN
                elif self.game_matrix[y][x] == 2:
                    color = RED
                elif self.game_matrix[y][x] == 3:
                    color = WHITE
                else:
                    continue  # Skip drawing for 0 (background)
                pygame.draw.rect(self.screen, color, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

import numpy as np
import random

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.snakes = [Snake() for _ in range(population_size)]
        self.best_weights_input_hidden = []
        self.best_weights_hidden_output = []

    def evaluate_fitness(self, snake, food_position):
        # Start each snake with a base fitness of 1000
        fitness = 1000

        # Reward for the snake's length (number of segments)
        fitness += (len(snake.positions) - 3) * 500  # Reward for eating apples

        # Reward for staying alive longer
        survival_bonus = snake.total_frames_alive  # Reward for survival time
        fitness += survival_bonus

        # Calculate distance to the apple (Euclidean distance)
        head_x, head_y = snake.positions[0]
        food_x, food_y = food_position
        distance_to_apple = np.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2)

        # Add fitness points based on how close the snake is to the apple
        if distance_to_apple > 0:  # Avoid division by zero
            fitness += 100 / distance_to_apple  # Closer distance gives more fitness

        # Penalize the snake for dying
        if not snake.alive:
            fitness -= 1000  # General penalty for dying
            if snake.out_of_bounds:
                fitness -= 20  # Extra penalty for going out of bounds

        # Penalize for being close to boundaries
        distance_to_left_wall = head_x
        distance_to_right_wall = SCREEN_WIDTH - head_x
        distance_to_top_wall = head_y
        distance_to_bottom_wall = SCREEN_HEIGHT - head_y

        min_distance = 2 * CELL_SIZE  # Threshold for penalizing proximity to walls
        if distance_to_left_wall < min_distance:
            fitness -= 5
        if distance_to_right_wall < min_distance:
            fitness -= 5
        if distance_to_top_wall < min_distance:
            fitness -= 5
        if distance_to_bottom_wall < min_distance:
            fitness -= 5

        # Ensure fitness is non-negative
        return max(fitness, 0)

    def select(self, food_position):
        # Sort the snakes by their fitness scores
        self.snakes.sort(key=lambda snake: self.evaluate_fitness(snake, food_position), reverse=True)
        return self.snakes[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        child = Snake()
        child.brain.weights_input_hidden = (parent1.brain.weights_input_hidden + parent2.brain.weights_input_hidden) / 2
        child.brain.weights_hidden_output = (parent1.brain.weights_hidden_output + parent2.brain.weights_hidden_output) / 2
        return child

    def mutate(self, snake, mutation_rate=0.05):
        mutation = np.random.randn(*snake.brain.weights_input_hidden.shape) * mutation_rate
        snake.brain.weights_input_hidden += mutation
        mutation = np.random.randn(*snake.brain.weights_hidden_output.shape) * mutation_rate
        snake.brain.weights_hidden_output += mutation

    def create_new_generation(self, food):
        # Select parents based on fitness
        parents = self.select(food.position)
        next_generation = []
        for _ in range(self.population_size):
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            next_generation.append(child)
        self.snakes = next_generation

        food.spawn()  # Reposition the food after creating a new generation

    def log_best_snake_weights(self):
        # Log the weights of the best-performing snake's brain (for tracking progress over generations)
        best_snake = self.snakes[0]  # The best snake is the first one after sorting by fitness
        self.best_weights_input_hidden.append(best_snake.brain.weights_input_hidden.copy())
        self.best_weights_hidden_output.append(best_snake.brain.weights_hidden_output.copy())

    def plot_weight_changes(self):
        num_generations = len(self.best_weights_input_hidden)
        plt.figure(figsize=(12, 6))

        # Plot input-hidden layer weights
        plt.subplot(1, 2, 1)
        for i in range(self.best_weights_input_hidden[0].shape[1]):
            plt.plot([self.best_weights_input_hidden[g][:, i].mean() for g in range(num_generations)],
                     label=f"Hidden Neuron {i + 1}")
        plt.title('Input-to-Hidden Weights Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Weight')
        plt.legend()

        # Plot hidden-output layer weights
        plt.subplot(1, 2, 2)
        for i in range(self.best_weights_hidden_output[0].shape[1]):
            plt.plot([self.best_weights_hidden_output[g][:, i].mean() for g in range(num_generations)],
                     label=f"Output Neuron {i + 1}")
        plt.title('Hidden-to-Output Weights Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Weight')
        plt.legend()

        plt.tight_layout()
        plt.show()



# Main execution
if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=1)
    game = Game()

    for generation in range(1):
        print(f"Generation {generation + 1}")
        game.run(ga.snakes)

