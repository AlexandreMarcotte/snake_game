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
        self.positions = [(100, 100), (80, 100), (60, 100)]
        self.direction = (1, 0)
        self.grow = False
        self.alive = True
        self.brain = NeuralNetwork(8, 10, 4)  # Updated input size from 6 to 8
        self.frames_since_last_food = 0
        self.total_frames_alive = 0  # Track total frames survived
        self.out_of_bounds = False
        self.growth_rate = 3  # Snake grows by 3 segments after eating food

    def move(self):
        if self.alive:
            self.total_frames_alive += 1  # Increment total frames counter
            x, y = self.positions[0]
            dx, dy = self.direction
            new_head = (x + dx * CELL_SIZE, y + dy * CELL_SIZE)
            self.positions.insert(0, new_head)

            if not self.grow:
                self.positions.pop()
            else:
                self.grow = False

    def set_direction(self, direction):
        self.direction = direction

    def check_collision(self):
        head = self.positions[0]
        if head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT:
            self.alive = False
            self.out_of_bounds = True  # Track if the snake went out of bounds
        elif head in self.positions[1:]:
            self.alive = False

    def grow_snake(self):
        # When the snake eats the apple, grow by multiple segments
        for _ in range(self.growth_rate):
            self.positions.append(self.positions[-1])  # Add extra segments to the tail
        self.frames_since_last_food = 0

    def decide(self, food_position):
        head_x, head_y = self.positions[0]
        food_x, food_y = food_position

        # Calculate distances to the boundaries
        distance_to_left_wall = head_x
        distance_to_right_wall = SCREEN_WIDTH - head_x
        distance_to_top_wall = head_y
        distance_to_bottom_wall = SCREEN_HEIGHT - head_y

        inputs = np.array([
            food_x - head_x,
            food_y - head_y,
            self.direction[0],
            self.direction[1],
            distance_to_left_wall,  # New input: distance to the left wall
            distance_to_right_wall,  # New input: distance to the right wall
            distance_to_top_wall,  # New input: distance to the top wall
            distance_to_bottom_wall  # New input: distance to the bottom wall
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
    def __init__(self, size=CELL_SIZE):  # Set the apple size to one square (CELL_SIZE)
        self.size = size
        self.position = self.random_position()

    def random_position(self):
        return (random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
                random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE)

    def spawn(self):
        self.position = self.random_position()

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.position[0], self.position[1], self.size, self.size))


# Game class
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game with Genetic Algorithm")
        self.clock = pygame.time.Clock()
        self.food = Food()

    def run(self, snakes, max_no_food_frames=200):
        # Main game loop for all snakes
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            all_dead = True

            self.screen.fill(BLACK)
            self.food.draw(self.screen)

            for snake in snakes:
                if snake.alive:
                    all_dead = False
                    snake.decide(self.food.position)
                    snake.move()
                    snake.check_collision()

                    # Check if the snake eats food
                    if snake.positions[0] == self.food.position:
                        snake.grow_snake()
                        self.food.spawn()

                    # Check for no food for too long
                    snake.frames_since_last_food += 1
                    if snake.frames_since_last_food > max_no_food_frames:
                        snake.alive = False

                    for pos in snake.positions:
                        pygame.draw.rect(self.screen, GREEN, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))

            pygame.display.flip()
            self.clock.tick(10)

            if all_dead:  # If all snakes are dead, end the game loop for this generation
                break


# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.snakes = [Snake() for _ in range(population_size)]
        self.best_weights_input_hidden = []
        self.best_weights_hidden_output = []

    def evaluate_fitness(self, snake):
        # Start each snake with a base fitness of 1000
        fitness = 100

        # Reward for the snake's length (number of segments)
        fitness += (len(snake.positions) - 3) * 5  # Reward for eating apples

        # Reward for staying alive longer
        survival_bonus = snake.total_frames_alive  # Reward for survival time
        fitness += survival_bonus

        # Penalize the snake for dying
        if not snake.alive:
            fitness -= 10  # General penalty for dying
            if snake.out_of_bounds:
                fitness -= 20  # Extra penalty for going out of bounds

        # Penalize for being close to boundaries
        head_x, head_y = snake.positions[0]
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

    def select(self):
        self.snakes.sort(key=self.evaluate_fitness, reverse=True)
        return self.snakes[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        child = Snake()
        child.brain.weights_input_hidden = (parent1.brain.weights_input_hidden + parent2.brain.weights_input_hidden) / 2
        child.brain.weights_hidden_output = (
                                                        parent1.brain.weights_hidden_output + parent2.brain.weights_hidden_output) / 2
        return child

    def mutate(self, snake, mutation_rate=0.05):  # Slightly higher mutation rate for exploration
        mutation = np.random.randn(*snake.brain.weights_input_hidden.shape) * mutation_rate
        snake.brain.weights_input_hidden += mutation
        mutation = np.random.randn(*snake.brain.weights_hidden_output.shape) * mutation_rate
        snake.brain.weights_hidden_output += mutation

    def create_new_generation(self, food):
        parents = self.select()
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


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=10)
    game = Game()

    for generation in range(100):
        print(f"Generation {generation + 1}")
        game.run(ga.snakes, max_no_food_frames=200)  # Run the game, limiting frames without food

        # Print the fitness of each snake in this generation
        for i, snake in enumerate(ga.snakes):
            fitness = ga.evaluate_fitness(snake)
            print(f"Snake {i + 1} Fitness: {fitness}")

        ga.log_best_snake_weights()  # Log best snake's weights
        ga.create_new_generation(game.food)  # Produce new generation and reposition the food

    ga.plot_weight_changes()  # Plot the weight changes after all generations are done
