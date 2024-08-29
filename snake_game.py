import pygame
import random
import numpy as np

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
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)  # Bias for hidden layer

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)  # Bias for output layer

    def forward(self, inputs):
        # Calculate hidden layer activations
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = self.softmax(hidden)  # Apply softmax activation to hidden layer

        # Calculate output layer activations
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        return self.softmax(output)  # Apply softmax activation to output layer

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x))  # Prevent overflow by subtracting max
        return exp_values / np.sum(exp_values)



# Snake class
class Snake:
    def __init__(self):
        self.positions = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.grow = False
        self.alive = True
        self.brain = NeuralNetwork(MATRIX_HEIGHT * MATRIX_WIDTH, 10, 4)
        self.frames_since_last_food = 0
        self.total_frames_alive = 0
        self.move_counter = 0


    def move(self, game_matrix):
        if self.alive:
            self.total_frames_alive += 1
            print(self.total_frames_alive)
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

    def decide(self, game_matrix):
        self.move_counter += 1

        if self.move_counter % 10 == 0:  # Every 4th move
            decision = random.choice([0, 1, 2, 3])  # Random move
        else:
            inputs = game_matrix.flatten()  # Flatten the game matrix to feed into the neural network
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

    def clear_matrix(self):
        self.game_matrix = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH), dtype=int)
        self.initialize_fence()

    def print_matrix(self):
        for row in self.game_matrix:
            print(" ".join(map(str, row)))
        print()

    def run(self, snake, max_no_food_frames=200, display=True, speed=10):
        self.clear_matrix()
        self.food.spawn(self.game_matrix)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if display:
                self.screen.fill(BLACK)

            if snake.alive:
                snake.decide(self.game_matrix)
                snake.move(self.game_matrix)

                # Check if the snake eats food
                if snake.positions[0] == self.food.position:
                    snake.grow_snake()
                    self.food.spawn(self.game_matrix)

                # Check for no food for too long
                snake.frames_since_last_food += 1
                if snake.frames_since_last_food > max_no_food_frames:
                    snake.alive = False

            if display:
                self.draw_matrix()
                pygame.display.flip()
                self.clock.tick(speed)
            else:
                self.clock.tick(speed * 5)  # Increase speed when not displaying

            if not snake.alive:
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


# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, population_size=10, mutation_rate=0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [Snake() for _ in range(self.population_size)]
        self.generation = 0

    def evaluate_fitness(self, snake):
        # Fitness function based on survival time and length
        print(f"Total_frame_alive = {snake.total_frames_alive}")
        return snake.total_frames_alive + len(snake.positions) * 10

    def select_parents(self):
        # Sort the population based on fitness
        fitness_scores = [(self.evaluate_fitness(snake), snake) for snake in self.population]
        fitness_scores.sort(reverse=True, key=lambda x: x[0])

        # Select top 50% as parents
        num_parents = self.population_size // 2
        parents = [snake for _, snake in fitness_scores[:num_parents]]
        return parents

    def crossover(self, parent1, parent2):
        child = Snake()

        # Crossover weights of input to hidden layer
        crossover_point = np.random.randint(0, parent1.brain.weights_input_hidden.size)
        flat_parent1 = parent1.brain.weights_input_hidden.flatten()
        flat_parent2 = parent2.brain.weights_input_hidden.flatten()
        new_weights_input_hidden = np.concatenate((flat_parent1[:crossover_point],
                                                   flat_parent2[crossover_point:]))
        child.brain.weights_input_hidden = new_weights_input_hidden.reshape(parent1.brain.weights_input_hidden.shape)

        # Crossover hidden biases
        crossover_point = np.random.randint(0, parent1.brain.bias_hidden.size)
        new_bias_hidden = np.concatenate((parent1.brain.bias_hidden[:crossover_point],
                                          parent2.brain.bias_hidden[crossover_point:]))
        child.brain.bias_hidden = new_bias_hidden

        # Crossover weights of hidden to output layer
        crossover_point = np.random.randint(0, parent1.brain.weights_hidden_output.size)
        flat_parent1 = parent1.brain.weights_hidden_output.flatten()
        flat_parent2 = parent2.brain.weights_hidden_output.flatten()
        new_weights_hidden_output = np.concatenate((flat_parent1[:crossover_point],
                                                    flat_parent2[crossover_point:]))
        child.brain.weights_hidden_output = new_weights_hidden_output.reshape(parent1.brain.weights_hidden_output.shape)

        # Crossover output biases
        crossover_point = np.random.randint(0, parent1.brain.bias_output.size)
        new_bias_output = np.concatenate((parent1.brain.bias_output[:crossover_point],
                                          parent2.brain.bias_output[crossover_point:]))
        child.brain.bias_output = new_bias_output

        return child

    def mutate(self, snake):
        # Mutate weights and biases with a small probability
        mutation = np.random.randn(*snake.brain.weights_input_hidden.shape) * self.mutation_rate
        snake.brain.weights_input_hidden += mutation

        mutation = np.random.randn(*snake.brain.bias_hidden.shape) * self.mutation_rate
        snake.brain.bias_hidden += mutation

        mutation = np.random.randn(*snake.brain.weights_hidden_output.shape) * self.mutation_rate
        snake.brain.weights_hidden_output += mutation

        mutation = np.random.randn(*snake.brain.bias_output.shape) * self.mutation_rate
        snake.brain.bias_output += mutation

    def create_new_generation(self, parents):
        new_population = []

        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def evolve(self, game, generations=10):
        for _ in range(generations):
            for snake in self.population:
                game.run(snake)

            parents = self.select_parents()
            self.create_new_generation(parents)

            best_snake = max(self.population, key=self.evaluate_fitness)
            best_fitness = self.evaluate_fitness(best_snake)
            print(f"Generation {self.generation} - Best Fitness: {best_fitness}")




if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=10, mutation_rate=0.05)  # Example population size and mutation rate
    game = Game()

    print("Starting Evolution...")
    ga.evolve(game, generations=50)  # Evolve for 50 generations
    print("Evolution Completed.")

