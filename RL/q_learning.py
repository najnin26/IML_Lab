import numpy as np
import matplotlib.pyplot as plt

# Grid world setup
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # Up, Down, Left, Right
goal_state = 15

# Q-learning parameters
Q_table = np.zeros((n_states, n_actions))
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000

# Action definitions: Up, Down, Left, Right
actions = {
    0: -grid_size,   # Up
    1: grid_size,    # Down
    2: -1,           # Left
    3: 1             # Right
}

# Function to get next state based on action with boundary check
def get_next_state(state, action):
    row, col = divmod(state, grid_size)
    if action == 0 and row > 0:
        return state - grid_size
    elif action == 1 and row < grid_size - 1:
        return state + grid_size
    elif action == 2 and col > 0:
        return state - 1
    elif action == 3 and col < grid_size - 1:
        return state + 1
    return state  # invalid move

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)
    
    while current_state != goal_state:
        # ε-greedy policy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(Q_table[current_state])
        
        next_state = get_next_state(current_state, action)
        reward = 1 if next_state == goal_state else 0
        
        # Q-value update
        Q_table[current_state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )
        
        current_state = next_state

# ---------------- Visualization ----------------

# Heatmap of max Q-values per state
q_values_grid = np.max(Q_table, axis=1).reshape((grid_size, grid_size))

plt.figure(figsize=(6, 6))
plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Max Q-value')
plt.title('Learned Q-values for each state')
plt.xticks(np.arange(grid_size))
plt.yticks(np.arange(grid_size))
plt.gca().invert_yaxis()
plt.grid(True)

# Annotate values
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, f'{q_values_grid[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()

# Optimal Policy Visualization
action_arrows = {
    0: '↑',  # Up
    1: '↓',
    2: '←',
    3: '→'
}

policy_grid = np.full((grid_size, grid_size), '', dtype=object)
for state in range(n_states):
    i, j = divmod(state, grid_size)
    if state == goal_state:
        policy_grid[i, j] = 'G'
    else:
        best_action = np.argmax(Q_table[state])
        policy_grid[i, j] = action_arrows[best_action]

# Plot policy arrows
plt.figure(figsize=(6, 6))
plt.imshow(np.zeros_like(q_values_grid), cmap='gray', vmin=0, vmax=1)  # Blank background
plt.title('Optimal Policy (Best Action Directions)')
plt.xticks(np.arange(grid_size))
plt.yticks(np.arange(grid_size))
plt.gca().invert_yaxis()
plt.grid(True)

# Annotate arrows
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=16, color='blue')

plt.show()

# Print Q-table
print("Learned Q-table:")
print(Q_table)
