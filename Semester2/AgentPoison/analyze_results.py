import json
import matplotlib.pyplot as plt
import numpy as np

# Load benign results
with open('MyAgent/result/benign_run/results.json', 'r') as f:
    benign_results = json.load(f)

# Extract metrics
episodes = [r['episode'] for r in benign_results]
total_rewards = [r['total_reward'] for r in benign_results]
memory_sizes = [r['memory_size'] for r in benign_results]
avg_agent_rewards = [np.mean([np.mean(rewards) for rewards in r['agent_rewards']]) for r in benign_results]
steps = [r['steps'] for r in benign_results]

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot total rewards
ax1.plot(episodes, total_rewards, 'b-', label='Total Reward')
ax1.set_title('Total Reward per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.grid(True)

# Plot memory growth
ax2.plot(episodes, memory_sizes, 'g-', label='Memory Size')
ax2.set_title('Memory Size Growth')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Memory Size')
ax2.grid(True)

# Plot average agent rewards
ax3.plot(episodes, avg_agent_rewards, 'r-', label='Avg Agent Reward')
ax3.set_title('Average Agent Reward per Episode')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Average Reward')
ax3.grid(True)

# Plot episode steps
ax4.plot(episodes, steps, 'm-', label='Steps')
ax4.set_title('Steps per Episode')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Steps')
ax4.grid(True)

plt.tight_layout()
plt.savefig('benign_analysis.png')

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print(f"Average Steps per Episode: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
print(f"Final Memory Size: {memory_sizes[-1]}")
print(f"Average Agent Reward: {np.mean(avg_agent_rewards):.2f} ± {np.std(avg_agent_rewards):.2f}") 