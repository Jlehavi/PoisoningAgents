import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def add_trend_line(x, y, color, label):
    # Calculate trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    # Add trend line to plot
    plt.plot(x, p(x), f"--{color}", alpha=0.8, label=f"{label} Trend")
    return p[1]  # Return slope

def plot_rewards(benign_results, adversarial_results=None):
    plt.figure(figsize=(12, 6))
    
    # Process benign results
    benign_episodes = [r['episode'] for r in benign_results]
    benign_rewards = [r['total_reward'] for r in benign_results]
    plt.plot(benign_episodes, benign_rewards, label='Benign', color='blue')
    benign_slope = add_trend_line(benign_episodes, benign_rewards, 'b', 'Benign')
    benign_avg = np.mean(benign_rewards)
    
    # Process adversarial results if available
    if adversarial_results:
        # Limit to first 50 episodes
        adv_episodes = [r['episode'] for r in adversarial_results[:50]]
        adv_rewards = [r['total_reward'] for r in adversarial_results[:50]]
        plt.plot(adv_episodes, adv_rewards, label='Adversarial', color='red')
        adv_slope = add_trend_line(adv_episodes, adv_rewards, 'r', 'Adversarial')
        adv_avg = np.mean(adv_rewards)
        
        # Add average reward comparison text
        plt.text(0.02, 0.98, f'Average Rewards:\nBenign: {benign_avg:.2f}\nAdversarial: {adv_avg:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_comparison.png')
    plt.close()

def plot_steps(benign_results, adversarial_results=None):
    plt.figure(figsize=(12, 6))
    
    # Process benign results
    benign_episodes = [r['episode'] for r in benign_results]
    benign_steps = [r['steps'] for r in benign_results]
    plt.plot(benign_episodes, benign_steps, label='Benign', color='blue')
    benign_slope = add_trend_line(benign_episodes, benign_steps, 'b', 'Benign')
    benign_avg = np.mean(benign_steps)
    
    # Process adversarial results if available
    if adversarial_results:
        # Limit to first 50 episodes
        adv_episodes = [r['episode'] for r in adversarial_results[:50]]
        adv_steps = [r['steps'] for r in adversarial_results[:50]]
        plt.plot(adv_episodes, adv_steps, label='Adversarial', color='red')
        adv_slope = add_trend_line(adv_episodes, adv_steps, 'r', 'Adversarial')
        adv_avg = np.mean(adv_steps)
        
        # Add average steps comparison text
        plt.text(0.02, 0.98, f'Average Steps:\nBenign: {benign_avg:.2f}\nAdversarial: {adv_avg:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('steps_comparison.png')
    plt.close()

def plot_agent_rewards(benign_results, adversarial_results=None):
    plt.figure(figsize=(12, 6))
    
    # Process benign results
    benign_episodes = [r['episode'] for r in benign_results]
    # Extract agent rewards from the nested structure
    benign_agent_rewards = []
    for r in benign_results:
        episode_rewards = [0] * 3  # Initialize with 0 for each agent
        for step in r['results']:
            agents_rewards = step['info']['agents_rewards']
            for i, reward in enumerate(agents_rewards):
                episode_rewards[i] += reward
        benign_agent_rewards.append(episode_rewards)
    benign_agent_rewards = np.array(benign_agent_rewards)
    
    benign_avgs = []
    for i in range(benign_agent_rewards.shape[1]):
        plt.plot(benign_episodes, benign_agent_rewards[:, i], 
                label=f'Benign Agent {i+1}', color=f'C{i}', linestyle='-')
        slope = add_trend_line(benign_episodes, benign_agent_rewards[:, i], f'C{i}', f'Benign Agent {i+1}')
        benign_avgs.append(np.mean(benign_agent_rewards[:, i]))
    
    # Process adversarial results if available
    if adversarial_results:
        # Limit to first 50 episodes
        adv_episodes = [r['episode'] for r in adversarial_results[:50]]
        # Extract agent rewards from the adversarial structure
        adv_agent_rewards = []
        for r in adversarial_results[:50]:
            episode_rewards = [0] * 3  # Initialize with 0 for each agent
            for step_rewards in r['agent_rewards']:
                for i, reward in enumerate(step_rewards):
                    episode_rewards[i] += reward
            adv_agent_rewards.append(episode_rewards)
        adv_agent_rewards = np.array(adv_agent_rewards)
        
        adv_avgs = []
        for i in range(adv_agent_rewards.shape[1]):
            plt.plot(adv_episodes, adv_agent_rewards[:, i], 
                    label=f'Adversarial Agent {i+1}', color=f'C{i}', linestyle='--')
            slope = add_trend_line(adv_episodes, adv_agent_rewards[:, i], f'C{i}', f'Adv Agent {i+1}')
            adv_avgs.append(np.mean(adv_agent_rewards[:, i]))
        
        # Add average reward comparison text
        avg_text = "Average Rewards:\n"
        for i in range(len(benign_avgs)):
            avg_text += f"Agent {i+1}: Benign={benign_avgs[i]:.2f}, Adv={adv_avgs[i]:.2f}\n"
        plt.text(0.02, 0.98, avg_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Agent Reward')
    plt.title('Total Reward per Agent')
    plt.legend()
    plt.grid(True)
    plt.savefig('agent_rewards_comparison.png')
    plt.close()

def main():
    # Load benign results
    benign_path = Path('results/benign/results.json')
    benign_results = load_results(benign_path)
    
    # Try to load adversarial results
    adv_path = Path('result/MultiAgentPoison/results.json')
    adversarial_results = None
    
    if adv_path.exists():
        adversarial_results = load_results(adv_path)
        print(f"Found adversarial results in {adv_path}")
    else:
        # Fallback to other locations
        possible_adv_paths = [
            Path('results/ad/results.json'),
            Path('results/myagent/results.json'),
            Path('results/ad/ap/results.json')
        ]
        
        for path in possible_adv_paths:
            if path.exists():
                adversarial_results = load_results(path)
                print(f"Found adversarial results in {path}")
                break
    
    # Create plots
    plot_rewards(benign_results, adversarial_results)
    plot_steps(benign_results, adversarial_results)
    plot_agent_rewards(benign_results, adversarial_results)

if __name__ == '__main__':
    main() 