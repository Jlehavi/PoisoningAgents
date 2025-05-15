import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

def get_latest_result_dir(base_dir):
    """Get the most recent results directory."""
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    return os.path.join(base_dir, sorted(dirs)[-1])

def load_results(results_dir):
    """Load results from the specified directory."""
    # First try to load from the directory itself
    results_file = os.path.join(results_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    # If not found, look in the most recent subdirectory
    latest_dir = get_latest_result_dir(results_dir)
    if latest_dir:
        results_file = os.path.join(latest_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"No results.json found in {results_dir} or its subdirectories")

def analyze_results(results):
    """Extract metrics from results."""
    total_rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]
    agent_rewards = [sum(rewards)/len(rewards) for r in results for rewards in r['agent_rewards']]
    
    return {
        'total_rewards': total_rewards,
        'steps': steps,
        'agent_rewards': agent_rewards,
        'total_reward_mean': np.mean(total_rewards),
        'total_reward_std': np.std(total_rewards),
        'steps_mean': np.mean(steps),
        'steps_std': np.std(steps),
        'agent_reward_mean': np.mean(agent_rewards),
        'agent_reward_std': np.std(agent_rewards)
    }

def create_comparison_plots(benign_results, adv_results, save_dir):
    """Create comparison plots between benign and adversarial results."""
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Total Rewards Distribution
    sns.boxplot(data=[benign_results['total_rewards'], adv_results['total_rewards']], 
                ax=ax1)
    ax1.set_xticklabels(['Benign', 'Adversarial'])
    ax1.set_title('Distribution of Total Rewards')
    ax1.set_ylabel('Total Reward')
    
    # Plot 2: Steps Distribution
    sns.boxplot(data=[benign_results['steps'], adv_results['steps']], 
                ax=ax2)
    ax2.set_xticklabels(['Benign', 'Adversarial'])
    ax2.set_title('Distribution of Steps per Episode')
    ax2.set_ylabel('Steps')
    
    # Plot 3: Agent Rewards Distribution
    sns.boxplot(data=[benign_results['agent_rewards'], adv_results['agent_rewards']], 
                ax=ax3)
    ax3.set_xticklabels(['Benign', 'Adversarial'])
    ax3.set_title('Distribution of Agent Rewards')
    ax3.set_ylabel('Agent Reward')
    
    # Add statistical test results
    for ax, benign_data, adv_data, metric in zip(
        [ax1, ax2, ax3],
        [benign_results['total_rewards'], benign_results['steps'], benign_results['agent_rewards']],
        [adv_results['total_rewards'], adv_results['steps'], adv_results['agent_rewards']],
        ['Total Rewards', 'Steps', 'Agent Rewards']
    ):
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(benign_data, adv_data)
        ax.text(0.5, -0.15, f'p-value: {p_val:.3f}', 
                horizontalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benign_vs_adversarial_comparison.png'))
    print(f"Saved comparison plot to {os.path.join(save_dir, 'benign_vs_adversarial_comparison.png')}")
    plt.close()

def main():
    # Define paths
    base_dir = "./results/myagent"
    benign_dir = os.path.join(base_dir, "benign")
    adv_dir = os.path.join(base_dir, "ap")
    
    try:
        # Load and analyze results
        print("Loading benign results...")
        benign_results = analyze_results(load_results(benign_dir))
        print("Loading adversarial results...")
        adv_results = analyze_results(load_results(adv_dir))
        
        # Create results directory if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        
        # Create comparison plots
        print("Creating comparison plots...")
        create_comparison_plots(benign_results, adv_results, "./results")
        
        # Print summary statistics
        print("\nBenign Results:")
        print(f"Average Total Reward: {benign_results['total_reward_mean']:.2f} ± {benign_results['total_reward_std']:.2f}")
        print(f"Average Steps: {benign_results['steps_mean']:.2f} ± {benign_results['steps_std']:.2f}")
        print(f"Average Agent Reward: {benign_results['agent_reward_mean']:.2f} ± {benign_results['agent_reward_std']:.2f}")
        
        print("\nAdversarial Results:")
        print(f"Average Total Reward: {adv_results['total_reward_mean']:.2f} ± {adv_results['total_reward_std']:.2f}")
        print(f"Average Steps: {adv_results['steps_mean']:.2f} ± {adv_results['steps_std']:.2f}")
        print(f"Average Agent Reward: {adv_results['agent_reward_mean']:.2f} ± {adv_results['agent_reward_std']:.2f}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have both benign and adversarial results in the following directories:")
        print(f"Benign: {benign_dir}")
        print(f"Adversarial: {adv_dir}")

if __name__ == "__main__":
    main() 