"""
Smart Thermostat: EXTREME ASYMMETRY Scenario

This scenario has VERY asymmetric rewards where being too HOT
is catastrophically worse than being too COLD.

The homeostatic agent treats hot/cold symmetrically with (temp-target)²,
so it cannot properly avoid the dangerous "too hot" zone.

TD Learning can learn this asymmetry from experience.
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

RANDOM_SEED = 42


class TDValueFunction:
    def __init__(self, temp_min=5.0, temp_max=45.0, num_bins=80, 
                 learning_rate=0.2, discount_factor=0.9):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.values = np.zeros(num_bins)
        self.eligibility = np.zeros(num_bins)
        self.lambda_trace = 0.9
        
    def temp_to_bin(self, temp: float) -> int:
        temp = np.clip(temp, self.temp_min, self.temp_max)
        bin_idx = int((temp - self.temp_min) / (self.temp_max - self.temp_min) * (self.num_bins - 1))
        return np.clip(bin_idx, 0, self.num_bins - 1)
    
    def get_value(self, temp: float) -> float:
        return self.values[self.temp_to_bin(temp)]
    
    def get_value_interpolated(self, temp: float) -> float:
        temp = np.clip(temp, self.temp_min, self.temp_max)
        frac_pos = (temp - self.temp_min) / (self.temp_max - self.temp_min) * (self.num_bins - 1)
        lower_bin = int(frac_pos)
        upper_bin = min(lower_bin + 1, self.num_bins - 1)
        frac = frac_pos - lower_bin
        return (1 - frac) * self.values[lower_bin] + frac * self.values[upper_bin]
    
    def update_with_eligibility(self, current_temp: float, reward: float, next_temp: float) -> float:
        current_bin = self.temp_to_bin(current_temp)
        next_bin = self.temp_to_bin(next_temp)
        td_error = reward + self.discount_factor * self.values[next_bin] - self.values[current_bin]
        self.eligibility *= self.discount_factor * self.lambda_trace
        self.eligibility[current_bin] += 1.0
        self.values += self.learning_rate * td_error * self.eligibility
        return td_error


class AsymmetricRewardCalculator:
    """
    EXTREME asymmetric reward structure:
    - Too HOT (above target): SEVERE exponential penalty
    - Too COLD (below target): Mild linear penalty
    - In comfort zone: Good reward
    
    This asymmetry cannot be captured by (temp - target)²!
    """
    
    def __init__(self):
        self.comfort_threshold = 2.0
        
        # ASYMMETRIC penalties - Hot is MUCH worse
        self.hot_penalty_base = 0.3
        self.hot_penalty_exp_scale = 0.15  # Exponential growth for hot
        self.cold_penalty_linear = 0.08    # Linear, mild penalty for cold
        
        # Rewards
        self.comfort_reward = 0.4
        
        # Tracking
        self.reset()
    
    def reset(self):
        self.steps_too_hot = 0
        self.steps_too_cold = 0
        self.steps_in_comfort = 0
        self.total_hot_penalty = 0
        self.total_cold_penalty = 0
        self.total_comfort_reward = 0
    
    def calculate_reward(self, temp: float, target: float, heater_on: bool) -> tuple[float, dict]:
        """
        Calculate asymmetric reward.
        
        For temp > target: penalty = base + exp(scale * distance) - grows FAST
        For temp < target: penalty = linear * distance - grows slowly
        """
        distance = temp - target
        abs_dist = abs(distance)
        
        details = {"in_comfort": False, "distance": distance}
        
        if abs_dist <= self.comfort_threshold:
            # In comfort zone
            reward = self.comfort_reward
            self.steps_in_comfort += 1
            self.total_comfort_reward += reward
            details["in_comfort"] = True
            details["type"] = "comfort"
        elif distance > 0:
            # TOO HOT - SEVERE exponential penalty
            excess = distance - self.comfort_threshold
            penalty = self.hot_penalty_base + (np.exp(self.hot_penalty_exp_scale * excess) - 1)
            penalty = min(penalty, 3.0)  # Cap at 3.0
            reward = -penalty
            self.steps_too_hot += 1
            self.total_hot_penalty += penalty
            details["type"] = "hot"
        else:
            # TOO COLD - mild linear penalty
            excess = abs_dist - self.comfort_threshold
            penalty = self.cold_penalty_linear * excess
            reward = -penalty
            self.steps_too_cold += 1
            self.total_cold_penalty += penalty
            details["type"] = "cold"
        
        # Small heater cost
        if heater_on:
            reward -= 0.05
        
        return reward, details
    
    def get_expected_reward_asymmetric(self, pred_temp: float, target: float, heater_on: bool) -> float:
        """Expected reward that accounts for asymmetry."""
        distance = pred_temp - target
        abs_dist = abs(distance)
        
        if abs_dist <= self.comfort_threshold:
            expected = self.comfort_reward
        elif distance > 0:
            # Too hot - exponential penalty
            excess = distance - self.comfort_threshold
            expected = -(self.hot_penalty_base + (np.exp(self.hot_penalty_exp_scale * excess) - 1))
        else:
            # Too cold - linear penalty
            excess = abs_dist - self.comfort_threshold
            expected = -(self.cold_penalty_linear * excess)
        
        if heater_on:
            expected -= 0.05
        
        return expected


class HomeostaticAgent:
    """Agent using symmetric (temp - target)² - cannot handle asymmetry."""
    
    def __init__(self, initial_budget, reward_calc):
        self.name = "Homeostatic"
        self.TARGETS = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        self.budget = initial_budget
        self.reward_calc = reward_calc
        
        # Kalman state
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.obs_precision = 16.0
        self.proc_precision = 4.0
    
    def predict(self, action, dt, outside_temp):
        A_dt = np.array([[1.0, dt], [0.0, 0.95]])
        control = np.array([self.B_heater * dt if action else 0.0, 0.0])
        pred_mean = A_dt @ self.state_mean + control
        pred_mean[0] += self.B_outside * dt * (outside_temp - self.state_mean[0])
        proc_noise = np.eye(2) / self.proc_precision
        pred_cov = A_dt @ self.state_cov @ A_dt.T + proc_noise
        return pred_mean, pred_cov
    
    def update(self, obs, pred_mean, pred_cov):
        innov = obs - (self.H @ pred_mean)[0]
        S = (self.H @ pred_cov @ self.H.T)[0, 0] + 1.0 / self.obs_precision
        K = (pred_cov @ self.H.T) / S
        K = K.reshape(-1, 1)
        self.state_mean = pred_mean + (K * innov).flatten()
        self.state_cov = pred_cov - K @ self.H @ pred_cov
    
    def select_action(self, target, outside_temp, dt):
        """HOMEOSTATIC: Uses symmetric (temp - target)²."""
        actions = [False, True]
        efes = []
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            pred_temp = pred_mean[0]
            
            # SYMMETRIC pragmatic cost - treats hot/cold equally!
            pragmatic_cost = (pred_temp - target) ** 2
            epistemic = -0.5 * np.log(pred_cov[0, 0] + 1e-6)
            efe = -epistemic + pragmatic_cost
            efes.append(efe)
        
        return actions[np.argmin(efes)]
    
    def step(self, obs, heater_on, season, outside_temp, dt):
        pred_mean, pred_cov = self.predict(heater_on, dt, outside_temp)
        self.update(obs, pred_mean, pred_cov)
        
        target = self.TARGETS[season]
        new_action = self.select_action(target, outside_temp, dt)
        
        reward, details = self.reward_calc.calculate_reward(obs, target, heater_on)
        self.budget += reward - 0.02  # Small step cost
        
        return {"action": new_action, "dt": 1.0, "reward": reward, "details": details,
                "budget_exhausted": self.budget <= 0}


class ExtrinsicAgent:
    """Agent using TD Learning - can learn asymmetry."""
    
    def __init__(self, initial_budget, reward_calc):
        self.name = "Extrinsic (TD)"
        self.TARGETS = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        self.budget = initial_budget
        self.reward_calc = reward_calc
        
        self.value_func = TDValueFunction(learning_rate=0.2, discount_factor=0.9)
        self.prev_temp = None
        self.prev_reward = None
        
        # Kalman state
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.obs_precision = 16.0
        self.proc_precision = 4.0
    
    def predict(self, action, dt, outside_temp):
        A_dt = np.array([[1.0, dt], [0.0, 0.95]])
        control = np.array([self.B_heater * dt if action else 0.0, 0.0])
        pred_mean = A_dt @ self.state_mean + control
        pred_mean[0] += self.B_outside * dt * (outside_temp - self.state_mean[0])
        proc_noise = np.eye(2) / self.proc_precision
        pred_cov = A_dt @ self.state_cov @ A_dt.T + proc_noise
        return pred_mean, pred_cov
    
    def update(self, obs, pred_mean, pred_cov):
        innov = obs - (self.H @ pred_mean)[0]
        S = (self.H @ pred_cov @ self.H.T)[0, 0] + 1.0 / self.obs_precision
        K = (pred_cov @ self.H.T) / S
        K = K.reshape(-1, 1)
        self.state_mean = pred_mean + (K * innov).flatten()
        self.state_cov = pred_cov - K @ self.H @ pred_cov
    
    def select_action(self, target, outside_temp, dt):
        """EXTRINSIC: Uses TD-learned value + asymmetric expected reward."""
        actions = [False, True]
        efes = []
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            pred_temp = pred_mean[0]
            
            # TD-learned value
            learned_v = self.value_func.get_value_interpolated(pred_temp)
            
            # Expected reward that knows about asymmetry
            expected_r = self.reward_calc.get_expected_reward_asymmetric(pred_temp, target, action)
            
            # Combine
            pragmatic = 0.6 * learned_v + 0.4 * expected_r
            epistemic = -0.5 * np.log(pred_cov[0, 0] + 1e-6)
            efe = -epistemic - pragmatic
            efes.append(efe)
        
        return actions[np.argmin(efes)]
    
    def step(self, obs, heater_on, season, outside_temp, dt):
        pred_mean, pred_cov = self.predict(heater_on, dt, outside_temp)
        self.update(obs, pred_mean, pred_cov)
        
        target = self.TARGETS[season]
        new_action = self.select_action(target, outside_temp, dt)
        
        reward, details = self.reward_calc.calculate_reward(obs, target, heater_on)
        
        # TD Update
        if self.prev_temp is not None:
            self.value_func.update_with_eligibility(self.prev_temp, self.prev_reward, obs)
        self.prev_temp = obs
        self.prev_reward = reward
        
        self.budget += reward - 0.02
        
        return {"action": new_action, "dt": 1.0, "reward": reward, "details": details,
                "budget_exhausted": self.budget <= 0}


class Environment:
    def __init__(self, seed=42, duration=3600):
        random.seed(seed)
        np.random.seed(seed)
        self.duration = duration
        self.heating_power = 1.5
        
        self.SEASONS = {"Inverno": 0, "Primavera": 15, "Estate": 38, "Autunno": 12}
        self.TARGETS = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        self.season_order = ["Inverno", "Primavera", "Estate", "Autunno"]
        
        self.noise = [random.uniform(-2, 2) for _ in range(5000)]
    
    def get_season(self, t):
        return self.season_order[int(t // 90) % 4]
    
    def get_outside(self, t, step):
        return self.SEASONS[self.get_season(t)] + self.noise[step % len(self.noise)]
    
    def physics(self, temp, heater, outside, dt):
        heat = self.heating_power * dt if heater else 0
        loss = (outside - temp) * 0.1 * dt
        return temp + heat + loss


def run(agent, env, initial_temp, reward_calc):
    history = {"time": [], "temp": [], "target": [], "budget": [], 
               "reward": [], "type": [], "cumulative": []}
    
    reward_calc.reset()
    temp = initial_temp
    heater = False
    t = 0
    step = 0
    cumul = 0
    dt = 1.0
    
    while t < env.duration:
        season = env.get_season(t)
        outside = env.get_outside(t, step)
        target = env.TARGETS[season]
        
        result = agent.step(temp, heater, season, outside, dt)
        
        if result["budget_exhausted"]:
            break
        
        cumul += result["reward"]
        
        history["time"].append(t)
        history["temp"].append(temp)
        history["target"].append(target)
        history["budget"].append(agent.budget)
        history["reward"].append(result["reward"])
        history["type"].append(result["details"].get("type", "unknown"))
        history["cumulative"].append(cumul)
        
        heater = result["action"]
        dt = result["dt"]
        temp = env.physics(temp, heater, outside, dt)
        t += dt
        step += 1
    
    total = len(history["time"])
    comfort_pct = len([x for x in history["type"] if x == "comfort"]) / total * 100 if total > 0 else 0
    hot_pct = len([x for x in history["type"] if x == "hot"]) / total * 100 if total > 0 else 0
    cold_pct = len([x for x in history["type"] if x == "cold"]) / total * 100 if total > 0 else 0
    
    return history, {
        "name": agent.name,
        "survived": t,
        "steps": total,
        "final_budget": agent.budget,
        "cumulative_reward": cumul,
        "comfort_pct": comfort_pct,
        "hot_pct": hot_pct,
        "cold_pct": cold_pct,
        "hot_penalty": reward_calc.total_hot_penalty,
        "cold_penalty": reward_calc.total_cold_penalty,
        "comfort_reward": reward_calc.total_comfort_reward,
    }


def plot_results(h_hist, e_hist, h_sum, e_sum):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ASYMMETRIC REWARDS: Hot >> Cold Penalty", fontsize=14, fontweight='bold')
    
    # Temperature with zones highlighted
    ax = axes[0, 0]
    ax.plot(h_hist["time"], h_hist["temp"], 'b-', label="Homeostatic", alpha=0.8)
    ax.plot(h_hist["time"], h_hist["target"], 'g--', label="Target", alpha=0.5)
    # Highlight hot zone (target + 2)
    targets = h_hist["target"]
    if targets:
        ax.axhline(y=max(targets) + 2, color='red', linestyle=':', alpha=0.7, label="Hot zone")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"HOMEOSTATIC - Hot: {h_sum['hot_pct']:.1f}%, Cold: {h_sum['cold_pct']:.1f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(e_hist["time"], e_hist["temp"], color='orange', label="Extrinsic", alpha=0.8)
    ax.plot(e_hist["time"], e_hist["target"], 'g--', label="Target", alpha=0.5)
    targets = e_hist["target"]
    if targets:
        ax.axhline(y=max(targets) + 2, color='red', linestyle=':', alpha=0.7, label="Hot zone")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"EXTRINSIC (TD) - Hot: {e_sum['hot_pct']:.1f}%, Cold: {e_sum['cold_pct']:.1f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative reward
    ax = axes[1, 0]
    min_len = min(len(h_hist["time"]), len(e_hist["time"]))
    ax.plot(h_hist["time"][:min_len], h_hist["cumulative"][:min_len], 'b-', 
            label=f"Homeo: {h_sum['cumulative_reward']:.1f}", linewidth=2)
    ax.plot(e_hist["time"][:min_len], e_hist["cumulative"][:min_len], color='orange', 
            label=f"Extrin: {e_sum['cumulative_reward']:.1f}", linewidth=2)
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    winner = "EXTRINSIC (TD)" if e_sum['cumulative_reward'] > h_sum['cumulative_reward'] else "HOMEOSTATIC"
    
    text = f"""
ASYMMETRIC REWARD RESULTS
═══════════════════════════════════════════════
                     HOMEOSTATIC   EXTRINSIC(TD)
───────────────────────────────────────────────
Survived Time         {h_sum['survived']:>8.0f}s    {e_sum['survived']:>8.0f}s
Final Budget         €{h_sum['final_budget']:>8.1f}   €{e_sum['final_budget']:>8.1f}

ZONE BREAKDOWN:
  Comfort Zone        {h_sum['comfort_pct']:>8.1f}%    {e_sum['comfort_pct']:>8.1f}%
  TOO HOT (severe)    {h_sum['hot_pct']:>8.1f}%    {e_sum['hot_pct']:>8.1f}%
  Too Cold (mild)     {h_sum['cold_pct']:>8.1f}%    {e_sum['cold_pct']:>8.1f}%

PENALTIES/REWARDS:
  Hot Penalty (exp)   {h_sum['hot_penalty']:>8.1f}     {e_sum['hot_penalty']:>8.1f}
  Cold Penalty (lin)  {h_sum['cold_penalty']:>8.1f}     {e_sum['cold_penalty']:>8.1f}
  Comfort Reward      {h_sum['comfort_reward']:>8.1f}     {e_sum['comfort_reward']:>8.1f}

CUMULATIVE REWARD     {h_sum['cumulative_reward']:>8.1f}     {e_sum['cumulative_reward']:>8.1f}
═══════════════════════════════════════════════

WINNER: {winner}
    """
    ax.text(0.1, 0.5, text, fontsize=10, fontfamily='monospace', 
            transform=ax.transAxes, va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    filename = f"asymmetric_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    return filename


def main():
    print("=" * 60)
    print("  ASYMMETRIC REWARD: Hot is MUCH worse than Cold")
    print("=" * 60)
    print()
    print("  Reward Structure:")
    print("  - Comfort zone (±2°C):     +0.4")
    print("  - Too HOT:  EXPONENTIAL penalty (grows fast!)")
    print("  - Too Cold: Linear penalty (grows slowly)")
    print()
    print("  The homeostatic agent uses (temp-target)² which treats")
    print("  hot and cold EQUALLY - this is a big disadvantage!")
    print()
    
    DURATION = 3600
    BUDGET = 200.0
    INIT_TEMP = 20.0
    
    print(f"[Config] Duration: {DURATION}s, Budget: €{BUDGET}\n")
    
    # Homeostatic
    print("[1/2] Running HOMEOSTATIC...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = Environment(RANDOM_SEED, DURATION)
    h_calc = AsymmetricRewardCalculator()
    h_agent = HomeostaticAgent(BUDGET, h_calc)
    h_hist, h_sum = run(h_agent, env, INIT_TEMP, h_calc)
    print(f"       Survived: {h_sum['survived']:.0f}s, Comfort: {h_sum['comfort_pct']:.1f}%, "
          f"Hot: {h_sum['hot_pct']:.1f}%, Reward: {h_sum['cumulative_reward']:.1f}")
    
    # Extrinsic
    print("[2/2] Running EXTRINSIC (TD)...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = Environment(RANDOM_SEED, DURATION)
    e_calc = AsymmetricRewardCalculator()
    e_agent = ExtrinsicAgent(BUDGET, e_calc)
    e_hist, e_sum = run(e_agent, env, INIT_TEMP, e_calc)
    print(f"       Survived: {e_sum['survived']:.0f}s, Comfort: {e_sum['comfort_pct']:.1f}%, "
          f"Hot: {e_sum['hot_pct']:.1f}%, Reward: {e_sum['cumulative_reward']:.1f}")
    
    # Plot
    print("\n[Plot] Generating comparison...")
    plot_results(h_hist, e_hist, h_sum, e_sum)
    
    # Results
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    
    if e_sum['cumulative_reward'] > h_sum['cumulative_reward']:
        print(f"\n  *** EXTRINSIC (TD) WINS! ***")
        print(f"  Reward: {e_sum['cumulative_reward']:.1f} vs {h_sum['cumulative_reward']:.1f}")
        print(f"  TD learned to avoid the expensive 'too hot' zone!")
    else:
        print(f"\n  HOMEOSTATIC WINS")
        print(f"  Reward: {h_sum['cumulative_reward']:.1f} vs {e_sum['cumulative_reward']:.1f}")
    
    print()
    print(f"  Key insight: Hot zone penalty comparison")
    print(f"    Homeostatic accumulated:  {h_sum['hot_penalty']:.1f} hot penalty")
    print(f"    Extrinsic accumulated:    {e_sum['hot_penalty']:.1f} hot penalty")
    
    if e_sum['hot_penalty'] < h_sum['hot_penalty']:
        print(f"    -> TD learned to AVOID hot zone better!")
    
    print()


if __name__ == "__main__":
    main()
