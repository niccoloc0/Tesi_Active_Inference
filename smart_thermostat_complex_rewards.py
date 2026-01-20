"""
Smart Thermostat Comparison: Complex Reward Scenario

This scenario introduces ASYMMETRIC and NON-LINEAR rewards that
favor TD Learning over the simple homeostatic approach.

Complex Reward Structure:
1. ASYMMETRIC: Too hot (-0.8) is worse than too cold (-0.3)
2. EFFICIENCY BONUS: Not using heater when not needed = +0.2
3. SWITCHING PENALTY: Turning heater on/off frequently = -0.15
4. STABILITY BONUS: 5+ consecutive steps in comfort = +0.5 cumulative
5. EXTREME PENALTY: Very far from target (>5°C) = -1.5

The homeostatic (temp - target)² cannot capture these patterns,
but TD can learn them from experience.
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

RANDOM_SEED = 42


# ============================================================================
# TD VALUE FUNCTION (Enhanced)
# ============================================================================

class TDValueFunction:
    """Tabular Value Function with TD Learning."""
    
    def __init__(self, temp_min=10.0, temp_max=40.0, num_bins=60, 
                 learning_rate=0.15, discount_factor=0.95):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.values = np.zeros(num_bins)
        self.eligibility = np.zeros(num_bins)
        self.lambda_trace = 0.9
        self.td_errors = []
        
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
        self.td_errors.append(td_error)
        return td_error


# ============================================================================
# COMPLEX REWARD CALCULATOR
# ============================================================================

class ComplexRewardCalculator:
    """
    Calculates complex, non-linear rewards that favor TD learning.
    
    The homeostatic agent cannot capture these patterns because:
    1. Asymmetry: (temp - target)² treats hot and cold equally
    2. History: Switching penalty requires memory of previous action
    3. Cumulative: Stability bonus requires counting consecutive steps
    4. Non-linear: Extreme penalty is not quadratic
    """
    
    def __init__(self):
        # Base comfort threshold
        self.comfort_threshold = 2.0
        
        # Asymmetric penalties
        self.penalty_too_hot = 0.8      # Much worse to be too hot
        self.penalty_too_cold = 0.3     # Less bad to be too cold
        self.penalty_extreme = 1.5      # Very far from target (>5°C)
        self.extreme_threshold = 5.0
        
        # Base reward
        self.reward_comfort = 0.3
        
        # Efficiency bonus (not using heater when temperature is fine)
        self.efficiency_bonus = 0.2
        
        # Switching penalty
        self.switching_penalty = 0.15
        
        # Stability bonus (consecutive steps in comfort)
        self.stability_bonus = 0.1
        self.stability_threshold = 5  # Steps needed for bonus
        
        # Heater cost
        self.heater_cost = 0.2
        
        # Tracking for history-dependent rewards
        self.consecutive_comfort_steps = 0
        self.prev_heater_state = None
        self.switch_count = 0
        
        # Statistics
        self.total_base_rewards = 0
        self.total_efficiency_bonus = 0
        self.total_stability_bonus = 0
        self.total_switching_penalties = 0
        self.total_asymmetric_penalties = 0
    
    def reset_tracking(self):
        """Reset for new simulation."""
        self.consecutive_comfort_steps = 0
        self.prev_heater_state = None
        self.switch_count = 0
        self.total_base_rewards = 0
        self.total_efficiency_bonus = 0
        self.total_stability_bonus = 0
        self.total_switching_penalties = 0
        self.total_asymmetric_penalties = 0
    
    def calculate_reward(self, observation: float, target_temp: float, 
                        heater_on: bool, outside_temp: float) -> tuple[float, dict]:
        """
        Calculate complex reward based on multiple factors.
        
        Returns: (total_reward, details_dict)
        """
        total_reward = 0.0
        details = {
            "base": 0, "asymmetric": 0, "efficiency": 0, 
            "switching": 0, "stability": 0, "heater_cost": 0,
            "in_comfort": False, "distance": 0
        }
        
        # Distance from target
        distance = observation - target_temp
        abs_distance = abs(distance)
        details["distance"] = distance
        
        # Check comfort zone
        in_comfort = abs_distance <= self.comfort_threshold
        details["in_comfort"] = in_comfort
        
        # 1. BASE REWARD/PENALTY (asymmetric)
        if in_comfort:
            # In comfort zone - base reward
            base_reward = self.reward_comfort
            self.total_base_rewards += base_reward
            details["base"] = base_reward
            total_reward += base_reward
            
            # Update consecutive comfort counter
            self.consecutive_comfort_steps += 1
        else:
            # Outside comfort zone - ASYMMETRIC penalty
            if distance > 0:
                # TOO HOT - severe penalty
                if abs_distance > self.extreme_threshold:
                    penalty = self.penalty_extreme
                else:
                    # Scale penalty with distance
                    penalty = self.penalty_too_hot * (abs_distance / self.comfort_threshold)
            else:
                # TOO COLD - lighter penalty
                if abs_distance > self.extreme_threshold:
                    penalty = self.penalty_extreme * 0.7  # Still less than too hot
                else:
                    penalty = self.penalty_too_cold * (abs_distance / self.comfort_threshold)
            
            self.total_asymmetric_penalties += penalty
            details["asymmetric"] = -penalty
            total_reward -= penalty
            
            # Reset consecutive comfort counter
            self.consecutive_comfort_steps = 0
        
        # 2. EFFICIENCY BONUS (not using heater when not needed)
        # Bonus if heater is OFF and we're still in comfort
        if in_comfort and not heater_on:
            efficiency = self.efficiency_bonus
            self.total_efficiency_bonus += efficiency
            details["efficiency"] = efficiency
            total_reward += efficiency
        
        # 3. SWITCHING PENALTY (discourage frequent on/off)
        if self.prev_heater_state is not None:
            if heater_on != self.prev_heater_state:
                self.switch_count += 1
                self.total_switching_penalties += self.switching_penalty
                details["switching"] = -self.switching_penalty
                total_reward -= self.switching_penalty
        
        self.prev_heater_state = heater_on
        
        # 4. STABILITY BONUS (consecutive comfort steps)
        if self.consecutive_comfort_steps >= self.stability_threshold:
            # Bonus increases with consecutive steps (cumulative)
            stability = self.stability_bonus
            self.total_stability_bonus += stability
            details["stability"] = stability
            total_reward += stability
        
        # 5. HEATER COST (if on)
        if heater_on:
            details["heater_cost"] = -self.heater_cost
            total_reward -= self.heater_cost
        
        return total_reward, details
    
    def get_expected_reward(self, pred_temp: float, target_temp: float,
                           pred_heater: bool) -> float:
        """Estimate expected reward for action selection (simplified)."""
        distance = pred_temp - target_temp
        abs_distance = abs(distance)
        
        if abs_distance <= self.comfort_threshold:
            expected = self.reward_comfort
            if not pred_heater:
                expected += self.efficiency_bonus
        else:
            if distance > 0:
                expected = -self.penalty_too_hot * (abs_distance / self.comfort_threshold)
            else:
                expected = -self.penalty_too_cold * (abs_distance / self.comfort_threshold)
        
        if pred_heater:
            expected -= self.heater_cost
        
        return expected


# ============================================================================
# HOMEOSTATIC AGENT (with complex rewards)
# ============================================================================

class HomeostaticAgentComplex:
    """
    Homeostatic agent that still uses (temp - target)² for decisions,
    but receives complex rewards it cannot optimize for directly.
    """
    
    def __init__(self, initial_budget, reward_calculator):
        self.name = "Homeostatic"
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        self.budget = initial_budget
        self.initial_budget = initial_budget
        self.cost_per_step = 0.05
        self.reward_calculator = reward_calculator
        
        # Tracking
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        
        # Kalman Filter State
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.observation_precision = 16.0
        self.process_precision = 4.0
        self.dynamics_learning_rate = 0.01
        
        # History
        self.prediction_errors = []
        self.epistemic_values = []
        self.pragmatic_values = []
        
        # Epistemic parameters
        self.min_dt = 0.5
        self.max_dt = 3.0
        self.uncertainty_threshold = 2.0
    
    def predict(self, action, dt, outside_temp):
        A_dt = np.array([[1.0, dt], [0.0, 0.95]])
        control_effect = np.array([self.B_heater * dt if action else 0.0, 0.0])
        outside_effect = np.array([self.B_outside * dt, 0.0])
        predicted_mean = A_dt @ self.state_mean + control_effect
        predicted_mean[0] += outside_effect[0] * (outside_temp - self.state_mean[0])
        process_noise = np.eye(2) / self.process_precision
        predicted_cov = A_dt @ self.state_cov @ A_dt.T + process_noise
        return predicted_mean, predicted_cov
    
    def update(self, observation, predicted_mean, predicted_cov):
        innovation = observation - (self.H @ predicted_mean)[0]
        observation_noise = 1.0 / self.observation_precision
        S = (self.H @ predicted_cov @ self.H.T)[0, 0] + observation_noise
        K = (predicted_cov @ self.H.T) / S
        K = K.reshape(-1, 1)
        self.state_mean = predicted_mean + (K * innovation).flatten()
        self.state_cov = predicted_cov - K @ self.H @ predicted_cov
        free_energy = 0.5 * (innovation**2 * self.observation_precision + 
                            np.log(2 * np.pi / self.observation_precision))
        return innovation, free_energy
    
    def compute_efe_homeostatic(self, action, target_temp, predicted_mean, predicted_cov):
        """HOMEOSTATIC: Still uses (temp - target)² - cannot adapt to complex rewards."""
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        predicted_temp = predicted_mean[0]
        
        # HOMEOSTATIC COST: Simple quadratic - treats hot/cold equally!
        pragmatic_cost = (predicted_temp - target_temp) ** 2
        pragmatic_value = -pragmatic_cost
        
        efe = -epistemic_value + pragmatic_cost
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        actions = [False, True]
        efes = []
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, _, _ = self.compute_efe_homeostatic(action, target_temp, pred_mean, pred_cov)
            efes.append(efe)
        best_idx = np.argmin(efes)
        return actions[best_idx]
    
    def select_sampling_rate(self):
        uncertainty = self.state_cov[0, 0]
        if uncertainty > self.uncertainty_threshold:
            return self.min_dt
        elif uncertainty > self.uncertainty_threshold / 2:
            return 1.0
        elif uncertainty > self.uncertainty_threshold / 4:
            return 2.0
        return self.max_dt
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        
        target_temp = self.SEASONS_TARGET[current_season]
        new_action = self.select_action(observation, target_temp, current_action, outside_temp, dt)
        new_dt = self.select_sampling_rate()
        
        # Get COMPLEX reward (but agent doesn't use it for decisions!)
        reward, details = self.reward_calculator.calculate_reward(
            observation, target_temp, current_action, outside_temp
        )
        
        # Track rewards
        if reward > 0:
            self.total_rewards += reward
        else:
            self.total_penalties += abs(reward)
        
        # Update budget
        net_change = reward - self.cost_per_step
        self.budget += net_change
        
        return {
            "action": new_action,
            "dt": new_dt,
            "reward": reward,
            "reward_details": details,
            "budget_exhausted": self.budget <= 0,
        }


# ============================================================================
# EXTRINSIC AGENT (TD Learning with Complex Rewards)
# ============================================================================

class ExtrinsicAgentComplex:
    """
    TD Learning agent that learns from complex reward signals.
    Can adapt to asymmetric and history-dependent rewards.
    """
    
    def __init__(self, initial_budget, reward_calculator, 
                 td_learning_rate=0.15, td_discount=0.95):
        self.name = "Extrinsic (TD)"
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        self.budget = initial_budget
        self.initial_budget = initial_budget
        self.cost_per_step = 0.05
        self.reward_calculator = reward_calculator
        
        # TD Value Function
        self.value_function = TDValueFunction(
            temp_min=10.0, temp_max=40.0, num_bins=60,
            learning_rate=td_learning_rate,
            discount_factor=td_discount
        )
        self.prev_temp = None
        self.prev_reward = None
        
        # Tracking
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        self.td_error_history = []
        
        # Kalman Filter State
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.observation_precision = 16.0
        self.process_precision = 4.0
        
        # Epistemic parameters
        self.min_dt = 0.5
        self.max_dt = 3.0
        self.uncertainty_threshold = 2.0
    
    def predict(self, action, dt, outside_temp):
        A_dt = np.array([[1.0, dt], [0.0, 0.95]])
        control_effect = np.array([self.B_heater * dt if action else 0.0, 0.0])
        outside_effect = np.array([self.B_outside * dt, 0.0])
        predicted_mean = A_dt @ self.state_mean + control_effect
        predicted_mean[0] += outside_effect[0] * (outside_temp - self.state_mean[0])
        process_noise = np.eye(2) / self.process_precision
        predicted_cov = A_dt @ self.state_cov @ A_dt.T + process_noise
        return predicted_mean, predicted_cov
    
    def update(self, observation, predicted_mean, predicted_cov):
        innovation = observation - (self.H @ predicted_mean)[0]
        observation_noise = 1.0 / self.observation_precision
        S = (self.H @ predicted_cov @ self.H.T)[0, 0] + observation_noise
        K = (predicted_cov @ self.H.T) / S
        K = K.reshape(-1, 1)
        self.state_mean = predicted_mean + (K * innovation).flatten()
        self.state_cov = predicted_cov - K @ self.H @ predicted_cov
        return innovation
    
    def compute_efe_extrinsic(self, action, target_temp, predicted_mean, predicted_cov):
        """EXTRINSIC: Uses learned value function + expected reward."""
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        predicted_temp = predicted_mean[0]
        
        # Learned value from TD
        learned_value = self.value_function.get_value_interpolated(predicted_temp)
        
        # Expected immediate reward (including asymmetry awareness)
        expected_reward = self.reward_calculator.get_expected_reward(
            predicted_temp, target_temp, action
        )
        
        # Combine learned + expected
        pragmatic_value = 0.6 * learned_value + 0.4 * expected_reward
        
        # EFE (minimize)
        efe = -epistemic_value - pragmatic_value
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        actions = [False, True]
        efes = []
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, _, _ = self.compute_efe_extrinsic(action, target_temp, pred_mean, pred_cov)
            efes.append(efe)
        best_idx = np.argmin(efes)
        return actions[best_idx]
    
    def select_sampling_rate(self):
        uncertainty = self.state_cov[0, 0]
        if uncertainty > self.uncertainty_threshold:
            return self.min_dt
        elif uncertainty > self.uncertainty_threshold / 2:
            return 1.0
        elif uncertainty > self.uncertainty_threshold / 4:
            return 2.0
        return self.max_dt
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        self.update(observation, predicted_mean, predicted_cov)
        
        target_temp = self.SEASONS_TARGET[current_season]
        new_action = self.select_action(observation, target_temp, current_action, outside_temp, dt)
        new_dt = self.select_sampling_rate()
        
        # Get COMPLEX reward (and USE IT for learning!)
        reward, details = self.reward_calculator.calculate_reward(
            observation, target_temp, current_action, outside_temp
        )
        
        # TD Update - Learn from complex reward
        td_error = 0.0
        if self.prev_temp is not None:
            td_error = self.value_function.update_with_eligibility(
                self.prev_temp, self.prev_reward, observation
            )
            self.td_error_history.append(td_error)
        
        self.prev_temp = observation
        self.prev_reward = reward
        
        # Track rewards
        if reward > 0:
            self.total_rewards += reward
        else:
            self.total_penalties += abs(reward)
        
        # Update budget
        net_change = reward - self.cost_per_step
        self.budget += net_change
        
        return {
            "action": new_action,
            "dt": new_dt,
            "reward": reward,
            "reward_details": details,
            "td_error": td_error,
            "budget_exhausted": self.budget <= 0,
        }


# ============================================================================
# ENVIRONMENT
# ============================================================================

class SharedEnvironment:
    def __init__(self, seed=42, simulation_duration=3600, heating_power=1.5, weather_fluctuation=2.0):
        random.seed(seed)
        np.random.seed(seed)
        
        self.simulation_duration = simulation_duration
        self.heating_power = heating_power
        self.weather_fluctuation = weather_fluctuation
        
        self.SEASONS = {
            "Inverno": {"outside_temp": 0}, "Primavera": {"outside_temp": 15},
            "Estate": {"outside_temp": 35}, "Autunno": {"outside_temp": 12}
        }
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        self.season_order = ["Inverno", "Primavera", "Estate", "Autunno"]
        
        max_steps = 5000
        self.weather_noise = [random.uniform(-self.weather_fluctuation, self.weather_fluctuation) 
                              for _ in range(max_steps)]
    
    def get_season(self, sim_time):
        season_idx = int(sim_time // 90) % 4  # 90s per season (more transitions)
        return self.season_order[season_idx]
    
    def get_outside_temp(self, sim_time, step):
        season = self.get_season(sim_time)
        base_temp = self.SEASONS[season]["outside_temp"]
        return base_temp + self.weather_noise[step % len(self.weather_noise)]
    
    def get_target_temp(self, sim_time):
        return self.SEASONS_TARGET[self.get_season(sim_time)]
    
    def physics_update(self, room_temp, heater_on, outside_temp, dt):
        heat_from_heater = (self.heating_power * dt) if heater_on else 0
        heat_loss_or_gain = (outside_temp - room_temp) * 0.1 * dt
        return room_temp + heat_from_heater + heat_loss_or_gain


def run_simulation(agent, env, initial_temp, reward_calc):
    """Run simulation for a single agent."""
    
    history = {
        "time": [], "room_temp": [], "target_temp": [], "budget": [],
        "reward": [], "in_comfort": [], "cumulative_reward": []
    }
    
    reward_calc.reset_tracking()
    
    room_temp = initial_temp
    heater_on = False
    current_dt = 1.0
    current_sim_time = 0
    step = 0
    cumulative_reward = 0
    
    while current_sim_time < env.simulation_duration:
        season = env.get_season(current_sim_time)
        outside_temp = env.get_outside_temp(current_sim_time, step)
        target_temp = env.get_target_temp(current_sim_time)
        
        result = agent.step(room_temp, heater_on, season, outside_temp, current_dt)
        
        if result["budget_exhausted"]:
            break
        
        cumulative_reward += result["reward"]
        
        history["time"].append(current_sim_time)
        history["room_temp"].append(room_temp)
        history["target_temp"].append(target_temp)
        history["budget"].append(agent.budget)
        history["reward"].append(result["reward"])
        history["in_comfort"].append(1 if result["reward_details"]["in_comfort"] else 0)
        history["cumulative_reward"].append(cumulative_reward)
        
        heater_on = result["action"]
        current_dt = result["dt"]
        room_temp = env.physics_update(room_temp, heater_on, outside_temp, current_dt)
        
        current_sim_time += current_dt
        step += 1
    
    total_steps = len(history["time"])
    comfort_pct = (sum(history["in_comfort"]) / total_steps * 100) if total_steps > 0 else 0
    
    summary = {
        "name": agent.name,
        "survived_time": current_sim_time,
        "total_steps": total_steps,
        "final_budget": agent.budget,
        "total_rewards": agent.total_rewards,
        "total_penalties": agent.total_penalties,
        "comfort_percentage": comfort_pct,
        "cumulative_reward": cumulative_reward,
        # Complex reward breakdowns
        "base_rewards": reward_calc.total_base_rewards,
        "efficiency_bonus": reward_calc.total_efficiency_bonus,
        "stability_bonus": reward_calc.total_stability_bonus,
        "switching_penalties": reward_calc.total_switching_penalties,
        "asymmetric_penalties": reward_calc.total_asymmetric_penalties,
    }
    
    return history, summary


def generate_plot(h_hist, e_hist, h_sum, e_sum):
    """Generate comparison plot."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("COMPLEX REWARDS: Homeostatic vs Extrinsic (TD Learning)", fontsize=16, fontweight='bold')
    
    colors = {"h": "blue", "e": "orange"}
    
    # Temperature
    ax = axes[0, 0]
    ax.plot(h_hist["time"], h_hist["room_temp"], color=colors["h"], label="Homeostatic", alpha=0.8)
    ax.plot(h_hist["time"], h_hist["target_temp"], 'g--', label="Target", alpha=0.5)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"HOMEOSTATIC - Temperature (Comfort: {h_sum['comfort_percentage']:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(e_hist["time"], e_hist["room_temp"], color=colors["e"], label="Extrinsic (TD)", alpha=0.8)
    ax.plot(e_hist["time"], e_hist["target_temp"], 'g--', label="Target", alpha=0.5)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"EXTRINSIC (TD) - Temperature (Comfort: {e_sum['comfort_percentage']:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Budget
    ax = axes[1, 0]
    ax.plot(h_hist["time"], h_hist["budget"], color=colors["h"], linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Budget (€)")
    ax.set_title(f"HOMEOSTATIC - Budget (Final: €{h_sum['final_budget']:.1f})")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(e_hist["time"], e_hist["budget"], color=colors["e"], linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Budget (€)")
    ax.set_title(f"EXTRINSIC (TD) - Budget (Final: €{e_sum['final_budget']:.1f})")
    ax.grid(True, alpha=0.3)
    
    # Cumulative Reward Comparison
    ax = axes[2, 0]
    min_len = min(len(h_hist["time"]), len(e_hist["time"]))
    ax.plot(h_hist["time"][:min_len], h_hist["cumulative_reward"][:min_len], 
            color=colors["h"], label=f"Homeo: {h_sum['cumulative_reward']:.1f}", linewidth=2)
    ax.plot(e_hist["time"][:min_len], e_hist["cumulative_reward"][:min_len], 
            color=colors["e"], label=f"Extrin: {e_sum['cumulative_reward']:.1f}", linewidth=2)
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary Table
    ax = axes[2, 1]
    ax.axis('off')
    
    winner = "EXTRINSIC (TD)" if e_sum['cumulative_reward'] > h_sum['cumulative_reward'] else "HOMEOSTATIC"
    winner_color = "orange" if winner == "EXTRINSIC (TD)" else "blue"
    
    summary_text = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║              COMPLEX REWARD COMPARISON RESULTS                 ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                           HOMEOSTATIC │ EXTRINSIC (TD)         ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Survived Time         {h_sum['survived_time']:>10.1f}s │ {e_sum['survived_time']:>10.1f}s          ║
    ║  Total Steps           {h_sum['total_steps']:>10d}  │ {e_sum['total_steps']:>10d}           ║
    ║  Final Budget         €{h_sum['final_budget']:>10.2f} │ €{e_sum['final_budget']:>10.2f}          ║
    ║  Comfort Zone         {h_sum['comfort_percentage']:>10.1f}% │ {e_sum['comfort_percentage']:>10.1f}%          ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  CUMULATIVE REWARD    {h_sum['cumulative_reward']:>10.1f}  │ {e_sum['cumulative_reward']:>10.1f}           ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  REWARD BREAKDOWN:                                             ║
    ║    Base Rewards       {h_sum['base_rewards']:>10.1f}  │ {e_sum['base_rewards']:>10.1f}           ║
    ║    Efficiency Bonus   {h_sum['efficiency_bonus']:>10.1f}  │ {e_sum['efficiency_bonus']:>10.1f}           ║
    ║    Stability Bonus    {h_sum['stability_bonus']:>10.1f}  │ {e_sum['stability_bonus']:>10.1f}           ║
    ║    Switching Penalty  {h_sum['switching_penalties']:>10.1f}  │ {e_sum['switching_penalties']:>10.1f}           ║
    ║    Asymmetric Penalty {h_sum['asymmetric_penalties']:>10.1f}  │ {e_sum['asymmetric_penalties']:>10.1f}           ║
    ╚════════════════════════════════════════════════════════════════╝
    
                    🏆 WINNER: {winner}
    """
    
    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_complex_rewards_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    print(f"[Plot] Saved to {filename}")
    
    return filename


def main():
    print("=" * 70)
    print("  COMPLEX REWARD SCENARIO")
    print("  Asymmetric + Efficiency + Switching + Stability Rewards")
    print("=" * 70)
    print()
    print("  This scenario introduces rewards that favor TD Learning:")
    print("  - Asymmetric penalty: Too hot (-0.8) >> Too cold (-0.3)")
    print("  - Efficiency bonus: +0.2 for comfort without heater")
    print("  - Switching penalty: -0.15 for turning heater on/off")
    print("  - Stability bonus: +0.1 for consecutive comfort steps")
    print()
    
    # Configuration
    SIMULATION_DURATION = 3600  # 1 hour
    INITIAL_BUDGET = 300.0
    INITIAL_TEMP = 19.0
    
    print(f"[Config] Duration: {SIMULATION_DURATION}s, Budget: €{INITIAL_BUDGET}")
    print()
    
    # Create environment
    env = SharedEnvironment(seed=RANDOM_SEED, simulation_duration=SIMULATION_DURATION)
    
    # Run Homeostatic
    print("[1/2] Running HOMEOSTATIC agent...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = SharedEnvironment(seed=RANDOM_SEED, simulation_duration=SIMULATION_DURATION)
    h_reward_calc = ComplexRewardCalculator()
    h_agent = HomeostaticAgentComplex(INITIAL_BUDGET, h_reward_calc)
    h_hist, h_sum = run_simulation(h_agent, env, INITIAL_TEMP, h_reward_calc)
    print(f"       Survived: {h_sum['survived_time']:.1f}s, Comfort: {h_sum['comfort_percentage']:.1f}%, "
          f"Cumulative Reward: {h_sum['cumulative_reward']:.1f}")
    
    # Run Extrinsic
    print("[2/2] Running EXTRINSIC (TD) agent...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = SharedEnvironment(seed=RANDOM_SEED, simulation_duration=SIMULATION_DURATION)
    e_reward_calc = ComplexRewardCalculator()
    e_agent = ExtrinsicAgentComplex(INITIAL_BUDGET, e_reward_calc)
    e_hist, e_sum = run_simulation(e_agent, env, INITIAL_TEMP, e_reward_calc)
    print(f"       Survived: {e_sum['survived_time']:.1f}s, Comfort: {e_sum['comfort_percentage']:.1f}%, "
          f"Cumulative Reward: {e_sum['cumulative_reward']:.1f}")
    
    # Generate plot
    print()
    print("[Plot] Generating comparison...")
    plot_file = generate_plot(h_hist, e_hist, h_sum, e_sum)
    
    # Print results
    print()
    print("=" * 70)
    print("  FINAL RESULTS - COMPLEX REWARD SCENARIO")
    print("=" * 70)
    
    print(f"\n  {'Metric':<30} {'Homeostatic':>15} {'Extrinsic (TD)':>15} {'Winner':>10}")
    print("  " + "-" * 70)
    
    metrics = [
        ("Survived Time (s)", h_sum['survived_time'], e_sum['survived_time'], "max"),
        ("Final Budget (€)", h_sum['final_budget'], e_sum['final_budget'], "max"),
        ("Comfort Zone (%)", h_sum['comfort_percentage'], e_sum['comfort_percentage'], "max"),
        ("CUMULATIVE REWARD", h_sum['cumulative_reward'], e_sum['cumulative_reward'], "max"),
        ("Base Rewards", h_sum['base_rewards'], e_sum['base_rewards'], "max"),
        ("Efficiency Bonus", h_sum['efficiency_bonus'], e_sum['efficiency_bonus'], "max"),
        ("Stability Bonus", h_sum['stability_bonus'], e_sum['stability_bonus'], "max"),
        ("Switching Penalty", h_sum['switching_penalties'], e_sum['switching_penalties'], "min"),
        ("Asymmetric Penalty", h_sum['asymmetric_penalties'], e_sum['asymmetric_penalties'], "min"),
    ]
    
    h_wins = 0
    e_wins = 0
    
    for name, h_val, e_val, mode in metrics:
        if mode == "max":
            winner = "Homeo" if h_val > e_val else ("Extrin" if e_val > h_val else "Tie")
            if h_val > e_val: h_wins += 1
            elif e_val > h_val: e_wins += 1
        else:
            winner = "Homeo" if h_val < e_val else ("Extrin" if e_val < h_val else "Tie")
            if h_val < e_val: h_wins += 1
            elif e_val < h_val: e_wins += 1
        
        print(f"  {name:<30} {h_val:>15.2f} {e_val:>15.2f} {winner:>10}")
    
    print("  " + "-" * 70)
    print(f"  {'SCORE':<30} {h_wins:>15} {e_wins:>15}")
    
    overall = "HOMEOSTATIC" if h_wins > e_wins else ("EXTRINSIC (TD)" if e_wins > h_wins else "TIE")
    print(f"\n  🏆 OVERALL WINNER: {overall}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
