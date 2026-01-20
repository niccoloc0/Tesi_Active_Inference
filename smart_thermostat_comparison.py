"""
Smart Thermostat Comparison: Homeostatic vs Extrinsic (TD Learning)

This script runs both agents in identical environmental conditions
and generates comparative visualizations.

Key comparison:
1. Homeostatic Agent: pragmatic_cost = (predicted_temp - target_temp)²
2. Extrinsic Agent: pragmatic_value = TD-learned V(s) + Expected Reward

Author: Active Inference Thermostat Project
Date: 2026-01-20
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Set consistent random seed for fair comparison
RANDOM_SEED = 42


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

class TDValueFunction:
    """Tabular Value Function with TD Learning for temperature states."""
    
    def __init__(self, temp_min=10.0, temp_max=35.0, num_bins=50, 
                 learning_rate=0.1, discount_factor=0.95):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.values = np.zeros(num_bins)
        self.eligibility = np.zeros(num_bins)
        self.lambda_trace = 0.8
        self.td_errors = []
        self.visit_counts = np.zeros(num_bins)
        
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
        self.visit_counts[current_bin] += 1
        return td_error
    
    def get_expected_reward(self, temp: float, target_temp: float, 
                           comfort_threshold: float, reward_amount: float, 
                           penalty_amount: float) -> float:
        dist = abs(temp - target_temp)
        scale = 0.5
        p_comfort = 1.0 / (1.0 + np.exp((dist - comfort_threshold) / scale))
        return p_comfort * reward_amount + (1 - p_comfort) * (-penalty_amount)


# ============================================================================
# HOMEOSTATIC AGENT
# ============================================================================

class HomeostaticAgent:
    """
    Active Inference agent with HOMEOSTATIC pragmatic value.
    pragmatic_cost = (predicted_temp - target_temp)²
    """
    
    def __init__(self, initial_budget, comfort_threshold=2.0, 
                 reward_amount=0.3, penalty_amount=0.5, heater_cost=0.2):
        self.name = "Homeostatic"
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        # Budget
        self.budget = initial_budget
        self.initial_budget = initial_budget
        self.cost_per_step = 0.1
        self.comfort_threshold = comfort_threshold
        self.reward_amount = reward_amount
        self.penalty_amount = penalty_amount
        self.heater_cost = heater_cost
        
        # Tracking
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        self.steps_in_comfort = 0
        self.steps_out_comfort = 0
        
        # Kalman Filter State
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics model
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.observation_precision = 16.0
        self.process_precision = 4.0
        
        # Learning rates
        self.dynamics_learning_rate = 0.01
        
        # History
        self.prediction_errors = []
        self.epistemic_values = []
        self.pragmatic_values = []
        
        # Epistemic action parameters
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
        """Homeostatic EFE: pragmatic_cost = (predicted_temp - target)²"""
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        predicted_temp = predicted_mean[0]
        pragmatic_cost = (predicted_temp - target_temp) ** 2
        pragmatic_value = -pragmatic_cost
        efe = -epistemic_value + pragmatic_cost
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        actions = [False, True]
        efes = []
        epistemic_vals = []
        pragmatic_vals = []
        
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, epist, prag = self.compute_efe_homeostatic(action, target_temp, pred_mean, pred_cov)
            efes.append(efe)
            epistemic_vals.append(epist)
            pragmatic_vals.append(prag)
        
        best_idx = np.argmin(efes)
        selected_action = actions[best_idx]
        self.epistemic_values.append(epistemic_vals[best_idx])
        self.pragmatic_values.append(pragmatic_vals[best_idx])
        
        return selected_action, epistemic_vals[best_idx], pragmatic_vals[best_idx]
    
    def select_sampling_rate(self):
        uncertainty = self.state_cov[0, 0]
        if uncertainty > self.uncertainty_threshold:
            dt = self.min_dt
        elif uncertainty > self.uncertainty_threshold / 2:
            dt = 1.0
        elif uncertainty > self.uncertainty_threshold / 4:
            dt = 2.0
        else:
            dt = self.max_dt
        return dt, uncertainty
    
    def learn_from_error(self, prediction_error, observation):
        squared_error = prediction_error ** 2
        estimated_variance = 0.9 / self.observation_precision + 0.1 * squared_error
        self.observation_precision = max(1.0, min(100.0, 1.0 / estimated_variance))
        
        if len(self.prediction_errors) > 5:
            recent_errors = self.prediction_errors[-5:]
            avg_error = sum(recent_errors) / len(recent_errors)
            self.B_heater *= (1.0 - self.dynamics_learning_rate * np.sign(avg_error))
            self.B_heater = max(0.5, min(3.0, self.B_heater))
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        self.learn_from_error(prediction_error, observation)
        
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        
        target_temp = self.SEASONS_TARGET[current_season]
        new_action, epistemic_val, pragmatic_val = self.select_action(
            observation, target_temp, current_action, outside_temp, dt
        )
        new_dt, uncertainty = self.select_sampling_rate()
        
        # Reward/Penalty
        temp_error = abs(observation - target_temp)
        in_comfort_zone = temp_error <= self.comfort_threshold
        
        step_cost = self.cost_per_step
        if current_action:
            step_cost += self.heater_cost
        
        if in_comfort_zone:
            step_reward = self.reward_amount
            self.total_rewards += step_reward
            self.steps_in_comfort += 1
        else:
            step_reward = -self.penalty_amount
            self.total_penalties += self.penalty_amount
            self.steps_out_comfort += 1
        
        net_change = step_reward - step_cost
        self.budget += net_change
        
        return {
            "action": new_action,
            "dt": new_dt,
            "free_energy": free_energy,
            "epistemic_value": epistemic_val,
            "pragmatic_value": pragmatic_val,
            "uncertainty": uncertainty,
            "belief_mean": self.state_mean[0],
            "in_comfort_zone": in_comfort_zone,
            "step_reward": step_reward,
            "budget_exhausted": self.budget <= 0,
        }


# ============================================================================
# EXTRINSIC AGENT (TD Learning)
# ============================================================================

class ExtrinsicAgent:
    """
    Active Inference agent with EXTRINSIC pragmatic value.
    pragmatic_value = TD-learned V(s) + Expected Reward
    """
    
    def __init__(self, initial_budget, comfort_threshold=2.0, 
                 reward_amount=0.3, penalty_amount=0.5, heater_cost=0.2,
                 td_learning_rate=0.1, td_discount=0.95):
        self.name = "Extrinsic (TD)"
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        # Budget
        self.budget = initial_budget
        self.initial_budget = initial_budget
        self.cost_per_step = 0.1
        self.comfort_threshold = comfort_threshold
        self.reward_amount = reward_amount
        self.penalty_amount = penalty_amount
        self.heater_cost = heater_cost
        
        # TD Value Function
        self.value_function = TDValueFunction(
            temp_min=10.0, temp_max=35.0, num_bins=50,
            learning_rate=td_learning_rate,
            discount_factor=td_discount
        )
        self.prev_temp = None
        self.prev_reward = None
        
        # Tracking
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        self.steps_in_comfort = 0
        self.steps_out_comfort = 0
        
        # Kalman Filter State
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics model
        self.B_heater = 1.5
        self.B_outside = 0.1
        self.observation_precision = 16.0
        self.process_precision = 4.0
        
        # Learning rates
        self.dynamics_learning_rate = 0.01
        
        # History
        self.prediction_errors = []
        self.epistemic_values = []
        self.pragmatic_values = []
        self.td_error_history = []
        
        # Epistemic action parameters
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
    
    def compute_efe_extrinsic(self, action, target_temp, predicted_mean, predicted_cov):
        """Extrinsic EFE: pragmatic_value = V(s) + Expected Reward"""
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        predicted_temp = predicted_mean[0]
        
        # TD-learned value
        learned_value = self.value_function.get_value_interpolated(predicted_temp)
        
        # Expected immediate reward
        expected_reward = self.value_function.get_expected_reward(
            predicted_temp, target_temp, 
            self.comfort_threshold, 
            self.reward_amount, 
            self.penalty_amount
        )
        
        # Combine: 50% learned + 50% expected
        pragmatic_value = 0.5 * learned_value + 0.5 * expected_reward
        
        # EFE (lower is better, so we minimize -pragmatic_value)
        efe = -epistemic_value - pragmatic_value
        
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        actions = [False, True]
        efes = []
        epistemic_vals = []
        pragmatic_vals = []
        
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, epist, prag = self.compute_efe_extrinsic(action, target_temp, pred_mean, pred_cov)
            efes.append(efe)
            epistemic_vals.append(epist)
            pragmatic_vals.append(prag)
        
        best_idx = np.argmin(efes)
        selected_action = actions[best_idx]
        self.epistemic_values.append(epistemic_vals[best_idx])
        self.pragmatic_values.append(pragmatic_vals[best_idx])
        
        return selected_action, epistemic_vals[best_idx], pragmatic_vals[best_idx]
    
    def select_sampling_rate(self):
        uncertainty = self.state_cov[0, 0]
        if uncertainty > self.uncertainty_threshold:
            dt = self.min_dt
        elif uncertainty > self.uncertainty_threshold / 2:
            dt = 1.0
        elif uncertainty > self.uncertainty_threshold / 4:
            dt = 2.0
        else:
            dt = self.max_dt
        return dt, uncertainty
    
    def learn_from_error(self, prediction_error, observation):
        squared_error = prediction_error ** 2
        estimated_variance = 0.9 / self.observation_precision + 0.1 * squared_error
        self.observation_precision = max(1.0, min(100.0, 1.0 / estimated_variance))
        
        if len(self.prediction_errors) > 5:
            recent_errors = self.prediction_errors[-5:]
            avg_error = sum(recent_errors) / len(recent_errors)
            self.B_heater *= (1.0 - self.dynamics_learning_rate * np.sign(avg_error))
            self.B_heater = max(0.5, min(3.0, self.B_heater))
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        self.learn_from_error(prediction_error, observation)
        
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        
        target_temp = self.SEASONS_TARGET[current_season]
        new_action, epistemic_val, pragmatic_val = self.select_action(
            observation, target_temp, current_action, outside_temp, dt
        )
        new_dt, uncertainty = self.select_sampling_rate()
        
        # Reward/Penalty
        temp_error = abs(observation - target_temp)
        in_comfort_zone = temp_error <= self.comfort_threshold
        
        step_cost = self.cost_per_step
        if current_action:
            step_cost += self.heater_cost
        
        if in_comfort_zone:
            step_reward = self.reward_amount
            self.total_rewards += step_reward
            self.steps_in_comfort += 1
        else:
            step_reward = -self.penalty_amount
            self.total_penalties += self.penalty_amount
            self.steps_out_comfort += 1
        
        # TD Update
        td_error = 0.0
        if self.prev_temp is not None:
            td_error = self.value_function.update_with_eligibility(
                self.prev_temp, self.prev_reward, observation
            )
            self.td_error_history.append(td_error)
        
        self.prev_temp = observation
        self.prev_reward = step_reward
        
        net_change = step_reward - step_cost
        self.budget += net_change
        
        return {
            "action": new_action,
            "dt": new_dt,
            "free_energy": free_energy,
            "epistemic_value": epistemic_val,
            "pragmatic_value": pragmatic_val,
            "uncertainty": uncertainty,
            "belief_mean": self.state_mean[0],
            "in_comfort_zone": in_comfort_zone,
            "step_reward": step_reward,
            "td_error": td_error,
            "value_at_state": self.value_function.get_value(observation),
            "budget_exhausted": self.budget <= 0,
        }


# ============================================================================
# ENVIRONMENT SIMULATION
# ============================================================================

class SharedEnvironment:
    """Shared environment that generates identical conditions for both agents."""
    
    def __init__(self, seed=42, simulation_duration=720, heating_power=1.5, weather_fluctuation=1.5):
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
        
        # Pre-generate weather fluctuations for consistency
        max_steps = 2000  # More than enough
        self.weather_noise = [random.uniform(-self.weather_fluctuation, self.weather_fluctuation) 
                              for _ in range(max_steps)]
    
    def get_season(self, sim_time):
        season_idx = int(sim_time // 60) % 4
        return self.season_order[season_idx]
    
    def get_outside_temp(self, sim_time, step):
        season = self.get_season(sim_time)
        base_temp = self.SEASONS[season]["outside_temp"]
        return base_temp + self.weather_noise[step % len(self.weather_noise)]
    
    def get_target_temp(self, sim_time):
        season = self.get_season(sim_time)
        return self.SEASONS_TARGET[season]
    
    def physics_update(self, room_temp, heater_on, outside_temp, dt):
        heat_from_heater = (self.heating_power * dt) if heater_on else 0
        heat_loss_or_gain = (outside_temp - room_temp) * 0.1 * dt
        return room_temp + heat_from_heater + heat_loss_or_gain


def run_simulation_for_agent(agent, env, initial_temp):
    """Run simulation for a single agent and collect history."""
    
    history = {
        "time": [], "room_temp": [], "target_temp": [], "outside_temp": [],
        "heater_on": [], "budget": [], "rewards": [], "penalties": [],
        "pragmatic_value": [], "epistemic_value": [], "in_comfort": []
    }
    
    room_temp = initial_temp
    heater_on = False
    current_dt = 1.0
    current_sim_time = 0
    step = 0
    
    while current_sim_time < env.simulation_duration:
        season = env.get_season(current_sim_time)
        outside_temp = env.get_outside_temp(current_sim_time, step)
        target_temp = env.get_target_temp(current_sim_time)
        
        # Agent step
        result = agent.step(room_temp, heater_on, season, outside_temp, current_dt)
        
        if result["budget_exhausted"]:
            break
        
        # Record history
        history["time"].append(current_sim_time)
        history["room_temp"].append(room_temp)
        history["target_temp"].append(target_temp)
        history["outside_temp"].append(outside_temp)
        history["heater_on"].append(1 if heater_on else 0)
        history["budget"].append(agent.budget)
        history["rewards"].append(agent.total_rewards)
        history["penalties"].append(agent.total_penalties)
        history["pragmatic_value"].append(result["pragmatic_value"])
        history["epistemic_value"].append(result["epistemic_value"])
        history["in_comfort"].append(1 if result["in_comfort_zone"] else 0)
        
        # Update for next step
        heater_on = result["action"]
        current_dt = result["dt"]
        
        # Physics update
        room_temp = env.physics_update(room_temp, heater_on, outside_temp, current_dt)
        
        current_sim_time += current_dt
        step += 1
    
    # Compute summary statistics
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
        "avg_pragmatic": np.mean(history["pragmatic_value"]) if history["pragmatic_value"] else 0,
    }
    
    return history, summary


def generate_comparison_plot(homeostatic_history, extrinsic_history, 
                              homeostatic_summary, extrinsic_summary):
    """Generate comprehensive comparison plot."""
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("Comparison: Homeostatic vs Extrinsic (TD Learning) EFE", fontsize=16, fontweight='bold')
    
    colors = {"homeostatic": "blue", "extrinsic": "orange"}
    
    # Row 1: Temperature Dynamics
    ax1 = axes[0, 0]
    ax1.plot(homeostatic_history["time"], homeostatic_history["room_temp"], 
             color=colors["homeostatic"], label="Room Temp", linewidth=1.5)
    ax1.plot(homeostatic_history["time"], homeostatic_history["target_temp"], 
             'g--', label="Target", alpha=0.7)
    ax1.plot(homeostatic_history["time"], homeostatic_history["outside_temp"], 
             'gray', label="Outside", alpha=0.4)
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(f"HOMEOSTATIC Agent - Temperature")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(extrinsic_history["time"], extrinsic_history["room_temp"], 
             color=colors["extrinsic"], label="Room Temp", linewidth=1.5)
    ax2.plot(extrinsic_history["time"], extrinsic_history["target_temp"], 
             'g--', label="Target", alpha=0.7)
    ax2.plot(extrinsic_history["time"], extrinsic_history["outside_temp"], 
             'gray', label="Outside", alpha=0.4)
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title(f"EXTRINSIC Agent (TD) - Temperature")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Budget Evolution
    ax3 = axes[1, 0]
    ax3.plot(homeostatic_history["time"], homeostatic_history["budget"], 
             color=colors["homeostatic"], linewidth=2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel("Budget (€)")
    ax3.set_title(f"HOMEOSTATIC - Budget (Final: {homeostatic_summary['final_budget']:.1f})")
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(extrinsic_history["time"], extrinsic_history["budget"], 
             color=colors["extrinsic"], linewidth=2)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_ylabel("Budget (€)")
    ax4.set_title(f"EXTRINSIC - Budget (Final: {extrinsic_summary['final_budget']:.1f})")
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Pragmatic Values Comparison
    ax5 = axes[2, 0]
    ax5.plot(homeostatic_history["time"], homeostatic_history["pragmatic_value"], 
             color=colors["homeostatic"], alpha=0.7, label="Homeostatic")
    ax5.plot(extrinsic_history["time"], extrinsic_history["pragmatic_value"][:len(homeostatic_history["time"])], 
             color=colors["extrinsic"], alpha=0.7, label="Extrinsic")
    ax5.set_ylabel("Pragmatic Value")
    ax5.set_title("Pragmatic Value Comparison")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[2, 1]
    ax6.plot(homeostatic_history["time"], homeostatic_history["epistemic_value"], 
             color=colors["homeostatic"], alpha=0.7, label="Homeostatic")
    ax6.plot(extrinsic_history["time"], extrinsic_history["epistemic_value"][:len(homeostatic_history["time"])], 
             color=colors["extrinsic"], alpha=0.7, label="Extrinsic")
    ax6.set_ylabel("Epistemic Value")
    ax6.set_title("Epistemic Value Comparison")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 4: Comfort Zone & Summary
    ax7 = axes[3, 0]
    
    # Rolling comfort percentage
    window = 20
    h_comfort_rolling = np.convolve(homeostatic_history["in_comfort"], np.ones(window)/window, mode='valid')
    e_comfort_rolling = np.convolve(extrinsic_history["in_comfort"], np.ones(window)/window, mode='valid')
    
    ax7.plot(homeostatic_history["time"][:len(h_comfort_rolling)], h_comfort_rolling * 100, 
             color=colors["homeostatic"], label=f"Homeostatic ({homeostatic_summary['comfort_percentage']:.1f}%)")
    ax7.plot(extrinsic_history["time"][:len(e_comfort_rolling)], e_comfort_rolling * 100, 
             color=colors["extrinsic"], label=f"Extrinsic ({extrinsic_summary['comfort_percentage']:.1f}%)")
    ax7.set_ylabel("Comfort %")
    ax7.set_xlabel("Time (s)")
    ax7.set_title("Rolling Comfort Zone Percentage")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 100])
    
    # Summary Table
    ax8 = axes[3, 1]
    ax8.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║                 COMPARISON SUMMARY                   ║
    ╠══════════════════════════════════════════════════════╣
    ║                  HOMEOSTATIC │ EXTRINSIC (TD)        ║
    ╠══════════════════════════════════════════════════════╣
    ║  Survived Time     {homeostatic_summary['survived_time']:>7.1f}s │ {extrinsic_summary['survived_time']:>7.1f}s             ║
    ║  Total Steps       {homeostatic_summary['total_steps']:>7d}  │ {extrinsic_summary['total_steps']:>7d}              ║
    ║  Final Budget     €{homeostatic_summary['final_budget']:>7.2f} │ €{extrinsic_summary['final_budget']:>7.2f}            ║
    ║  Total Rewards    +{homeostatic_summary['total_rewards']:>7.2f} │ +{extrinsic_summary['total_rewards']:>7.2f}            ║
    ║  Total Penalties  -{homeostatic_summary['total_penalties']:>7.2f} │ -{extrinsic_summary['total_penalties']:>7.2f}            ║
    ║  Comfort Zone     {homeostatic_summary['comfort_percentage']:>7.1f}% │ {extrinsic_summary['comfort_percentage']:>7.1f}%             ║
    ║  Avg Pragmatic    {homeostatic_summary['avg_pragmatic']:>8.3f} │ {extrinsic_summary['avg_pragmatic']:>8.3f}            ║
    ╚══════════════════════════════════════════════════════╝
    
    Winner: {"HOMEOSTATIC" if homeostatic_summary['survived_time'] > extrinsic_summary['survived_time'] else "EXTRINSIC (TD)" if extrinsic_summary['survived_time'] > homeostatic_summary['survived_time'] else "TIE"}
    (Based on survival time)
    """
    
    ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_homeostatic_vs_extrinsic_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    print(f"[Comparison] Plot saved to {filename}")
    
    return filename


def main():
    print("=" * 70)
    print("  COMPARISON: HOMEOSTATIC vs EXTRINSIC (TD Learning) EFE")
    print("=" * 70)
    
    # Configuration
    SIMULATION_DURATION = 3600  # 60 minutes (1 hour) - longer for TD convergence
    INITIAL_BUDGET = 500.0  # Increased for longer simulation
    INITIAL_TEMP = 18.5
    
    # Create shared environment with fixed seed
    print("\n[1/4] Creating shared environment with fixed random seed...")
    env = SharedEnvironment(seed=RANDOM_SEED, simulation_duration=SIMULATION_DURATION)
    
    # Create agents with identical initial conditions
    print("[2/4] Initializing agents...")
    homeostatic_agent = HomeostaticAgent(initial_budget=INITIAL_BUDGET)
    extrinsic_agent = ExtrinsicAgent(initial_budget=INITIAL_BUDGET)
    
    print(f"      - Homeostatic Agent: budget={INITIAL_BUDGET}")
    print(f"      - Extrinsic (TD) Agent: budget={INITIAL_BUDGET}")
    
    # Run simulations
    print("\n[3/4] Running simulations...")
    
    # Reset random for fair comparison (same noise sequence)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print("      Running Homeostatic Agent...")
    homeostatic_history, homeostatic_summary = run_simulation_for_agent(
        homeostatic_agent, env, INITIAL_TEMP
    )
    print(f"      - Completed: {homeostatic_summary['survived_time']:.1f}s, "
          f"Comfort: {homeostatic_summary['comfort_percentage']:.1f}%")
    
    # Reset random for fair comparison (same noise sequence)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env = SharedEnvironment(seed=RANDOM_SEED, simulation_duration=SIMULATION_DURATION)
    print("      Running Extrinsic (TD) Agent...")
    extrinsic_history, extrinsic_summary = run_simulation_for_agent(
        extrinsic_agent, env, INITIAL_TEMP
    )
    print(f"      - Completed: {extrinsic_summary['survived_time']:.1f}s, "
          f"Comfort: {extrinsic_summary['comfort_percentage']:.1f}%")
    
    # Generate comparison plot
    print("\n[4/4] Generating comparison plot...")
    plot_file = generate_comparison_plot(
        homeostatic_history, extrinsic_history,
        homeostatic_summary, extrinsic_summary
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n  {'Metric':<25} {'Homeostatic':>15} {'Extrinsic (TD)':>15} {'Winner':>12}")
    print("  " + "-" * 67)
    
    metrics = [
        ("Survived Time (s)", homeostatic_summary['survived_time'], extrinsic_summary['survived_time'], "max"),
        ("Total Steps", homeostatic_summary['total_steps'], extrinsic_summary['total_steps'], "max"),
        ("Final Budget (€)", homeostatic_summary['final_budget'], extrinsic_summary['final_budget'], "max"),
        ("Total Rewards", homeostatic_summary['total_rewards'], extrinsic_summary['total_rewards'], "max"),
        ("Total Penalties", homeostatic_summary['total_penalties'], extrinsic_summary['total_penalties'], "min"),
        ("Comfort Zone (%)", homeostatic_summary['comfort_percentage'], extrinsic_summary['comfort_percentage'], "max"),
    ]
    
    homeostatic_wins = 0
    extrinsic_wins = 0
    
    for name, h_val, e_val, mode in metrics:
        if mode == "max":
            winner = "Homeo" if h_val > e_val else ("Extrin" if e_val > h_val else "Tie")
            if h_val > e_val:
                homeostatic_wins += 1
            elif e_val > h_val:
                extrinsic_wins += 1
        else:  # min
            winner = "Homeo" if h_val < e_val else ("Extrin" if e_val < h_val else "Tie")
            if h_val < e_val:
                homeostatic_wins += 1
            elif e_val < h_val:
                extrinsic_wins += 1
        
        print(f"  {name:<25} {h_val:>15.2f} {e_val:>15.2f} {winner:>12}")
    
    print("  " + "-" * 67)
    print(f"  {'SCORE':<25} {homeostatic_wins:>15} {extrinsic_wins:>15}")
    
    overall_winner = "HOMEOSTATIC" if homeostatic_wins > extrinsic_wins else \
                     ("EXTRINSIC (TD)" if extrinsic_wins > homeostatic_wins else "TIE")
    print(f"\n  🏆 OVERALL WINNER: {overall_winner}")
    
    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)
    
    return plot_file


if __name__ == "__main__":
    main()
