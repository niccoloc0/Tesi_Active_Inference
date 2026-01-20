"""
Smart Thermostat with Extrinsic Reward (Temporal Difference Learning)

This version replaces the homeostatic pragmatic cost with an extrinsic reward-based
approach using Temporal Difference (TD) Learning.

Key differences from the homeostatic version:
1. Pragmatic Value: Uses learned value function V(s) instead of (temp - target)²
2. TD Learning: Updates value function based on actual rewards received
3. Reward Signal: +reward if in comfort zone, -penalty if outside

Formula:
- TD Error: δ = r + γ * V(s') - V(s)
- Value Update: V(s) ← V(s) + α * δ
- Pragmatic Value for EFE: Expected reward E[r(s', a)]

Author: Active Inference Thermostat Project
Date: 2026-01-20
"""

import threading
import time
import random
import wandb
from queue import Queue
import numpy as np
import hmac
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- HMAC Security for Sensor-Agent Communication ---
class SecureMessage:
    """HMAC-based message authentication for secure sensor-agent communication."""
    
    SECRET_KEY = b"!T\\q!Un'8AL4bpHH"
    
    @staticmethod
    def create_signature(data: dict) -> str:
        message = json.dumps(data, sort_keys=True).encode('utf-8')
        signature = hmac.new(SecureMessage.SECRET_KEY, message, hashlib.sha256).hexdigest()
        return signature
    
    @staticmethod
    def sign_message(data: dict) -> dict:
        signature = SecureMessage.create_signature(data)
        return {"payload": data, "signature": signature}
    
    @staticmethod
    def verify_message(signed_message: dict) -> tuple[bool, dict]:
        try:
            payload = signed_message.get("payload", {})
            received_signature = signed_message.get("signature", "")
            expected_signature = SecureMessage.create_signature(payload)
            is_valid = hmac.compare_digest(received_signature, expected_signature)
            if is_valid:
                return True, payload
            else:
                print("[Security] ⚠️ HMAC verification failed!")
                return False, {}
        except Exception as e:
            print(f"[Security] ⚠️ Error during verification: {e}")
            return False, {}


# --- Temporal Difference Value Function ---
class TDValueFunction:
    """
    Tabular Value Function with TD Learning for temperature states.
    
    Discretizes temperature into bins and learns value for each bin
    based on actual rewards received.
    """
    
    def __init__(self, temp_min=10.0, temp_max=35.0, num_bins=50, 
                 learning_rate=0.1, discount_factor=0.95):
        """
        Initialize TD Value Function.
        
        Args:
            temp_min: Minimum temperature in the state space
            temp_max: Maximum temperature in the state space
            num_bins: Number of discrete temperature bins
            learning_rate: TD learning rate (α)
            discount_factor: Discount factor for future rewards (γ)
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize value function (slightly optimistic initialization)
        self.values = np.zeros(num_bins)
        
        # Eligibility traces for TD(λ) (optional, set λ=0 for TD(0))
        self.eligibility = np.zeros(num_bins)
        self.lambda_trace = 0.8  # Eligibility trace decay
        
        # Statistics
        self.td_errors = []
        self.visit_counts = np.zeros(num_bins)
        
    def temp_to_bin(self, temp: float) -> int:
        """Convert continuous temperature to discrete bin index."""
        # Clip to valid range
        temp = np.clip(temp, self.temp_min, self.temp_max)
        # Linear mapping to bin
        bin_idx = int((temp - self.temp_min) / (self.temp_max - self.temp_min) * (self.num_bins - 1))
        return np.clip(bin_idx, 0, self.num_bins - 1)
    
    def bin_to_temp(self, bin_idx: int) -> float:
        """Convert bin index back to temperature (center of bin)."""
        return self.temp_min + (bin_idx + 0.5) / self.num_bins * (self.temp_max - self.temp_min)
    
    def get_value(self, temp: float) -> float:
        """Get the value V(s) for a given temperature state."""
        bin_idx = self.temp_to_bin(temp)
        return self.values[bin_idx]
    
    def get_value_interpolated(self, temp: float) -> float:
        """Get interpolated value between bins for smoother estimation."""
        temp = np.clip(temp, self.temp_min, self.temp_max)
        # Calculate fractional bin position
        frac_pos = (temp - self.temp_min) / (self.temp_max - self.temp_min) * (self.num_bins - 1)
        lower_bin = int(frac_pos)
        upper_bin = min(lower_bin + 1, self.num_bins - 1)
        frac = frac_pos - lower_bin
        
        # Linear interpolation
        return (1 - frac) * self.values[lower_bin] + frac * self.values[upper_bin]
    
    def update(self, current_temp: float, reward: float, next_temp: float) -> float:
        """
        TD(0) Update: V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
        
        Args:
            current_temp: Current state temperature
            reward: Reward received (positive for comfort, negative for penalty)
            next_temp: Next state temperature
            
        Returns:
            TD error (δ)
        """
        current_bin = self.temp_to_bin(current_temp)
        next_bin = self.temp_to_bin(next_temp)
        
        # TD Error: δ = r + γ * V(s') - V(s)
        td_error = reward + self.discount_factor * self.values[next_bin] - self.values[current_bin]
        
        # Update Value: V(s) ← V(s) + α * δ
        self.values[current_bin] += self.learning_rate * td_error
        
        # Track statistics
        self.td_errors.append(td_error)
        self.visit_counts[current_bin] += 1
        
        return td_error
    
    def update_with_eligibility(self, current_temp: float, reward: float, next_temp: float) -> float:
        """
        TD(λ) Update with eligibility traces for faster credit assignment.
        
        Args:
            current_temp: Current state temperature
            reward: Reward received
            next_temp: Next state temperature
            
        Returns:
            TD error (δ)
        """
        current_bin = self.temp_to_bin(current_temp)
        next_bin = self.temp_to_bin(next_temp)
        
        # TD Error
        td_error = reward + self.discount_factor * self.values[next_bin] - self.values[current_bin]
        
        # Update eligibility trace: e(s) ← γλe(s) + 1 for current state
        self.eligibility *= self.discount_factor * self.lambda_trace
        self.eligibility[current_bin] += 1.0
        
        # Update all values proportional to eligibility
        self.values += self.learning_rate * td_error * self.eligibility
        
        # Track statistics
        self.td_errors.append(td_error)
        self.visit_counts[current_bin] += 1
        
        return td_error
    
    def reset_eligibility(self):
        """Reset eligibility traces (call at episode start)."""
        self.eligibility = np.zeros(self.num_bins)
    
    def get_expected_reward(self, temp: float, target_temp: float, 
                           comfort_threshold: float, reward_amount: float, 
                           penalty_amount: float) -> float:
        """
        Estimate expected reward based on predicted temperature.
        Uses soft probability of being in comfort zone.
        
        Args:
            temp: Predicted temperature
            target_temp: Target temperature
            comfort_threshold: Threshold for comfort zone
            reward_amount: Reward for being in comfort
            penalty_amount: Penalty for being outside comfort
            
        Returns:
            Expected reward value
        """
        # Distance from target
        dist = abs(temp - target_temp)
        
        # Soft probability of being in comfort zone
        # Using sigmoid-like function for smooth transition
        # P(comfort) = 1 / (1 + exp((dist - threshold) / scale))
        scale = 0.5  # Sharpness of transition
        p_comfort = 1.0 / (1.0 + np.exp((dist - comfort_threshold) / scale))
        
        # Expected reward = P(comfort) * reward + P(not_comfort) * (-penalty)
        expected_reward = p_comfort * reward_amount + (1 - p_comfort) * (-penalty_amount)
        
        return expected_reward


# --- Active Inference Agent with Extrinsic Reward ---
class ActiveInferenceAgentExtrinsic:
    """
    Active Inference agent with EXTRINSIC reward-based pragmatic value.
    
    Key difference from homeostatic version:
    - Pragmatic Value uses TD-learned value function V(s) instead of -(temp - target)²
    - The agent learns from actual reward signals, not just distance to target
    
    This allows the agent to:
    1. Learn non-linear preferences (comfort zones with sharp boundaries)
    2. Adapt to changing reward structures
    3. Balance exploration (epistemic) vs exploitation (pragmatic) based on learned values
    """
    
    def __init__(self, initial_budget=None, comfort_threshold=2.0, 
                 reward_amount=0.3, penalty_amount=0.5, heater_cost=0.2,
                 td_learning_rate=0.1, td_discount=0.95,
                 use_td_lambda=True):
        """
        Initialize Active Inference Agent with Extrinsic Reward.
        
        Args:
            initial_budget: Starting budget (random if None)
            comfort_threshold: Temperature threshold for comfort zone (±°C)
            reward_amount: Reward for being in comfort zone
            penalty_amount: Penalty for being outside comfort zone
            heater_cost: Cost per step when heater is on
            td_learning_rate: Learning rate for TD updates (α)
            td_discount: Discount factor for future rewards (γ)
            use_td_lambda: Use TD(λ) with eligibility traces
        """
        # Configuration
        self.comfort_margin = 1.0
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        # Budget and Reward System
        self.budget = initial_budget if initial_budget else random.uniform(50, 100)
        self.initial_budget = self.budget
        self.cost_per_step = 0.1
        
        # Reward/Penalty parameters
        self.comfort_threshold = comfort_threshold
        self.reward_amount = reward_amount
        self.penalty_amount = penalty_amount
        self.heater_cost = heater_cost
        
        # Tracking rewards/penalties
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        self.steps_in_comfort = 0
        self.steps_out_comfort = 0
        
        # === TD Value Function for Extrinsic Reward ===
        self.value_function = TDValueFunction(
            temp_min=10.0, temp_max=35.0, num_bins=50,
            learning_rate=td_learning_rate,
            discount_factor=td_discount
        )
        self.use_td_lambda = use_td_lambda
        
        # Store previous state for TD update
        self.prev_temp = None
        self.prev_reward = None
        
        # Kalman Filter State
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        
        # Observation model
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics model
        self.A = np.array([[1.0, 1.0], [0.0, 0.95]])
        self.B_heater = 1.5
        self.B_outside = 0.1
        
        # Precision parameters
        self.observation_precision = 16.0
        self.process_precision = 4.0
        
        # Learning rates
        self.dynamics_learning_rate = 0.01
        self.precision_learning_rate = 0.05
        
        # History
        self.prediction_errors = []
        self.free_energy_history = []
        self.epistemic_values = []
        self.pragmatic_values = []
        self.td_error_history = []
        
        # Epistemic action parameters
        self.min_dt = 0.5
        self.max_dt = 3.0
        self.uncertainty_threshold = 2.0
        
        # === Weighting for EFE components ===
        self.epistemic_weight = 1.0
        self.pragmatic_weight = 1.0  # Can be tuned
        
        print(f"[Agent-Extrinsic] Active Inference with TD Learning initialized")
        print(f"[Agent-Extrinsic] Budget: {self.budget:.2f}")
        print(f"[Agent-Extrinsic] Comfort threshold: ±{self.comfort_threshold}°C")
        print(f"[Agent-Extrinsic] TD Learning: α={td_learning_rate}, γ={td_discount}, λ={self.use_td_lambda}")
        print(f"[Agent-Extrinsic] Reward: +{self.reward_amount} | Penalty: -{self.penalty_amount}")
    
    def predict(self, action, dt, outside_temp):
        """Prediction step: Use generative model to predict next state."""
        A_dt = np.array([[1.0, dt], [0.0, 0.95]])
        control_effect = np.array([self.B_heater * dt if action else 0.0, 0.0])
        outside_effect = np.array([self.B_outside * dt, 0.0])
        
        predicted_mean = A_dt @ self.state_mean + control_effect
        predicted_mean[0] += outside_effect[0] * (outside_temp - self.state_mean[0])
        
        process_noise = np.eye(2) / self.process_precision
        predicted_cov = A_dt @ self.state_cov @ A_dt.T + process_noise
        
        return predicted_mean, predicted_cov
    
    def update(self, observation, predicted_mean, predicted_cov):
        """Update step: Incorporate observation using Bayesian inference."""
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
    
    def compute_expected_free_energy_extrinsic(self, action, target_temp, 
                                                predicted_mean, predicted_cov):
        """
        Compute Expected Free Energy with EXTRINSIC pragmatic value.
        
        EFE = Epistemic Value + Pragmatic Value
        
        Where:
        - Epistemic: Information gain (same as homeostatic version)
        - Pragmatic: Expected reward based on TD-learned value function
        
        The key difference: Instead of -(predicted_temp - target)², we use:
        - V(predicted_temp): Learned value of predicted state
        - Expected Reward: Soft probability of comfort * reward structure
        """
        # === EPISTEMIC VALUE (unchanged) ===
        # Information gain from reducing uncertainty
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        
        # === PRAGMATIC VALUE (EXTRINSIC - TD-based) ===
        predicted_temp = predicted_mean[0]
        
        # Method 1: Use learned value function V(s)
        # This captures the long-term expected reward from being in state s
        learned_value = self.value_function.get_value_interpolated(predicted_temp)
        
        # Method 2: Expected immediate reward (soft comfort probability)
        expected_reward = self.value_function.get_expected_reward(
            predicted_temp, target_temp, 
            self.comfort_threshold, 
            self.reward_amount, 
            self.penalty_amount
        )
        
        # Combine: Learned + Expected immediate reward
        # The learned value captures long-term structure,
        # expected reward provides immediate guidance
        pragmatic_value = 0.5 * learned_value + 0.5 * expected_reward
        
        # For comparison: Calculate homeostatic cost as well
        homeostatic_cost = (predicted_temp - target_temp) ** 2
        
        # Expected Free Energy (lower is better)
        # Minimize: -epistemic_value - pragmatic_value
        # = Minimize uncertainty growth + Maximize expected reward
        efe = -self.epistemic_weight * epistemic_value - self.pragmatic_weight * pragmatic_value
        
        return efe, epistemic_value, pragmatic_value, homeostatic_cost
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        """Select action by minimizing expected free energy (extrinsic version)."""
        actions = [False, True]
        efes = []
        epistemic_vals = []
        pragmatic_vals = []
        homeostatic_costs = []
        
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, epist, prag, home_cost = self.compute_expected_free_energy_extrinsic(
                action, target_temp, pred_mean, pred_cov
            )
            efes.append(efe)
            epistemic_vals.append(epist)
            pragmatic_vals.append(prag)
            homeostatic_costs.append(home_cost)
        
        # Select action with minimum EFE
        best_idx = np.argmin(efes)
        selected_action = actions[best_idx]
        
        # Store values for logging
        self.epistemic_values.append(epistemic_vals[best_idx])
        self.pragmatic_values.append(pragmatic_vals[best_idx])
        
        return selected_action, epistemic_vals[best_idx], pragmatic_vals[best_idx], homeostatic_costs[best_idx]
    
    def select_sampling_rate(self):
        """Epistemic action: Adjust sampling rate based on uncertainty."""
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
        """Online learning: Update model parameters based on prediction errors."""
        squared_error = prediction_error ** 2
        estimated_variance = 0.9 / self.observation_precision + 0.1 * squared_error
        self.observation_precision = max(1.0, min(100.0, 1.0 / estimated_variance))
        
        if len(self.prediction_errors) > 5:
            recent_errors = self.prediction_errors[-5:]
            avg_error = sum(recent_errors) / len(recent_errors)
            self.B_heater *= (1.0 - self.dynamics_learning_rate * np.sign(avg_error))
            self.B_heater = max(0.5, min(3.0, self.B_heater))
    
    def update_td_value(self, current_temp: float, reward: float, next_temp: float):
        """
        Update TD value function with new experience.
        
        This is called after receiving the actual reward to update V(s).
        """
        if self.use_td_lambda:
            td_error = self.value_function.update_with_eligibility(
                current_temp, reward, next_temp
            )
        else:
            td_error = self.value_function.update(
                current_temp, reward, next_temp
            )
        
        self.td_error_history.append(td_error)
        return td_error
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        """
        Single step of active inference with TD learning.
        
        Key addition: TD update to learn value function from actual rewards.
        """
        # 1. Predict next state
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        
        # 2. Update beliefs with observation
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        
        # 3. Learn from prediction error (model learning)
        self.learn_from_error(prediction_error, observation)
        
        # 4. Store metrics
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        self.free_energy_history.append(free_energy)
        
        # 5. Select action (minimize expected free energy)
        target_temp = self.SEASONS_TARGET[current_season]
        new_action, epistemic_val, pragmatic_val, homeostatic_cost = self.select_action(
            observation, target_temp, current_action, outside_temp, dt
        )
        
        # 6. Select sampling rate
        new_dt, uncertainty = self.select_sampling_rate()
        
        # === 7. REWARD/PENALTY SYSTEM ===
        temp_error = abs(observation - target_temp)
        in_comfort_zone = temp_error <= self.comfort_threshold
        
        step_cost = self.cost_per_step
        if current_action:
            step_cost += self.heater_cost
        
        step_reward = 0.0
        if in_comfort_zone:
            step_reward = self.reward_amount
            self.total_rewards += step_reward
            self.steps_in_comfort += 1
        else:
            step_reward = -self.penalty_amount
            self.total_penalties += self.penalty_amount
            self.steps_out_comfort += 1
        
        # === 8. TD VALUE FUNCTION UPDATE ===
        # Update V(s) with actual reward received
        td_error = 0.0
        if self.prev_temp is not None:
            td_error = self.update_td_value(
                self.prev_temp,
                self.prev_reward,
                observation
            )
        
        # Store current state for next TD update
        self.prev_temp = observation
        self.prev_reward = step_reward  # Use actual reward signal
        
        # Update budget
        net_change = step_reward - step_cost
        self.budget += net_change
        
        # Check budget exhaustion
        budget_exhausted = self.budget <= 0
        
        return {
            "action": new_action,
            "dt": new_dt,
            "free_energy": free_energy,
            "prediction_error": prediction_error,
            "epistemic_value": epistemic_val,
            "pragmatic_value": pragmatic_val,
            "homeostatic_cost": homeostatic_cost,  # For comparison
            "uncertainty": uncertainty,
            "belief_mean": self.state_mean[0],
            "observation_precision": self.observation_precision,
            "budget_exhausted": budget_exhausted,
            # Reward metrics
            "in_comfort_zone": in_comfort_zone,
            "step_reward": step_reward,
            "step_cost": step_cost,
            "net_change": net_change,
            "temp_error": temp_error,
            "total_rewards": self.total_rewards,
            "total_penalties": self.total_penalties,
            # TD Learning metrics
            "td_error": td_error,
            "value_at_state": self.value_function.get_value(observation),
        }


def run_agent(sensor_to_agent_queue, agent_to_sensor_queue):
    """Agent using Active Inference with Extrinsic Reward (TD Learning)."""
    
    agent = ActiveInferenceAgentExtrinsic(
        td_learning_rate=0.1,
        td_discount=0.95,
        use_td_lambda=True
    )
    
    previous_action = False
    current_dt = 1.0
    step_count = 0
    
    try:
        while True:
            signed_sensor_data = sensor_to_agent_queue.get()
            is_valid, sensor_data = SecureMessage.verify_message(signed_sensor_data)
            if not is_valid:
                print("[Agent-Extrinsic] ⚠️ Received tampered message! Skipping...")
                continue
            
            if sensor_data.get("command") == "simulation_end":
                print("\n[Agent-Extrinsic] Shutdown signal received.")
                print(f"[Agent-Extrinsic] Final belief: temp={agent.state_mean[0]:.2f}°C")
                print(f"[Agent-Extrinsic] TD Value Function learned, visits: {agent.value_function.visit_counts.sum():.0f}")
                break
            
            observation = sensor_data["room_temperature"]
            current_season = sensor_data["current_season"]
            outside_temp = sensor_data.get("outside_temp", 15.0)
            sim_time = sensor_data.get("sim_time", 0)
            
            result = agent.step(observation, previous_action, current_season, outside_temp, current_dt)
            
            if result.get("budget_exhausted", False):
                comfort_pct = (agent.steps_in_comfort / step_count * 100) if step_count > 0 else 0
                print(f"\n{'='*60}")
                print(f"[Agent-Extrinsic] BUDGET EXHAUSTED at step {step_count}!")
                print(f"[Agent-Extrinsic] Survived: {sim_time:.1f}s ({step_count} steps)")
                print(f"[Agent-Extrinsic] Final budget: {agent.budget:.2f}")
                print(f"[Agent-Extrinsic] Rewards: +{agent.total_rewards:.2f} | Penalties: -{agent.total_penalties:.2f}")
                print(f"[Agent-Extrinsic] Comfort Zone: {comfort_pct:.1f}%")
                print(f"[Agent-Extrinsic] Avg TD Error: {np.mean(agent.td_error_history[-50:]):.4f}")
                print(f"{'='*60}\n")
                
                shutdown_command = {"command": "budget_exhausted", "step": step_count, "sim_time": sim_time}
                signed_shutdown = SecureMessage.sign_message(shutdown_command)
                agent_to_sensor_queue.put(signed_shutdown)
                break
            
            previous_action = result["action"]
            current_dt = result["dt"]
            
            if agent.budget <= 0:
                action_taken = "BUDGET_EXHAUSTED"
                previous_action = False
            elif result["action"] and not sensor_data["heater_on"]:
                action_taken = "TURN_ON"
            elif not result["action"] and sensor_data["heater_on"]:
                action_taken = "TURN_OFF"
            else:
                action_taken = "IDLE"
            
            if step_count % 10 == 0 or action_taken in ["TURN_ON", "TURN_OFF"]:
                comfort_icon = "[OK]" if result.get('in_comfort_zone', False) else "[!!]"
                print(f"\n[Agent-Extrinsic] Step {step_count} | Season: {current_season} | Budget: {agent.budget:.1f} {comfort_icon}")
                print(f"  Obs: {observation:.2f}°C | Belief: {result['belief_mean']:.2f}°C | V(s): {result['value_at_state']:.3f}")
                print(f"  TD Error: {result['td_error']:.4f} | Pragmatic: {result['pragmatic_value']:.3f} (vs Homeo: {result['homeostatic_cost']:.3f})")
                print(f"  Reward: {result.get('step_reward', 0):+.2f} | Action: {action_taken}")
            
            command = {
                "heater_on": previous_action if agent.budget > 0 else False,
                "action_taken": action_taken,
                "dt": current_dt,
                "budget": agent.budget,
                "free_energy": result["free_energy"],
                "prediction_error": result["prediction_error"],
                "epistemic_value": result["epistemic_value"],
                "pragmatic_value": result["pragmatic_value"],
                "homeostatic_cost": result["homeostatic_cost"],
                "uncertainty": result["uncertainty"],
                "belief_mean": result["belief_mean"],
                "observation_precision": result["observation_precision"],
                "in_comfort_zone": result.get("in_comfort_zone", False),
                "step_reward": result.get("step_reward", 0),
                "total_rewards": result.get("total_rewards", 0),
                "total_penalties": result.get("total_penalties", 0),
                "comfort_percentage": (agent.steps_in_comfort / step_count * 100) if step_count > 0 else 0,
                "td_error": result["td_error"],
                "value_at_state": result["value_at_state"],
            }
            signed_command = SecureMessage.sign_message(command)
            agent_to_sensor_queue.put(signed_command)
            
            step_count += 1

    except Exception as e:
        print(f"[Agent-Extrinsic] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Agent-Extrinsic] Terminated.")


def run_sensor(sensor_to_agent_queue, agent_to_sensor_queue):
    """Sensor simulates the environment and communicates with agent."""
    
    try:
        run = wandb.init(
            project="Smart Thermostat Extrinsic",
            config={
                "simulation_duration": 720,
                "steps_per_season": 60, 
                "heating_power": 1.5, 
                "weather_fluctuation": 1.5,
                "agent_type": "extrinsic_td",
            },
            mode="online" 
        )
    except Exception as e:
        print(f"[Sensor] WandB init failed: {e}. Continuing without logging.")
        class MockWandb:
            config = {
                "simulation_duration": 720, "steps_per_season": 60, 
                "heating_power": 1.5, "weather_fluctuation": 1.5
            }
            def log(self, data): pass
            def finish(self): pass
        run = MockWandb()
        wandb.config = run.config

    history = {
        "time": [], "room_temp": [], "target_temp": [], "outside_temp": [],
        "heater_on": [], "budget": [], "rewards": [], "penalties": [],
        "pragmatic_value": [], "homeostatic_cost": [], "td_error": [], "value_at_state": []
    }

    SEASONS = {
        "Inverno": {"outside_temp": 0}, "Primavera": {"outside_temp": 15},
        "Estate": {"outside_temp": 35}, "Autunno": {"outside_temp": 12}
    }
    SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
    
    season_order = ["Inverno", "Primavera", "Estate", "Autunno"]
    room_temperature = random.uniform(17.5, 19.5)
    heater_on = False
    
    current_sim_time = 0
    current_dt = 1.0
    step_count = 0
    agent_budget = 0

    print(f"[Sensor] Starting EXTRINSIC simulation. Initial temp: {room_temperature:.1f}°C.")

    try:
        while current_sim_time < wandb.config["simulation_duration"]:
            season_idx = int(current_sim_time // 60) % 4
            current_season_name = season_order[season_idx]
            
            season_config = SEASONS[current_season_name]
            outside_temp = season_config["outside_temp"] + random.uniform(
                -wandb.config["weather_fluctuation"], wandb.config["weather_fluctuation"]
            )
            
            sensor_data = {
                "room_temperature": room_temperature, 
                "heater_on": heater_on, 
                "current_season": current_season_name, 
                "step": step_count,
                "sim_time": current_sim_time,
                "outside_temp": outside_temp
            }
            signed_sensor_data = SecureMessage.sign_message(sensor_data)
            sensor_to_agent_queue.put(signed_sensor_data)

            signed_command = agent_to_sensor_queue.get()
            is_valid, command = SecureMessage.verify_message(signed_command)
            if not is_valid:
                print("[Sensor] ⚠️ Received tampered command! Skipping...")
                continue
            
            if command.get("command") == "budget_exhausted":
                print(f"\n[Sensor] Budget exhaustion signal received.")
                print(f"[Sensor] Terminated at {command['sim_time']:.1f}s (step {command['step']})")
                break
            
            heater_on = command["heater_on"]
            if "dt" in command:
                current_dt = command["dt"]
            if "budget" in command:
                agent_budget = command["budget"]
            
            # Extract metrics
            pragmatic_value = command.get("pragmatic_value", 0.0)
            homeostatic_cost = command.get("homeostatic_cost", 0.0)
            td_error = command.get("td_error", 0.0)
            value_at_state = command.get("value_at_state", 0.0)
            
            # Physics Update
            heat_from_heater = (wandb.config["heating_power"] * current_dt) if heater_on else 0
            heat_loss_or_gain = (outside_temp - room_temperature) * 0.1 * current_dt
            room_temperature += heat_from_heater + heat_loss_or_gain
            
            # Log to WandB
            if hasattr(run, 'log'):
                target_temp = SEASONS_TARGET[current_season_name]
                run.log({
                    "step": step_count, 
                    "sim_time": current_sim_time,
                    "Room Temperature": room_temperature,
                    "Heater On": int(heater_on),
                    "Outside Temp": outside_temp,
                    "Target Temp": target_temp,
                    "Sampling Rate (dt)": current_dt,
                    "Agent Budget": agent_budget,
                    "Pragmatic Value (Extrinsic)": pragmatic_value,
                    "Homeostatic Cost (Comparison)": homeostatic_cost,
                    "TD Error": td_error,
                    "Value at State V(s)": value_at_state,
                    "Step Reward": command.get("step_reward", 0),
                    "Total Rewards": command.get("total_rewards", 0),
                    "Total Penalties": command.get("total_penalties", 0),
                    "Comfort Percentage": command.get("comfort_percentage", 0),
                })

            # Collect history
            target_temp = SEASONS_TARGET[current_season_name]
            history["time"].append(current_sim_time)
            history["room_temp"].append(room_temperature)
            history["target_temp"].append(target_temp)
            history["outside_temp"].append(outside_temp)
            history["heater_on"].append(1 if heater_on else 0)
            history["budget"].append(agent_budget)
            history["rewards"].append(command.get("total_rewards", 0))
            history["penalties"].append(command.get("total_penalties", 0))
            history["pragmatic_value"].append(pragmatic_value)
            history["homeostatic_cost"].append(homeostatic_cost)
            history["td_error"].append(td_error)
            history["value_at_state"].append(value_at_state)
            
            current_sim_time += current_dt
            step_count += 1
            
            if step_count % 20 == 0:
                print(f"\n[Sensor] Step {step_count} | Time: {current_sim_time:.1f}s | Season: {current_season_name}")
                print(f"  Room: {room_temperature:.2f}°C | Outside: {outside_temp:.2f}°C | Target: {target_temp:.1f}°C")
            
            time.sleep(0.1 * current_dt)

        print(f"\n[Sensor] Simulation finished: {current_sim_time:.1f}s ({step_count} steps)")
        
        shutdown_signal = {"command": "simulation_end"}
        signed_shutdown = SecureMessage.sign_message(shutdown_signal)
        sensor_to_agent_queue.put(signed_shutdown)

        # Generate Local Plot
        print("[Sensor] Generating comparison plot...")
        plt.figure(figsize=(14, 12))
        
        # Subplot 1: Temperatures
        plt.subplot(4, 1, 1)
        plt.plot(history["time"], history["room_temp"], label="Room Temp", color="blue")
        plt.plot(history["time"], history["target_temp"], label="Target", color="green", linestyle="--")
        plt.plot(history["time"], history["outside_temp"], label="Outside", color="gray", alpha=0.5)
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Dynamics (Extrinsic TD Learning Agent)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Budget
        plt.subplot(4, 1, 2)
        plt.plot(history["time"], history["budget"], label="Budget", color="gold", linewidth=2)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.ylabel("Budget (€)")
        plt.title("Budget Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Pragmatic Value vs Homeostatic Cost
        plt.subplot(4, 1, 3)
        plt.plot(history["time"], history["pragmatic_value"], label="Pragmatic Value (TD)", color="blue", alpha=0.7)
        plt.plot(history["time"], [-h for h in history["homeostatic_cost"]], label="Negative Homeostatic Cost", color="red", alpha=0.7, linestyle="--")
        plt.ylabel("Value")
        plt.title("Pragmatic Value Comparison: TD-Learned vs Homeostatic")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: TD Learning Progress
        plt.subplot(4, 1, 4)
        # Smooth TD error for visualization
        window = min(20, len(history["td_error"]))
        if window > 0:
            smoothed_td = np.convolve(history["td_error"], np.ones(window)/window, mode='valid')
            td_time = history["time"][:len(smoothed_td)]
            plt.plot(td_time, smoothed_td, label="TD Error (smoothed)", color="purple")
        plt.plot(history["time"], history["value_at_state"], label="V(s)", color="orange", alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.ylabel("Value")
        plt.xlabel("Time (s)")
        plt.title("TD Learning Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = "smart_thermostat_extrinsic_results.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"[Sensor] Plot saved to {save_path}")

    finally:
        if hasattr(run, 'finish'):
            run.finish()
        print("[Sensor] Shutdown complete.")


if __name__ == "__main__":
    print("=" * 70)
    print("SMART THERMOSTAT - EXTRINSIC REWARD (Temporal Difference Learning)")
    print("=" * 70)
    print("\nThis version uses TD Learning to learn a value function V(s)")
    print("for pragmatic value, instead of the homeostatic (temp - target)²")
    print()
    
    sensor_to_agent_queue = Queue()
    agent_to_sensor_queue = Queue()
    
    agent_thread = threading.Thread(target=run_agent, args=(sensor_to_agent_queue, agent_to_sensor_queue))
    sensor_thread = threading.Thread(target=run_sensor, args=(sensor_to_agent_queue, agent_to_sensor_queue))
    
    agent_thread.start()
    sensor_thread.start()
    
    sensor_thread.join()
    agent_thread.join()
    
    print("\n" + "=" * 70)
    print("EXTRINSIC SIMULATION COMPLETE")
    print("=" * 70)
