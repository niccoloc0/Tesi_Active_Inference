"""
Smart Thermostat - EFE Experiments
==================================
Questo script permette di studiare come cambia il comportamento dell'agente
in base a diverse configurazioni dell'Expected Free Energy (EFE).

Modalità disponibili:
- "full": EFE completa = Epistemico + Pragmatico
- "epistemic_only": Solo componente epistemica (guadagno informativo)
- "pragmatic_only": Solo componente pragmatica (raggiungimento obiettivo)

Uso:
    python smart_thermostat_experiments.py --mode full
    python smart_thermostat_experiments.py --mode epistemic_only
    python smart_thermostat_experiments.py --mode pragmatic_only
    python smart_thermostat_experiments.py --run-all  # Esegue tutte le modalità
"""

import threading
import time
import random
import argparse
from queue import Queue
from enum import Enum
import numpy as np
import hmac
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    wandb = None
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- EFE Mode Configuration ---
class EFEMode(Enum):
    FULL = "full"                    # Epistemico + Pragmatico
    EPISTEMIC_ONLY = "epistemic_only"  # Solo Epistemico
    PRAGMATIC_ONLY = "pragmatic_only"  # Solo Pragmatico


@dataclass
class SimulationMetrics:
    """Classe per raccogliere le metriche di una simulazione."""
    mode: str
    temperatures: List[float] = field(default_factory=list)
    target_temps: List[float] = field(default_factory=list)
    beliefs: List[float] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    epistemic_values: List[float] = field(default_factory=list)
    pragmatic_values: List[float] = field(default_factory=list)
    heater_states: List[bool] = field(default_factory=list)
    sampling_rates: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)
    sim_times: List[float] = field(default_factory=list)
    
    # Budget and reward tracking
    budget_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    in_comfort_zone: List[bool] = field(default_factory=list)
    
    # Summary metrics
    total_steps: int = 0
    final_budget: float = 0
    avg_temp_error: float = 0
    avg_uncertainty: float = 0
    heater_on_percentage: float = 0
    total_rewards: float = 0
    total_penalties: float = 0
    comfort_zone_percentage: float = 0


# --- HMAC Security for Sensor-Agent Communication ---
class SecureMessage:
    """HMAC-based message authentication for secure sensor-agent communication."""
    
    SECRET_KEY = b"!T\\q!Un'8AL4bpHH"
    
    @staticmethod
    def create_signature(data: dict) -> str:
        message = json.dumps(data, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            SecureMessage.SECRET_KEY,
            message,
            hashlib.sha256
        ).hexdigest()
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


# --- Active Inference Agent with Configurable EFE ---
class ActiveInferenceAgentExperimental:
    """
    Active Inference agent with configurable EFE computation.
    
    Permette di sperimentare con diverse formulazioni dell'Expected Free Energy:
    - FULL: EFE = -epistemic_value + pragmatic_cost (standard)
    - EPISTEMIC_ONLY: EFE = -epistemic_value (solo riduzione incertezza)
    - PRAGMATIC_ONLY: EFE = pragmatic_cost (solo raggiungimento obiettivo)
    """
    
    def __init__(self, efe_mode: EFEMode = EFEMode.FULL, initial_budget=None,
                 comfort_threshold: float = 2.0, reward_amount: float = 0.3,
                 penalty_amount: float = 0.5, heater_cost: float = 0.2):
        # EFE Configuration
        self.efe_mode = efe_mode
        
        # Configuration
        self.comfort_margin = 1.0
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        # Budget and Reward System
        self.budget = initial_budget if initial_budget else random.uniform(50, 100)
        self.initial_budget = self.budget
        self.cost_per_step = 0.1  # Costo fisso per step (ridotto)
        
        # === NUOVO: Sistema Reward/Penalty basato sulla temperatura ===
        self.comfort_threshold = comfort_threshold  # Soglia: |temp - target| <= threshold
        self.reward_amount = reward_amount          # Soldi guadagnati se in comfort zone
        self.penalty_amount = penalty_amount        # Soldi persi se fuori comfort zone
        self.heater_cost = heater_cost              # Costo per usare il riscaldamento
        
        # Tracking rewards/penalties
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        self.steps_in_comfort = 0
        self.steps_out_comfort = 0
        
        # Kalman Filter State: [temperature, temperature_rate]
        self.state_mean = np.array([19.0, 0.0])
        self.state_cov = np.array([[4.0, 0.0],
                                     [0.0, 1.0]])
        
        # Observation model
        self.H = np.array([[1.0, 0.0]])
        
        # Dynamics model parameters
        self.A = np.array([[1.0, 1.0],
                           [0.0, 0.95]])
        self.B_heater = 1.5
        self.B_outside = 0.1
        
        # Precision
        self.observation_precision = 16.0
        self.process_precision = 4.0
        
        # Learning rates
        self.dynamics_learning_rate = 0.01
        self.precision_learning_rate = 0.05
        
        # History for tracking
        self.prediction_errors = []
        self.free_energy_history = []
        self.epistemic_values = []
        self.pragmatic_values = []
        
        # Epistemic action parameters
        self.min_dt = 0.5
        self.max_dt = 3.0
        self.uncertainty_threshold = 2.0
        
        mode_name = efe_mode.value.replace("_", " ").title()
        print(f"\n{'='*60}")
        print(f"[Agent] Active Inference initialized - Mode: {mode_name}")
        print(f"[Agent] Budget: {self.budget:.2f}")
        print(f"[Agent] Comfort threshold: ±{self.comfort_threshold}°C")
        print(f"[Agent] Reward: +{self.reward_amount} | Penalty: -{self.penalty_amount}")
        print(f"[Agent] Initial belief: temp={self.state_mean[0]:.2f}°C, uncertainty={self.state_cov[0,0]:.2f}")
        print(f"{'='*60}\n")
    
    def predict(self, action, dt, outside_temp):
        """Prediction step using generative model."""
        A_dt = np.array([[1.0, dt],
                         [0.0, 0.95]])
        
        control_effect = np.array([self.B_heater * dt if action else 0.0, 0.0])
        outside_effect = np.array([self.B_outside * dt, 0.0])
        
        predicted_mean = A_dt @ self.state_mean + control_effect
        predicted_mean[0] += outside_effect[0] * (outside_temp - self.state_mean[0])
        
        process_noise = np.eye(2) / self.process_precision
        predicted_cov = A_dt @ self.state_cov @ A_dt.T + process_noise
        
        return predicted_mean, predicted_cov
    
    def update(self, observation, predicted_mean, predicted_cov):
        """Update step: Bayesian inference with observation."""
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
    
    def compute_expected_free_energy(self, action, target_temp, predicted_mean, predicted_cov):
        """
        Compute Expected Free Energy (EFE) based on configured mode.
        """
        # Epistemic value: Information gain (riduzione incertezza)
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        
        # Pragmatic value: Distance from target
        predicted_temp = predicted_mean[0]
        pragmatic_cost = (predicted_temp - target_temp) ** 2
        pragmatic_value = -pragmatic_cost
        
        # Compute EFE based on mode
        if self.efe_mode == EFEMode.FULL:
            # Standard: considera sia epistemic che pragmatic
            efe = -epistemic_value + pragmatic_cost
        elif self.efe_mode == EFEMode.EPISTEMIC_ONLY:
            # Solo epistemic: l'agente cerca solo di ridurre l'incertezza
            # Non si preoccupa di raggiungere la temperatura target
            efe = -epistemic_value
        elif self.efe_mode == EFEMode.PRAGMATIC_ONLY:
            # Solo pragmatic: l'agente cerca solo di raggiungere il target
            # Non considera l'incertezza nelle sue decisioni
            efe = pragmatic_cost
        else:
            efe = -epistemic_value + pragmatic_cost  # Default to full
        
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        """Select action by minimizing expected free energy."""
        actions = [False, True]
        efes = []
        epistemic_vals = []
        pragmatic_vals = []
        
        for action in actions:
            pred_mean, pred_cov = self.predict(action, dt, outside_temp)
            efe, epist, prag = self.compute_expected_free_energy(
                action, target_temp, pred_mean, pred_cov
            )
            efes.append(efe)
            epistemic_vals.append(epist)
            pragmatic_vals.append(prag)
        
        best_idx = np.argmin(efes)
        selected_action = actions[best_idx]
        
        self.epistemic_values.append(epistemic_vals[best_idx])
        self.pragmatic_values.append(pragmatic_vals[best_idx])
        
        return selected_action, epistemic_vals[best_idx], pragmatic_vals[best_idx]
    
    def select_sampling_rate(self):
        """Epistemic action: Adjust sampling rate based on uncertainty."""
        uncertainty = self.state_cov[0, 0]
        
        # Solo in modalità EPISTEMIC_ONLY o FULL consideriamo l'incertezza
        if self.efe_mode == EFEMode.PRAGMATIC_ONLY:
            # In modalità pragmatica, usa un dt fisso
            dt = 1.0
        else:
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
        """Online learning: Update model parameters."""
        squared_error = prediction_error ** 2
        estimated_variance = 0.9 / self.observation_precision + 0.1 * squared_error
        self.observation_precision = max(1.0, min(100.0, 1.0 / estimated_variance))
        
        if len(self.prediction_errors) > 5:
            recent_errors = self.prediction_errors[-5:]
            avg_error = sum(recent_errors) / len(recent_errors)
            self.B_heater *= (1.0 - self.dynamics_learning_rate * np.sign(avg_error))
            self.B_heater = max(0.5, min(3.0, self.B_heater))
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        """
        Single step of active inference with reward/penalty system.
        
        Budget changes:
        - Fixed cost per step (small)
        - Heater cost if heater is on
        - REWARD if temperature is within comfort threshold
        - PENALTY if temperature is outside comfort threshold
        """
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        
        self.learn_from_error(prediction_error, observation)
        
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        self.free_energy_history.append(free_energy)
        
        target_temp = self.SEASONS_TARGET[current_season]
        new_action, epistemic_val, pragmatic_val = self.select_action(
            observation, target_temp, current_action, outside_temp, dt
        )
        
        new_dt, uncertainty = self.select_sampling_rate()
        
        # === SISTEMA REWARD/PENALTY ===
        temp_error = abs(observation - target_temp)
        in_comfort_zone = temp_error <= self.comfort_threshold
        
        # Costo fisso per step
        step_cost = self.cost_per_step
        
        # Costo riscaldamento (se acceso)
        if current_action:  # Se il riscaldamento era acceso
            step_cost += self.heater_cost
        
        # Reward o Penalty in base alla temperatura
        step_reward = 0.0
        if in_comfort_zone:
            # REWARD: temperatura entro la soglia comfort
            step_reward = self.reward_amount
            self.total_rewards += step_reward
            self.steps_in_comfort += 1
        else:
            # PENALTY: temperatura fuori dalla soglia comfort
            step_reward = -self.penalty_amount
            self.total_penalties += self.penalty_amount
            self.steps_out_comfort += 1
        
        # Aggiorna budget: -costi +reward
        net_change = step_reward - step_cost
        self.budget += net_change
        
        budget_exhausted = self.budget <= 0
        
        return {
            "action": new_action,
            "dt": new_dt,
            "free_energy": free_energy,
            "prediction_error": prediction_error,
            "epistemic_value": epistemic_val,
            "pragmatic_value": pragmatic_val,
            "uncertainty": uncertainty,
            "belief_mean": self.state_mean[0],
            "observation_precision": self.observation_precision,
            "budget_exhausted": budget_exhausted,
            # Nuove metriche reward
            "in_comfort_zone": in_comfort_zone,
            "step_reward": step_reward,
            "step_cost": step_cost,
            "net_change": net_change,
            "temp_error": temp_error
        }


def run_experiment(efe_mode: EFEMode, initial_budget: float = 75.0, 
                   simulation_duration: int = 300, verbose: bool = True) -> SimulationMetrics:
    """
    Esegue una singola simulazione con la modalità EFE specificata.
    
    Args:
        efe_mode: Modalità di calcolo EFE
        initial_budget: Budget iniziale dell'agente
        simulation_duration: Durata simulazione in secondi
        verbose: Se stampare i log dettagliati
    
    Returns:
        SimulationMetrics con tutti i dati raccolti
    """
    metrics = SimulationMetrics(mode=efe_mode.value)
    
    # Initialize agent
    agent = ActiveInferenceAgentExperimental(efe_mode=efe_mode, initial_budget=initial_budget)
    
    # Simulation constants
    SEASONS = {
        "Inverno": {"outside_temp": 0}, 
        "Primavera": {"outside_temp": 15},
        "Estate": {"outside_temp": 35}, 
        "Autunno": {"outside_temp": 12}
    }
    SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
    season_order = ["Inverno", "Primavera", "Estate", "Autunno"]
    
    # Initial state
    room_temperature = random.uniform(17.5, 19.5)
    heater_on = False
    current_sim_time = 0
    current_dt = 1.0
    step_count = 0
    weather_fluctuation = 1.5
    heating_power = 1.5
    
    print(f"[Experiment] Starting {efe_mode.value} simulation...")
    print(f"[Experiment] Initial temp: {room_temperature:.2f}°C, Budget: {initial_budget:.2f}")

    # Initialize WandB
    if wandb:
        try:
            wandb.init(
                project="Smart Thermostat Experiments",
                group="comparison_experiments",
                name=f"experiment_{efe_mode.value}",
                config={
                    "mode": efe_mode.value,
                    "initial_budget": initial_budget,
                    "simulation_duration": simulation_duration,
                    "seasons_config": SEASONS,
                    "seasons_target": SEASONS_TARGET
                },
                reinit=True,
                mode="online"  # Use 'offline' if no internet
            )
        except Exception as e:
            print(f"WandB init failed: {e}")

    while current_sim_time < simulation_duration and agent.budget > 0:
        # Determine season
        season_idx = int(current_sim_time // 60) % 4
        current_season_name = season_order[season_idx]
        
        season_config = SEASONS[current_season_name]
        outside_temp = season_config["outside_temp"] + random.uniform(-weather_fluctuation, weather_fluctuation)
        target_temp = SEASONS_TARGET[current_season_name]
        
        # Agent step
        result = agent.step(room_temperature, heater_on, current_season_name, outside_temp, current_dt)
        
        # Collect metrics
        metrics.temperatures.append(room_temperature)
        metrics.target_temps.append(target_temp)
        metrics.beliefs.append(result["belief_mean"])
        metrics.uncertainties.append(result["uncertainty"])
        metrics.free_energies.append(result["free_energy"])
        metrics.epistemic_values.append(result["epistemic_value"])
        metrics.pragmatic_values.append(result["pragmatic_value"])
        metrics.heater_states.append(result["action"])
        metrics.sampling_rates.append(result["dt"])
        metrics.prediction_errors.append(result["prediction_error"])
        metrics.sim_times.append(current_sim_time)
        
        # Collect reward/budget metrics
        metrics.budget_history.append(agent.budget)
        metrics.reward_history.append(result["step_reward"])
        metrics.in_comfort_zone.append(result["in_comfort_zone"])
        
        # Update state
        heater_on = result["action"]
        current_dt = result["dt"]
        
        # Physics update
        heat_from_heater = (heating_power * current_dt) if heater_on else 0
        heat_loss_or_gain = (outside_temp - room_temperature) * 0.1 * current_dt
        room_temperature += heat_from_heater + heat_loss_or_gain
        
        # Log to WandB
        if wandb and wandb.run:
            wandb.log({
                "time": current_sim_time,
                "room_temp": room_temperature,
                "target_temp": target_temp,
                "outside_temp": outside_temp,
                "heater_on": int(heater_on),
                "budget": agent.budget,
                "step_reward": result["step_reward"],
                "total_rewards": agent.total_rewards,
                "total_penalties": agent.total_penalties,
                "free_energy": result["free_energy"],
                "epistemic_value": result["epistemic_value"],
                "pragmatic_value": result["pragmatic_value"],
                "uncertainty": result["uncertainty"],
                "belief_mean": result["belief_mean"],
                "in_comfort_zone": int(result["in_comfort_zone"])
            })

        # Verbose logging with reward info
        if verbose and step_count % 20 == 0:
            mode_label = efe_mode.value.replace("_", " ").upper()
            comfort_icon = "✅" if result["in_comfort_zone"] else "❌"
            print(f"  [{mode_label}] Step {step_count} | Time: {current_sim_time:.1f}s | "
                  f"Temp: {room_temperature:.2f}°C | Target: {target_temp:.1f}°C | "
                  f"Heater: {'ON' if heater_on else 'OFF'} | Budget: {agent.budget:.1f} {comfort_icon}")
        
        current_sim_time += current_dt
        step_count += 1
    
    # Compute summary metrics
    metrics.total_steps = step_count
    metrics.final_budget = agent.budget
    metrics.total_rewards = agent.total_rewards
    metrics.total_penalties = agent.total_penalties
    
    if metrics.temperatures:
        temp_errors = [abs(t - tgt) for t, tgt in zip(metrics.temperatures, metrics.target_temps)]
        metrics.avg_temp_error = sum(temp_errors) / len(temp_errors)
        metrics.avg_uncertainty = sum(metrics.uncertainties) / len(metrics.uncertainties)
        metrics.heater_on_percentage = (sum(metrics.heater_states) / step_count * 100) if step_count > 0 else 0
        metrics.comfort_zone_percentage = (metrics.in_comfort_zone.count(True) / step_count * 100) if step_count > 0 else 0
    
    # Calculate budget change
    budget_change = metrics.final_budget - initial_budget
    budget_change_str = f"+{budget_change:.2f}" if budget_change >= 0 else f"{budget_change:.2f}"

    if wandb and wandb.run:
        wandb.finish()
    
    print(f"\n[Experiment] {efe_mode.value} completed!")
    print(f"  Total steps: {metrics.total_steps}")
    print(f"  Budget: {initial_budget:.2f} → {metrics.final_budget:.2f} ({budget_change_str})")
    print(f"  Total Rewards: +{metrics.total_rewards:.2f} | Penalties: -{metrics.total_penalties:.2f}")
    print(f"  Comfort Zone: {metrics.comfort_zone_percentage:.1f}% of time")
    print(f"  Avg temp error: {metrics.avg_temp_error:.2f}°C")
    print(f"  Heater ON: {metrics.heater_on_percentage:.1f}%")
    
    return metrics


def plot_comparison(results: Dict[str, SimulationMetrics], save_path: str = None):
    """
    Genera grafici comparativi tra le diverse modalità EFE.
    Include ora anche evoluzione del budget e metriche reward/penalty.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Confronto Modalità EFE - Active Inference Thermostat\n(Sistema Reward/Penalty)", 
                 fontsize=14, fontweight='bold')
    
    colors = {
        'full': '#2ecc71',           # Verde
        'epistemic_only': '#3498db',  # Blu
        'pragmatic_only': '#e74c3c'   # Rosso
    }
    
    labels = {
        'full': 'Completa (Epist. + Prag.)',
        'epistemic_only': 'Solo Epistemico',
        'pragmatic_only': 'Solo Pragmatico'
    }
    
    # 1. Temperature over time
    ax1 = axes[0, 0]
    for mode, metrics in results.items():
        ax1.plot(metrics.sim_times, metrics.temperatures, 
                 color=colors[mode], label=labels[mode], alpha=0.8)
    if results:
        first_metrics = list(results.values())[0]
        ax1.plot(first_metrics.sim_times, first_metrics.target_temps, 
                 'k--', label='Target', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Temperatura Stanza')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. BUDGET EVOLUTION (NEW!)
    ax2 = axes[0, 1]
    for mode, metrics in results.items():
        if metrics.budget_history:
            ax2.plot(metrics.sim_times, metrics.budget_history,
                     color=colors[mode], label=labels[mode], alpha=0.8, linewidth=2)
    ax2.axhline(y=75, color='gray', linestyle='--', alpha=0.5, label='Budget Iniziale')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Bancarotta')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Budget (€)')
    ax2.set_title('💰 Evoluzione Budget')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Free Energy over time
    ax3 = axes[1, 0]
    for mode, metrics in results.items():
        window = min(10, len(metrics.free_energies))
        if window > 0:
            smoothed = np.convolve(metrics.free_energies, np.ones(window)/window, mode='valid')
            time_adjusted = metrics.sim_times[:len(smoothed)]
            ax3.plot(time_adjusted, smoothed,
                     color=colors[mode], label=labels[mode], alpha=0.8)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Free Energy')
    ax3.set_title('Energia Libera Variazionale (Media Mobile)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty over time
    ax4 = axes[1, 1]
    for mode, metrics in results.items():
        ax4.plot(metrics.sim_times, metrics.uncertainties,
                 color=colors[mode], label=labels[mode], alpha=0.8)
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylabel('Incertezza (Varianza)')
    ax4.set_title('Incertezza dello Stato')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Comfort Zone percentage bar chart
    ax5 = axes[2, 0]
    modes = list(results.keys())
    comfort_percentages = [results[m].comfort_zone_percentage for m in modes]
    bars = ax5.bar(range(len(modes)), comfort_percentages, 
                   color=[colors[m] for m in modes], alpha=0.8)
    ax5.set_xticks(range(len(modes)))
    ax5.set_xticklabels([labels[m] for m in modes], rotation=15, ha='right')
    ax5.set_ylabel('Comfort Zone (%)')
    ax5.set_title('✅ Percentuale Tempo in Comfort Zone')
    ax5.set_ylim(0, 100)
    for bar, pct in zip(bars, comfort_percentages):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{pct:.1f}%', ha='center', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Rewards vs Penalties comparison
    ax6 = axes[2, 1]
    x_positions = np.arange(len(results))
    width = 0.35
    
    rewards = [m.total_rewards for m in results.values()]
    penalties = [m.total_penalties for m in results.values()]
    
    ax6.bar(x_positions - width/2, rewards, width, label='Rewards (+)', color='#27ae60', alpha=0.8)
    ax6.bar(x_positions + width/2, penalties, width, label='Penalties (-)', color='#c0392b', alpha=0.8)
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels([labels[m] for m in results.keys()], rotation=15, ha='right')
    ax6.set_ylabel('Importo (€)')
    ax6.set_title('💵 Rewards vs Penalties Totali')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Sampling rate distribution
    ax7 = axes[3, 0]
    for mode, metrics in results.items():
        ax7.hist(metrics.sampling_rates, bins=20, alpha=0.5, 
                 color=colors[mode], label=labels[mode])
    ax7.set_xlabel('Sampling Rate (dt)')
    ax7.set_ylabel('Frequenza')
    ax7.set_title('Distribuzione Sampling Rate')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary comparison with budget focus
    ax8 = axes[3, 1]
    metrics_names = ['Budget\nFinale', 'Comfort\nZone %', 'Err. Temp\n(°C)', 'Heater\nON %']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, (mode, metrics) in enumerate(results.items()):
        values = [
            metrics.final_budget,
            metrics.comfort_zone_percentage,
            metrics.avg_temp_error * 10,  # Scale up for visibility
            metrics.heater_on_percentage
        ]
        ax8.bar(x + i * width, values, width, label=labels[mode], color=colors[mode], alpha=0.8)
    
    ax8.set_xticks(x + width)
    ax8.set_xticklabels(metrics_names)
    ax8.set_ylabel('Valore')
    ax8.set_title('📊 Confronto Metriche Principali')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[Plot] Grafico salvato in: {save_path}")
    
    plt.show()
    return fig


def run_all_experiments(initial_budget: float = 75.0, simulation_duration: int = 300, 
                        verbose: bool = True) -> Dict[str, SimulationMetrics]:
    """
    Esegue tutte e tre le modalità EFE e genera grafici comparativi.
    """
    print("\n" + "="*70)
    print("ESPERIMENTO: Confronto Modalità EFE")
    print("="*70)
    print(f"Budget iniziale: {initial_budget}")
    print(f"Durata simulazione: {simulation_duration}s")
    print("Modalità: FULL, EPISTEMIC_ONLY, PRAGMATIC_ONLY")
    print("="*70)
    
    results = {}
    
    # Set same random seed for each experiment for fair comparison
    for mode in [EFEMode.FULL, EFEMode.EPISTEMIC_ONLY, EFEMode.PRAGMATIC_ONLY]:
        random.seed(42)  # Reset seed for reproducibility
        np.random.seed(42)
        results[mode.value] = run_experiment(
            efe_mode=mode,
            initial_budget=initial_budget,
            simulation_duration=simulation_duration,
            verbose=verbose
        )
    
    # Print summary table
    print("\n" + "="*90)
    print("RIEPILOGO COMPARATIVO - Sistema Reward/Penalty")
    print("="*90)
    print(f"{'Modalità':<20} {'Budget Finale':<14} {'Rewards':<12} {'Penalties':<12} {'Comfort%':<12} {'Err.Temp':<10}")
    print("-"*90)
    
    for mode, metrics in results.items():
        mode_label = mode.replace("_", " ").title()
        budget_change = metrics.final_budget - initial_budget
        budget_str = f"{metrics.final_budget:.1f} ({'+' if budget_change >= 0 else ''}{budget_change:.1f})"
        print(f"{mode_label:<20} {budget_str:<14} +{metrics.total_rewards:<11.1f} -{metrics.total_penalties:<11.1f} "
              f"{metrics.comfort_zone_percentage:<12.1f} {metrics.avg_temp_error:<10.2f}")
    
    print("="*90)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Esperimenti EFE per Smart Thermostat con Active Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python smart_thermostat_experiments.py --mode full
  python smart_thermostat_experiments.py --mode epistemic_only
  python smart_thermostat_experiments.py --mode pragmatic_only
  python smart_thermostat_experiments.py --run-all --plot
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['full', 'epistemic_only', 'pragmatic_only'],
                        help='Modalità EFE da usare')
    parser.add_argument('--run-all', action='store_true',
                        help='Esegue tutte le modalità e confronta i risultati')
    parser.add_argument('--budget', type=float, default=75.0,
                        help='Budget iniziale (default: 75.0)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Durata simulazione in secondi (default: 300)')
    parser.add_argument('--plot', action='store_true',
                        help='Genera grafici comparativi (solo con --run-all)')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Percorso per salvare il grafico')
    parser.add_argument('--quiet', action='store_true',
                        help='Riduce output verboso')
    
    args = parser.parse_args()
    
    if args.run_all:
        results = run_all_experiments(
            initial_budget=args.budget,
            simulation_duration=args.duration,
            verbose=not args.quiet
        )
        if args.plot or args.save_plot:
            save_path = args.save_plot or "efe_comparison.png"
            plot_comparison(results, save_path=save_path)
    elif args.mode:
        mode = EFEMode(args.mode)
        random.seed(42)
        np.random.seed(42)
        metrics = run_experiment(
            efe_mode=mode,
            initial_budget=args.budget,
            simulation_duration=args.duration,
            verbose=not args.quiet
        )
    else:
        parser.print_help()
        print("\n⚠️  Specifica --mode o --run-all per eseguire la simulazione")


if __name__ == "__main__":
    main()
