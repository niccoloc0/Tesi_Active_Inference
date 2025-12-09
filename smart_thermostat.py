import threading
import time
import random
import wandb
from queue import Queue
import numpy as np
import hmac
import hashlib
import json

# --- HMAC Security for Sensor-Agent Communication ---
class SecureMessage:
    """
    HMAC-based message authentication for secure sensor-agent communication.
    Ensures data integrity and authenticity of messages.
    """
    
    # Shared secret key
    SECRET_KEY = b"!T\q!Un'8AL4bpHH"
    
    @staticmethod
    def create_signature(data: dict) -> str:
        """Create HMAC-SHA256 signature for the given data."""
        # Serialize data to JSON with sorted keys for consistent ordering
        message = json.dumps(data, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            SecureMessage.SECRET_KEY,
            message,
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @staticmethod
    def sign_message(data: dict) -> dict:
        """Sign a message by adding HMAC signature."""
        signature = SecureMessage.create_signature(data)
        return {
            "payload": data,
            "signature": signature
        }
    
    @staticmethod
    def verify_message(signed_message: dict) -> tuple[bool, dict]:
        """
        Verify message signature and return (is_valid, payload).
        Returns (False, {}) if verification fails.
        """
        try:
            payload = signed_message.get("payload", {})
            received_signature = signed_message.get("signature", "")
            
            # Compute expected signature
            expected_signature = SecureMessage.create_signature(payload)
            
            # Use constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(received_signature, expected_signature)
            
            if is_valid:
                return True, payload
            else:
                print("[Security] ⚠️ HMAC verification failed! Message may be tampered.")
                return False, {}
        except Exception as e:
            print(f"[Security] ⚠️ Error during verification: {e}")
            return False, {}

# --- Active Inference Agent ---
class ActiveInferenceAgent:
    """
    Active Inference agent implementing:
    - Generative model with Kalman filtering
    - Free energy minimization
    - Epistemic and pragmatic action selection
    - Online learning of dynamics and precision
    """
    
    def __init__(self, initial_budget=None):
        # Configuration
        self.comfort_margin = 1.0
        self.SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
        
        # Budget
        self.budget = initial_budget if initial_budget else random.uniform(50, 100)
        self.cost_per_step = 0.5
        
        # Kalman Filter State: [temperature, temperature_rate]
        self.state_mean = np.array([19.0, 0.0])  # Initial belief
        self.state_cov = np.array([[4.0, 0.0],   # Initial uncertainty
                                     [0.0, 1.0]])
        
        # Observation model: We observe temperature directly
        self.H = np.array([[1.0, 0.0]])  # Observation matrix
        
        # Dynamics model parameters (learned online)
        self.A = np.array([[1.0, 1.0],   # State transition: temp_t+1 = temp_t + rate*dt
                           [0.0, 0.95]]) # Rate decays slightly
        self.B_heater = 1.5  # Effect of heater on temperature
        self.B_outside = 0.1  # Effect of outside temp
        
        # Precision (inverse variance) - learned online
        self.observation_precision = 16.0  # 1/0.25^2 (sensor noise)
        self.process_precision = 4.0       # Process noise precision
        
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
        self.uncertainty_threshold = 2.0  # Variance threshold for epistemic actions
        
        print(f"[Agent] Active Inference initialized with budget: {self.budget:.2f}")
        print(f"[Agent] Initial belief: temp={self.state_mean[0]:.2f}°C, uncertainty={self.state_cov[0,0]:.2f}")
    
    def predict(self, action, dt, outside_temp):
        """
        Prediction step: Use generative model to predict next state
        """
        # Modify dynamics based on dt
        A_dt = np.array([[1.0, dt],
                         [0.0, 0.95]])
        
        # Control input effect
        control_effect = np.array([self.B_heater * dt if action else 0.0, 0.0])
        
        # Outside temperature effect (heat loss/gain)
        outside_effect = np.array([self.B_outside * dt, 0.0])
        
        # Predict mean
        predicted_mean = A_dt @ self.state_mean + control_effect
        # Add outside temperature influence (simplified)
        predicted_mean[0] += outside_effect[0] * (outside_temp - self.state_mean[0])
        
        # Predict covariance (uncertainty grows)
        process_noise = np.eye(2) / self.process_precision
        predicted_cov = A_dt @ self.state_cov @ A_dt.T + process_noise
        
        return predicted_mean, predicted_cov
    
    def update(self, observation, predicted_mean, predicted_cov):
        """
        Update step: Incorporate observation using Bayesian inference
        """
        # Innovation (prediction error)
        innovation = observation - (self.H @ predicted_mean)[0]
        
        # Innovation covariance
        observation_noise = 1.0 / self.observation_precision
        S = (self.H @ predicted_cov @ self.H.T)[0, 0] + observation_noise
        
        # Kalman gain
        K = (predicted_cov @ self.H.T) / S
        K = K.reshape(-1, 1)
        
        # Update belief
        self.state_mean = predicted_mean + (K * innovation).flatten()
        self.state_cov = predicted_cov - K @ self.H @ predicted_cov
        
        # Compute free energy (variational free energy)
        free_energy = 0.5 * (innovation**2 * self.observation_precision + 
                            np.log(2 * np.pi / self.observation_precision))
        
        return innovation, free_energy
    
    def compute_expected_free_energy(self, action, target_temp, predicted_mean, predicted_cov):
        """
        Compute Expected Free Energy (EFE) = Epistemic Value + Pragmatic Value
        """
        # Epistemic value: Information gain (reduction in uncertainty)
        # Higher uncertainty -> higher epistemic value for observing
        epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
        
        # Pragmatic value: Expected distance from target
        predicted_temp = predicted_mean[0]
        pragmatic_cost = (predicted_temp - target_temp) ** 2
        pragmatic_value = -pragmatic_cost
        
        # Expected free energy (lower is better)
        efe = -epistemic_value + pragmatic_cost
        
        return efe, epistemic_value, pragmatic_value
    
    def select_action(self, current_temp, target_temp, current_action, outside_temp, dt):
        """
        Select action by minimizing expected free energy
        """
        # Evaluate both actions: keep heater off vs on
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
        
        # Select action with minimum EFE
        best_idx = np.argmin(efes)
        selected_action = actions[best_idx]
        
        # Store values for logging
        self.epistemic_values.append(epistemic_vals[best_idx])
        self.pragmatic_values.append(pragmatic_vals[best_idx])
        
        return selected_action, epistemic_vals[best_idx], pragmatic_vals[best_idx]
    
    def select_sampling_rate(self):
        """
        Epistemic action: Adjust sampling rate based on uncertainty
        Higher uncertainty -> sample more frequently (lower dt)
        """
        uncertainty = self.state_cov[0, 0]  # Temperature variance
        
        # Map uncertainty to sampling rate
        # High uncertainty -> low dt (frequent sampling)
        # Low uncertainty -> high dt (infrequent sampling)
        if uncertainty > self.uncertainty_threshold:
            dt = self.min_dt  # High uncertainty, sample frequently
        elif uncertainty > self.uncertainty_threshold / 2:
            dt = 1.0  # Medium uncertainty
        elif uncertainty > self.uncertainty_threshold / 4:
            dt = 2.0  # Low uncertainty
        else:
            dt = self.max_dt  # Very low uncertainty, sample infrequently
        
        return dt, uncertainty
    
    def learn_from_error(self, prediction_error, observation):
        """
        Online learning: Update model parameters based on prediction errors
        """
        # Update observation precision (inverse variance)
        # Use exponential moving average
        squared_error = prediction_error ** 2
        estimated_variance = 0.9 / self.observation_precision + 0.1 * squared_error
        self.observation_precision = max(1.0, min(100.0, 1.0 / estimated_variance))
        
        # Update dynamics parameters (simplified gradient descent)
        # In a full implementation, we'd update A and B matrices
        # Here we just adapt the heating effect based on errors
        if len(self.prediction_errors) > 5:
            recent_errors = self.prediction_errors[-5:]
            avg_error = sum(recent_errors) / len(recent_errors)
            # If consistently over-predicting, reduce heating effect
            self.B_heater *= (1.0 - self.dynamics_learning_rate * np.sign(avg_error))
            self.B_heater = max(0.5, min(3.0, self.B_heater))
    
    def step(self, observation, current_action, current_season, outside_temp, dt):
        """
        Single step of active inference
        """
        # 1. Predict next state
        predicted_mean, predicted_cov = self.predict(current_action, dt, outside_temp)
        
        # 2. Update beliefs with observation
        prediction_error, free_energy = self.update(observation, predicted_mean, predicted_cov)
        
        # 3. Learn from prediction error
        self.learn_from_error(prediction_error, observation)
        
        # 4. Store metrics
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        self.free_energy_history.append(free_energy)
        
        # 5. Select action (minimize expected free energy)
        target_temp = self.SEASONS_TARGET[current_season]
        new_action, epistemic_val, pragmatic_val = self.select_action(
            observation, target_temp, current_action, outside_temp, dt
        )
        
        # 6. Select sampling rate (epistemic action)
        new_dt, uncertainty = self.select_sampling_rate()
        
        # 7. Update budget
        self.budget -= self.cost_per_step
        
        # 8. Check budget exhaustion
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
            "budget_exhausted": budget_exhausted
        }

def run_agent(sensor_to_agent_queue, agent_to_sensor_queue):
    """Agent using Active Inference framework."""
    
    # Initialize Active Inference agent
    agent = ActiveInferenceAgent()
    
    # Track previous state
    previous_action = False
    current_dt = 1.0
    step_count = 0
    
    try:
        while True:
            # Get sensor data from queue
            signed_sensor_data = sensor_to_agent_queue.get()
            
            # Verify HMAC signature
            is_valid, sensor_data = SecureMessage.verify_message(signed_sensor_data)
            if not is_valid:
                print("[Agent] ⚠️ Received tampered message from sensor! Skipping...")
                continue
            
            # Check for shutdown signal
            if sensor_data.get("command") == "simulation_end":
                print("\n[Agent] Shutdown signal received from sensor.")
                print(f"[Agent] Final belief: temp={agent.state_mean[0]:.2f}°C, uncertainty={agent.state_cov[0,0]:.3f}")
                print(f"[Agent] Learned observation precision: {agent.observation_precision:.2f}")
                print(f"[Agent] Learned heating effect: {agent.B_heater:.3f}")
                break
            
            observation = sensor_data["room_temperature"]
            current_season = sensor_data["current_season"]
            outside_temp = sensor_data.get("outside_temp", 15.0)
            sim_time = sensor_data.get("sim_time", 0)
            
            # Active inference step
            result = agent.step(observation, previous_action, current_season, outside_temp, current_dt)
            
            # Check for budget exhaustion
            if result.get("budget_exhausted", False):
                print(f"\n{'='*60}")
                print(f"[Agent] 🔴 BUDGET EXHAUSTED at step {step_count}!")
                print(f"[Agent] Survived for {sim_time:.1f} seconds ({step_count} steps)")
                print(f"[Agent] Final belief: temp={agent.state_mean[0]:.2f}°C, uncertainty={agent.state_cov[0,0]:.3f}")
                print(f"[Agent] Learned observation precision: {agent.observation_precision:.2f}")
                print(f"[Agent] Learned heating effect: {agent.B_heater:.3f}")
                print(f"{'='*60}\n")
                
                # Send signed shutdown signal to sensor
                shutdown_command = {"command": "budget_exhausted", "step": step_count, "sim_time": sim_time}
                signed_shutdown = SecureMessage.sign_message(shutdown_command)
                agent_to_sensor_queue.put(signed_shutdown)
                break
            
            # Update for next iteration
            previous_action = result["action"]
            current_dt = result["dt"]
            
            # Determine action label
            if agent.budget <= 0:
                action_taken = "BUDGET_EXHAUSTED"
                previous_action = False
            elif result["action"] and not sensor_data["heater_on"]:
                action_taken = "TURN_ON"
            elif not result["action"] and sensor_data["heater_on"]:
                action_taken = "TURN_OFF"
            else:
                action_taken = "IDLE"
            
            # Verbose logging every 10 steps
            if step_count % 10 == 0 or action_taken in ["TURN_ON", "TURN_OFF"]:
                print(f"\n[Agent] Step {step_count} | Season: {current_season} | Budget: {agent.budget:.1f}")
                print(f"  Observation: {observation:.2f}°C | Belief: {result['belief_mean']:.2f}°C | Error: {abs(observation - result['belief_mean']):.2f}°C")
                print(f"  Uncertainty: {result['uncertainty']:.3f} | Free Energy: {result['free_energy']:.3f}")
                print(f"  Epistemic Value: {result['epistemic_value']:.3f} | Pragmatic Value: {result['pragmatic_value']:.3f}")
                print(f"  Action: {action_taken} | Heater: {'ON' if previous_action else 'OFF'} | Next dt: {current_dt:.1f}s")
            
            # Send command back to sensor
            command = {
                "heater_on": previous_action if agent.budget > 0 else False,
                "action_taken": action_taken,
                "dt": current_dt,
                "budget": agent.budget,
                # Active inference metrics
                "free_energy": result["free_energy"],
                "prediction_error": result["prediction_error"],
                "epistemic_value": result["epistemic_value"],
                "pragmatic_value": result["pragmatic_value"],
                "uncertainty": result["uncertainty"],
                "belief_mean": result["belief_mean"],
                "observation_precision": result["observation_precision"]
            }
            # Sign and send command to sensor
            signed_command = SecureMessage.sign_message(command)
            agent_to_sensor_queue.put(signed_command)
            
            step_count += 1

    except Exception as e:
        print(f"[Agent] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Agent] Terminated.")

# --- Sensor Logic ---
def run_sensor(sensor_to_agent_queue, agent_to_sensor_queue):
    """Sensor simulates the environment and communicates with agent."""
    
    # Initialize WandB
    try:
        run = wandb.init(
            project="Smart Thermostat Simulation",
            config={
                "simulation_duration": 720,
                "steps_per_season": 60, 
                "heating_power": 1.5, 
                "weather_fluctuation": 1.5,
            },
            mode="online" 
        )
    except Exception as e:
        print(f"[Sensor] WandB init failed: {e}. Continuing without logging.")
        class MockWandb:
            config = {
                "simulation_duration": 720, "steps_per_season": 60, "heating_power": 1.5, 
                "weather_fluctuation": 1.5
            }
            def log(self, data): pass
            def finish(self): pass
        run = MockWandb()
        wandb.config = run.config

    SEASONS = {
        "Inverno": {"outside_temp": 0}, "Primavera": {"outside_temp": 15},
        "Estate": {"outside_temp": 35}, "Autunno": {"outside_temp": 12}
    }
    SEASONS_TARGET = {"Inverno": 20.0, "Primavera": 21.0, "Estate": 24.0, "Autunno": 20.5}
    
    season_order = ["Inverno", "Primavera", "Estate", "Autunno"]
    room_temperature = random.uniform(17.5, 19.5)
    heater_on = False
    
    # Simulation State
    current_sim_time = 0
    current_dt = 1.0
    step_count = 0
    agent_budget = 0

    print(f"[Sensor] Starting simulation. Initial temp: {room_temperature:.1f}°C.")

    try:
        while current_sim_time < wandb.config["simulation_duration"]:
            # Determine Season based on Time
            season_idx = int(current_sim_time // 60) % 4
            current_season_name = season_order[season_idx]
            
            season_config = SEASONS[current_season_name]
            outside_temp = season_config["outside_temp"] + random.uniform(-wandb.config["weather_fluctuation"], wandb.config["weather_fluctuation"])
            
            # Send sensor data to agent (signed)
            sensor_data = {
                "room_temperature": room_temperature, 
                "heater_on": heater_on, 
                "current_season": current_season_name, 
                "step": step_count,
                "sim_time": current_sim_time,
                "outside_temp": outside_temp  # Add outside temp for agent's generative model
            }
            signed_sensor_data = SecureMessage.sign_message(sensor_data)
            sensor_to_agent_queue.put(signed_sensor_data)

            # Receive signed command from agent
            signed_command = agent_to_sensor_queue.get()
            
            # Verify HMAC signature
            is_valid, command = SecureMessage.verify_message(signed_command)
            if not is_valid:
                print("[Sensor] ⚠️ Received tampered command from agent! Skipping...")
                continue
            
            # Check for budget exhaustion shutdown
            if command.get("command") == "budget_exhausted":
                print(f"\n[Sensor] Received budget exhaustion signal from agent.")
                print(f"[Sensor] Simulation terminated at {command['sim_time']:.1f}s (step {command['step']})")
                break
            
            heater_on = command["heater_on"]
            
            # Update dt and budget from agent
            if "dt" in command:
                current_dt = command["dt"]
            if "budget" in command:
                agent_budget = command["budget"]
            
            # Extract active inference metrics
            free_energy = command.get("free_energy", 0.0)
            prediction_error = command.get("prediction_error", 0.0)
            epistemic_value = command.get("epistemic_value", 0.0)
            pragmatic_value = command.get("pragmatic_value", 0.0)
            uncertainty = command.get("uncertainty", 0.0)
            belief_mean = command.get("belief_mean", room_temperature)
            observation_precision = command.get("observation_precision", 16.0)
            
            # Physics Update (scaled by dt)
            heat_from_heater = (wandb.config["heating_power"] * current_dt) if heater_on else 0
            heat_loss_or_gain = (outside_temp - room_temperature) * 0.1 * current_dt
            room_temperature += heat_from_heater + heat_loss_or_gain
            
            # Log to WandB
            if hasattr(run, 'log'):
                target_temp = SEASONS_TARGET[current_season_name]
                run.log({
                    "step": step_count, 
                    "sim_time": current_sim_time,
                    "Temperatura Stanza (Reale)": room_temperature,
                    "Heater On": int(heater_on),
                    "Outside Temp": outside_temp,
                    "Target Temp": target_temp,
                    "Sampling Rate (dt)": current_dt,
                    "Agent Budget": agent_budget,
                    # Active Inference Metrics
                    "Free Energy": free_energy,
                    "Prediction Error": prediction_error,
                    "Epistemic Value": epistemic_value,
                    "Pragmatic Value": pragmatic_value,
                    "Uncertainty (Variance)": uncertainty,
                    "Belief Mean": belief_mean,
                    "Observation Precision": observation_precision,
                    "Belief Error": abs(belief_mean - room_temperature)
                })
            
            # Advance time
            current_sim_time += current_dt
            step_count += 1
            
            # Verbose sensor logging every 20 steps
            if step_count % 20 == 0:
                print(f"\n[Sensor] Step {step_count} | Time: {current_sim_time:.1f}s | Season: {current_season_name}")
                print(f"  Room Temp: {room_temperature:.2f}°C | Outside: {outside_temp:.2f}°C | Target: {SEASONS_TARGET[current_season_name]:.1f}°C")
                print(f"  Heater: {'ON' if heater_on else 'OFF'} | dt: {current_dt:.1f}s")
            
            # Sleep proportional to dt
            time.sleep(0.1 * current_dt)

        print(f"\n[Sensor] Simulation finished after {current_sim_time:.1f} sim seconds ({step_count} steps).")
        
        # Send signed shutdown signal to agent
        shutdown_signal = {"command": "simulation_end"}
        signed_shutdown = SecureMessage.sign_message(shutdown_signal)
        sensor_to_agent_queue.put(signed_shutdown)

    finally:
        if hasattr(run, 'finish'):
            run.finish()
        print("[Sensor] Shutdown complete.")

if __name__ == "__main__":
    # Create communication queues
    sensor_to_agent_queue = Queue()
    agent_to_sensor_queue = Queue()
    
    # Create threads
    agent_thread = threading.Thread(target=run_agent, args=(sensor_to_agent_queue, agent_to_sensor_queue))
    sensor_thread = threading.Thread(target=run_sensor, args=(sensor_to_agent_queue, agent_to_sensor_queue))
    
    # Start threads
    agent_thread.start()
    sensor_thread.start()
    
    # Wait for both threads to complete
    sensor_thread.join()
    agent_thread.join()
    
    print("\nSimulation complete. Exiting.")